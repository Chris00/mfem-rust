use autocxx::prelude::*;
use cxx::{UniquePtr, let_cxx_string};
use mfem_sys::{self as ffi, mfem};
use std::{
    convert::{From, TryFrom}, fmt, io, marker::PhantomData, path::Path, pin::Pin, ptr, slice
};

#[derive(Debug)]
pub enum Error {
    // TODO: be more precise.
    IO(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IO(e) => write!(f, "IO({e})"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Error::IO(value)
    }
}

macro_rules! with_lifetime {
    ($ty: ident <>) => {
        $ty
    };
    ($ty: ident <$l: lifetime>) => {
        $ty < 'a > // Macros are not hygienic w.r.t. lifetimes.
    };
}

/// Define a type owning a MFEM pointer and related types:
/// - `$tybase` for the "lower" type on which to implement
///    shared operations (on & or &mut),
/// - `$tyref` to hold immutable references to the pointer (but
///   not freeing it).
macro_rules! new_struct_with_ref {
    ($(#[$doc: meta])* $ty: ident < $($l: lifetime)? >,
        $tybase: ident, //
        $tyref: ident,
        $inner: ty
    ) => {
        new_struct_with_ref!(base $tybase <$($l)?>, $inner);
        new_struct_with_ref!(ref $tyref <$($l)?>, $tybase, $inner);
        new_struct_with_ref!(main $(#[$doc])* $ty <$($l)?>, $tybase, $inner);
        new_struct_with_ref!(main-constructor $ty <$($l)?>, $tybase, $inner);
    };
    (subclass $(#[$doc: meta])* $ty: ident < $($l: lifetime)? >,
        $tybase: ident, // Base class or slice.
        $tyref: ident, // FIXME: needed?
        $inner: ty, $inner_base: ty
    ) => {
        // In the case of a subclass, we suppose that the "base class"
        // $tybase has already been defined.
        new_struct_with_ref!(ref $tyref <$($l)?>, $tybase, $inner_base);
        new_struct_with_ref!(main $(#[$doc])* $ty <$($l)?>,
            $tybase, $inner_base);
        // Constructor from $inner
        #[allow(dead_code)]
        impl $(<$l>)? $ty $(<$l>)? {
            /// Takes ownership of the pointee by `x` and wrap it into `Self`.
            #[inline]
            unsafe fn from_raw(ptr: *mut $inner_base) -> Self {
                let tyref = unsafe { $tybase::from_raw(ptr) };
                $ty { inner: tyref }
            }
            #[inline]
            fn from_unique_ptr(ptr: UniquePtr<$inner_base>) -> Self {
                unsafe { Self::from_raw(ptr.into_raw()) }
            }
            #[inline]
            fn emplace<N>(n: N) -> Self
            where N: New<Output = $inner> {
                unsafe {
                    Self::from_raw(UniquePtr::emplace(n)
                        .into_raw() as *mut $inner_base)
                }
            }
            #[inline]
            fn as_ref_mfem(&self) -> & $inner {
                let x: & $inner_base = self.inner.as_ref();
                // $inner is a subclass of $inner_base but the type
                // makes sure that this instance is $inner, so this
                // cast is safe.
                unsafe { &*(x as *const _ as *const $inner) }
            }
        }
    };
    (base $tybase: ident < $($l: lifetime)? >, $inner: ty) => {
        // 1. We hold a mutable pointer, it must only be mutably
        //    de-referenced if we hold a `&mut $tybase`.
        // 2. The pointee is pinned.
        // 3. Immutable such values must always be returned behind
        //    a shared reference (& $tybase).  Since this does not
        //    implement `Copy`, the user will not be able to take
        //    ownership (and then violate the immutability).
        // 4. Basically, there should be no way to take ownership
        //    of such values outside this library.
        #[must_use]
        pub struct $tybase $(<$l>)? {
            inner: *mut $inner,
            marker: PhantomData<$(&$l)? ()>,
        }
        impl $(<$l>)? AsRef<$inner> for $tybase $(<$l>)? {
            fn as_ref(&self) -> & $inner {
                unsafe { self.inner.as_ref().unwrap() }
            }
        }
        #[allow(dead_code)]
        impl $(<$l>)? $tybase $(<$l>)?{
            #[inline]
            unsafe fn from_raw(raw: *mut $inner) -> Self {
                Self { inner: raw,  marker: PhantomData }
            }
            #[inline]
            fn as_const_ptr(&self) -> *const $inner {
                self.inner
            }
            #[inline]
            fn pin_mut(&mut self) -> Pin<&mut $inner> {
                unsafe {
                    let ptr: &mut $inner = &mut *self.inner;
                    Pin::new_unchecked(ptr)
                }
            }
        }
    };
    (ref $tyref: ident <$($l: lifetime)?>, $tybase: ident, $inner: ty) => {
        /// Immutable reference.  This has a single lifetime that
        /// accounts for all dependencies.
        #[must_use]
        #[allow(non_camel_case_types)]
        pub struct $tyref<'a> {
            // The intention is to hold `&'a $inner` but a $tybase is
            // used instead to be able to deref to `$tybase`.
            inner: with_lifetime!($tybase <$($l)?>),
            marker: PhantomData<&'a $inner>,
        }
        impl<'a> std::ops::Deref for $tyref<'a> {
            type Target = with_lifetime!($tybase <$($l)?>);
            fn deref(&self) -> &Self::Target { &self.inner }
        }
        #[allow(dead_code)]
        impl<'a> $tyref<'a> {
            /// Safety: the value pointed by `x` must live at least
            /// for `'a`.  Make sure you add the right lifetime constraints.
            #[inline]
            unsafe fn from_raw(x: *const $inner) -> Self {
                let x = x as *mut _; // This type must protect from mutating.
                let slice = unsafe { $tybase::from_raw(x) };
                Self { inner: slice,  marker: PhantomData }
            }
            #[inline]
            fn from_ref(x: &'a $inner) -> Self {
                unsafe { Self::from_raw(x as *const _ as *mut _) }
            }
        }
    };
    (main $(#[$doc: meta])* $ty: ident < $($l: lifetime)? >,
        $tybase: ident,
        $inner: ty
    ) => {
        // The pointer is here OWNED.  Assume it was created from a
        // `UniquePtr<$tybase>`.
        $(#[$doc])*
        #[must_use]
        pub struct $ty $(<$l>)? {
            inner: $tybase $(<$l>)?,
        }
        impl $(<$l>)? std::ops::Drop for $ty $(<$l>)? {
            fn drop(&mut self) {
                let ptr = unsafe { UniquePtr::from_raw(self.inner.inner) };
                drop(ptr);
            }
        }
        impl $(<$l>)? std::ops::Deref for $ty $(<$l>)? {
            type Target = $tybase $(<$l>)?;
            fn deref(&self) -> &Self::Target { &self.inner }
        }
        impl $(<$l>)? std::ops::DerefMut for $ty $(<$l>)? {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }
    };
    (main-constructor $ty: ident < $($l: lifetime)? >,
        $tybase: ident,
        $inner: ty
    ) => {
        #[allow(dead_code)]
        impl $(<$l>)? $ty $(<$l>)? {
            /// Takes ownership of the pointee by `x` and wrap it into `Self`.
            #[inline]
            unsafe fn from_raw(ptr: *mut $inner) -> Self {
                let tyref = unsafe { $tybase::from_raw(ptr) };
                $ty { inner: tyref }
            }
            #[inline]
            fn from_unique_ptr(ptr: UniquePtr<$inner>) -> Self {
                unsafe { Self::from_raw(ptr.into_raw()) }
            }
        }
    };
    (emplace $ty: ident < $($l: lifetime)? >, $inner: ty) => {
        impl $(<$l>)? $ty $(<$l>)? {
            #[inline]
            fn emplace<N>(n: N) -> Self
            where N: New<Output = $inner> {
                Self::from_unique_ptr(UniquePtr::emplace(n))
            }
        }
    };
}

// FIXME: Reverse: Array<T> if trait on T.
new_struct_with_ref!(
    ArrayInt<>,
    ArrayIntSlice,
    ArrayIntRef,
    ffi::ArrayInt);

impl Clone for ArrayInt {
    fn clone(&self) -> Self {
        Self::from_unique_ptr(ffi::Arrayint_copy(self.as_ref()))
    }
}

impl std::ops::Deref for ArrayIntSlice {
    type Target = [i32];
    fn deref(&self) -> &[i32] {
        let a = self.as_ref();
        let data = ffi::Arrayint_GetData(a);
        // autocxx::c_int is declared "transparent".
        let data = data as *const std::ffi::c_int;
        unsafe { slice::from_raw_parts(data, self.len()) }
    }
}

impl std::ops::DerefMut for ArrayIntSlice {
    fn deref_mut(&mut self) -> &mut [i32] {
        let len = self.len();
        let data = ffi::Arrayint_GetDataMut(self.pin_mut());
        let data = data as *mut std::ffi::c_int;
        unsafe { slice::from_raw_parts_mut(data, len) }
    }
}

impl Default for ArrayInt {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayInt {
    pub fn new() -> Self {
        Self::from_unique_ptr(ffi::Arrayint_with_len(c_int(0)))
    }

    pub fn with_len(len: usize) -> Self {
        let len = len.try_into().expect("Valid i32 len");
        Self::from_unique_ptr(ffi::Arrayint_with_len(c_int(len)))
    }
}

impl ArrayIntSlice {
    #[inline]
    pub fn len(&self) -> usize {
        let len: i32 = ffi::Arrayint_len(self.as_ref()).into();
        debug_assert!(len >= 0);
        len as usize
    }
}

new_struct_with_ref!(
    /// Vector of [`f64`] numbers.
    Vector<>,
    VectorSlice,
    VectorRef,
    mfem::Vector);

impl std::ops::Deref for VectorSlice {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        let len = self.len();
        let data = self.as_ref().GetData();
        unsafe { slice::from_raw_parts(data, len) }
    }
}

impl std::ops::DerefMut for VectorSlice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = self.len();
        let data = self.as_ref().GetData();
        unsafe { slice::from_raw_parts_mut(data, len) }
    }
}

impl VectorSlice {
    #[inline]
    pub fn len(&self) -> usize {
        let l: i32 = self.as_ref().Size().into();
        l as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}


/// Algorithm for [`Mesh::uniform_refinement`].
pub enum RefAlgo {
    /// Algorithm "A".
    ///
    /// Currently used only for pure tetrahedral meshes.
    /// Produces elements with better quality
    A = 0,
    /// Algorithm "B".
    B = 1,
}

pub struct Mesh {
    inner: UniquePtr<mfem::Mesh>,
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<mfem::Mesh> for Mesh {
    fn as_ref(&self) -> &mfem::Mesh {
        self.inner.as_ref().unwrap()
    }
}

impl Mesh {
    /// Return a new empty mesh.
    pub fn new() -> Self {
        Self {
            inner: UniquePtr::emplace(mfem::Mesh::new1()),
        }
    }

    /// Return a mesh read from file in MFEM, Netgen, or VTK format.
    #[doc(alias = "LoadFromFile")]
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        // TODO: capture errors
        let generate_edges = c_int(1);
        let refine = c_int(1);
        let fix_orientation = true;
        let_cxx_string!(
            filename = path.as_ref().as_os_str().as_encoded_bytes()
        );

        // mfem will abort the program in case of error.  Try to
        // detect common problems.
        if !std::fs::exists(path)? {
            Err(io::Error::from(io::ErrorKind::NotFound))?;
        }
        let mesh = mfem::Mesh::LoadFromFile(
            &filename,
            generate_edges,
            refine,
            fix_orientation,
        );
        Ok(Self {
            inner: UniquePtr::emplace(mesh),
        })
    }

    #[doc(alias = "Dimension")]
    pub fn dimension(&self) -> i32 {
        self.as_ref().Dimension().into()
    }

    #[doc(alias = "GetNE")]
    pub fn get_num_elems(&self) -> usize {
        self.as_ref().GetNE().0 as usize
    }

    /// Return the attribute of boundary element `i`.
    pub fn get_bdr_attribute(&self, i: i32) -> i32 {
        self.as_ref().GetBdrAttribute(c_int(i)).into()
    }

    pub fn bdr_attributes(&self) -> ArrayIntRef<'_> {
        // Return a pointer to a public attribute.
        let mesh = self.inner.as_ref().unwrap();
        ArrayIntRef::from_ref(ffi::Mesh_bdr_attributes(mesh))
    }

    // TODO: optional argument?
    pub fn uniform_refinement(&mut self, ref_algo: RefAlgo) {
        self.inner
            .pin_mut()
            .UniformRefinement1(c_int(ref_algo as i32))
    }

    // TODO: overloaded, invent meaningful names
    pub fn get_nodes(&self) -> Option<GridFunctionRef<'_>> {
        // GetNodes() returns an internal structure.  Since `self`
        // holds a `UniquePtr`, the memory pointed to does not move
        // (as is the rule for C++).  Thus the pointer `gf` does not
        // change location either.
        let gf: *const mfem::GridFunction =
            self.inner.as_ref().unwrap().GetNodes2();
        if gf.is_null() {
            None
        } else {
            Some(unsafe { GridFunctionRef::from_raw(gf as *mut _) })
        }
    }

    /// Save the mesh to the file `path`.  The given precision will be
    /// used for ASCII output.
    pub fn save(&self) -> MeshSave<'_> {
        MeshSave {
            mesh: self,
            precision: 16,
        }
    }
}

pub struct MeshSave<'a> {
    mesh: &'a Mesh,
    precision: i32,
}

impl MeshSave<'_> {
    // TODO: error?
    pub fn to_file(&self, path: impl AsRef<Path>) {
        let_cxx_string!(
            filename = path.as_ref().as_os_str().as_encoded_bytes()
        );
        self.mesh.inner.Save(&filename, c_int(self.precision));
    }

    pub fn precision(&self, p: i32) -> Self {
        Self {
            mesh: self.mesh,
            precision: p,
        }
    }
}

pub enum BasisType {
    //Invalid = -1,  // Removed and replaced by an Option<BasisType>
    /// Open type.
    GaussLegendre = 0,
    /// Closed type.
    GaussLobatto = 1,
    /// Bernstein polynomials.
    Positive = 2,
    /// Nodes: x_i = (i+1)/(n+1), i=0,...,n-1
    OpenUniform = 3,
    /// Nodes: x_i = i/(n-1),     i=0,...,n-1.
    ClosedUniform = 4,
    /// Nodes: x_i = (i+1/2)/n,   i=0,...,n-1.
    OpenHalfUniform = 5,
    /// Serendipity basis (squares / cubes).
    Serendipity = 6,
    /// Closed GaussLegendre.
    ClosedGL = 7,
    /// Integrated GLL indicator functions.
    IntegratedGLL = 8,
    // NumBasisTypes = 9, see test right after.
}

#[cfg(test)]
#[test]
fn test_num_basis_types() {
    assert_eq!(ffi::NumBasisTypes, 9);
}

impl TryFrom<c_int> for BasisType {
    type Error = ();

    fn try_from(c_int(value): c_int) -> Result<Self, ()> {
        match value {
            0 => Ok(BasisType::GaussLegendre),
            1 => Ok(BasisType::GaussLobatto),
            2 => Ok(BasisType::Positive),
            3 => Ok(BasisType::OpenUniform),
            4 => Ok(BasisType::ClosedUniform),
            5 => Ok(BasisType::OpenHalfUniform),
            6 => Ok(BasisType::Serendipity),
            7 => Ok(BasisType::ClosedGL),
            8 => Ok(BasisType::IntegratedGLL),
            _ => Err(()),
        }
    }
}

new_struct_with_ref!(
    /// Collection of finite elements from the same family in multiple
    /// dimensions.
    FiniteElementCollection<>,
    FiniteElementCollectionBase,
    FiniteElementCollectionRef,
    mfem::FiniteElementCollection);

impl FiniteElementCollectionBase {
    pub fn get_name(&self) -> String {
        let cstr = self.as_ref().Name();
        let cstr = unsafe { std::ffi::CStr::from_ptr(cstr) };
        cstr.to_owned().into_string().expect("Name must be ASCII")
    }

    pub fn get_range_dim(&self, dim: i32) -> i32 {
        self.as_ref().GetRangeDim(c_int(dim)).into()
    }
}

new_struct_with_ref!(subclass
    /// Arbitrary order H1-conforming (continuous) finite element.
    /// Implements [`FiniteElementCollection`].
    #[allow(non_camel_case_types)]
    H1_FECollection<>,
    FiniteElementCollectionBase,
    H1_FECollectionRef,  // FIXME: needed?
    mfem::H1_FECollection, // is a subclass of:
    mfem::FiniteElementCollection);

impl H1_FECollection {
    /// Return a H1-conforming (continuous) finite elements with order
    /// `p`, dimension `dim` and basis type `btype`.
    pub fn new(p: i32, dim: i32, btype: BasisType) -> Self {
        let h1 = mfem::H1_FECollection::new(
            c_int(p),
            c_int(dim),
            c_int(btype as i32),
        );
        H1_FECollection::emplace(h1)
    }

    /// Return a H1-conforming (continuous) finite elements with
    /// positive basis functions with order `p` and dimension `dim`.
    #[doc(alias = "H1Pos_FECollection")]
    pub fn pos(p: i32, dim: i32) -> Self {
        // https://docs.mfem.org/html/fe__coll_8hpp_source.html#l00305
        Self::new(p, dim, BasisType::Positive)
    }

    /// Return a H1-conforming (continuous) serendipity finite elements
    /// with order `p` and dimension `dim`.  Current implementation
    /// works in 2D only; 3D version is in development.
    #[doc(alias = "H1Ser_FECollection")]
    pub fn ser(p: i32, dim: i32) -> Self {
        Self::new(p, dim, BasisType::Serendipity)
    }

    /// Return a "H^{1/2}-conforming" trace finite elements with order
    /// `p` and dimension `dim` defined on the interface between mesh
    /// elements (faces,edges,vertices); these are the trace FEs of
    /// the H1-conforming FEs.
    #[doc(alias = "H1_Trace_FECollection")]
    pub fn trace(p: i32, dim: i32, btype: BasisType) -> Self {
        Self::new(p, dim - 1, btype)
    }

    pub fn get_basis_type(&self) -> Option<BasisType> {
        self.as_ref_mfem().GetBasisType().try_into().ok()
    }
}

pub enum Ordering {
    ByNodes = mfem::Ordering_Type::byNODES as isize,
    ByVdim = mfem::Ordering_Type::byVDIM as isize,
}

impl From<Ordering> for mfem::Ordering_Type {
    fn from(value: Ordering) -> Self {
        match value {
            Ordering::ByNodes => mfem::Ordering_Type::byNODES,
            Ordering::ByVdim => mfem::Ordering_Type::byVDIM,
        }
    }
}

new_struct_with_ref!(
    /// Responsible for providing FEM view of the mesh, mainly managing
    /// the set of degrees of freedom.
    FiniteElementSpace<'deps>, // Single lifetime for deps.
    FiniteElementSpaceBase,
    FiniteElementSpaceRef,
    ffi::FES);
//new_struct_with_ref!(emplace FiniteElementSpace<'deps>, ffi::FES);

impl<'a> FiniteElementSpace<'a> {
    pub fn new<'mesh: 'a, 'fec: 'a>(
        mesh: &'mesh Mesh,
        fec: &'fec FiniteElementCollectionBase,
        vdim: i32,
        ordering: Ordering,
    ) -> Self {
        let fes = ffi::FES_new(
            mesh.inner.as_ref().expect("Initialized Mesh"),
            fec.as_ref(),
            c_int(vdim),
            ordering.into());
        Self::from_unique_ptr(fes)
    }

    pub fn get_true_vsize(&self) -> i32 {
        ffi::FES_GetTrueVSize(self.as_ref()).into()
    }

    pub fn get_essential_true_dofs(
        &self,
        bdr_attr_is_ess: &ArrayInt,
        ess_tdof_list: &mut ArrayInt,
        component: Option<usize>,
    ) {
        ffi::FES_GetEssentialTrueDofs(
            self.as_ref(),
            bdr_attr_is_ess.as_ref(),
            ess_tdof_list.inner.pin_mut(),
            c_int(component.map(|c| c as i32).unwrap_or(-1)),
        )
    }
}


new_struct_with_ref!(
    /// Represent a [`Vector`] with associated FE space.
    GridFunction<'fes>,
    GridFunctionBase,
    GridFunctionRef,
    mfem::GridFunction);
new_struct_with_ref!(emplace GridFunction<'fes>, mfem::GridFunction);

impl<'fes> GridFunction<'fes> {
    // XXX Can `fes` be mutated?
    // Hopefully so because it is shared with e.g. `Linearform`
    // XXX can you pass a different `fes` than the one for `Linearform`?
    pub fn with_fes(fes: &'fes FiniteElementSpace) -> Self {
        let gf = unsafe {
            mfem::GridFunction::new2(fes.as_const_ptr() as *mut _)
        };
        GridFunction::emplace(gf)
    }
}

impl GridFunctionBase<'_> {
    pub fn fec<'g>(&self) -> FiniteElementCollectionRef<'g>
    where Self: 'g {
        // Safety: `self` is temporarily borrowed mutably because
        // `OwnFEC()` requires it (it gives access to the internal
        // field) but the return value CANNOT be mutated.
        unsafe {
            let gf = self.inner.as_mut().unwrap();
            let raw = Pin::new_unchecked(gf).OwnFEC();
            FiniteElementCollectionRef::from_raw(raw)
        }
    }

    pub fn slice_mut(&mut self) -> &mut [f64] {
        let v = self.pin_mut().GetTrueVector1();
        let len: i32 = v.Size().into();
        let data = v.GetData();
        unsafe { slice::from_raw_parts_mut(data, len as usize) }
    }

    pub fn fill(&mut self, c: f64) {
        self.slice_mut().fill(c);
    }
}


pub trait Coefficient: autocxx::PinMut<mfem::Coefficient> {
    // TODO
}

pub struct ConstantCoefficient {
    inner: UniquePtr<mfem::ConstantCoefficient>,
}

impl AsRef<mfem::Coefficient> for ConstantCoefficient {
    fn as_ref(&self) -> &mfem::Coefficient {
        self.inner.as_ref().unwrap().as_ref()
    }
}

impl autocxx::PinMut<mfem::Coefficient> for ConstantCoefficient {
    fn pin_mut(&mut self) -> Pin<&mut mfem::Coefficient> {
        unsafe {
            let c = self.inner.pin_mut().get_unchecked_mut();
            let c = c as *mut mfem::ConstantCoefficient;
            let c = c as *mut mfem::Coefficient;
            let c = &mut *c;
            Pin::new_unchecked(c)
        }
    }
}

impl Coefficient for ConstantCoefficient {}

impl ConstantCoefficient {
    pub fn new(c: f64) -> Self {
        let cc = mfem::ConstantCoefficient::new(c);
        Self { inner: UniquePtr::emplace(cc) }
    }
}

/// Common capabilities of `LinarFormIntegrator`s.
pub trait LinearFormIntegrator: Into<*mut ffi::LFI> {
    // Since linear form integrators are taken by value by
    // [`LinearForm::new`], there is no way to play the same as above
    // (de-referencing to a Base struct).

    // TODO
}

pub struct DomainLFIntegrator<'a> {
    inner: UniquePtr<mfem::DomainLFIntegrator>,
    marker: PhantomData<&'a ()>,
}

impl From<DomainLFIntegrator<'_>> for *mut ffi::LFI {
    fn from(value: DomainLFIntegrator<'_>) -> Self {
        let dlfi = value.inner.into_raw();
        dlfi as *mut ffi::LFI
    }
}

impl LinearFormIntegrator for DomainLFIntegrator<'_> {}

impl<'coeff> DomainLFIntegrator<'coeff> {
    pub fn new<QF>(qf: &'coeff mut QF, a: i32, b: i32) -> Self
    where QF: Coefficient {
        // Safety: The result does not seem to take ownership of `qf`.
        let qf = qf.pin_mut();
        let lfi = mfem::DomainLFIntegrator::new(qf, c_int(a), c_int(b));
        let inner = UniquePtr::emplace(lfi);
        Self { inner,  marker: PhantomData }
    }
}

/// Vector with associated FE space and [`LinearFormIntegrator`]s.
pub struct LinearForm<'a> {
    inner: UniquePtr<mfem::LinearForm>,
    marker: PhantomData<&'a ()>,
}

impl<'a> LinearForm<'a> {
    pub fn new(fes: &FiniteElementSpaceBase<'a>) -> Self {
        // XXX Safety: the underlying `fes` is not modified so can take it
        // by ref only.
        // TODO: Submit a PR to mfem.
        let lf = unsafe {
            mfem::LinearForm::new1(fes.as_const_ptr() as *mut _)
        };
        let inner = UniquePtr::emplace(lf);
        Self { inner,  marker: PhantomData }
    }

    pub fn fe_space(&self) -> FiniteElementSpaceRef<'a> {
        let raw = self.inner.FESpace1();
        unsafe { FiniteElementSpaceRef::from_raw(raw) }
    }

    pub fn assemble(&mut self) {
        self.inner.pin_mut().Assemble();
    }

    pub fn add_domain_integrator<Lfi>(&mut self, lfi: Lfi)
    where Lfi: LinearFormIntegrator + 'a {
        // The linear form "takes ownership of `lfi`".
        let lfi = lfi.into();
        unsafe {
            self.inner.pin_mut().AddDomainIntegrator(lfi);
        }
    }
}


pub trait BilinearFormIntegrator: Into<*mut ffi::BFI> {
    // Since bilinear form integrators are taken by value by
    // [`BilinearForm::new`], there is no way to play the same as above
    // (de-referencing to a Base struct).
}

pub struct DiffusionIntegrator<'a> {
    inner: UniquePtr<mfem::DiffusionIntegrator>,
    marker: PhantomData<&'a ()>,
}

impl BilinearFormIntegrator for DiffusionIntegrator<'_> {}

impl From<DiffusionIntegrator<'_>> for *mut ffi::BFI {
    fn from(value: DiffusionIntegrator) -> Self {
        let di: *mut mfem::DiffusionIntegrator = value.inner.into_raw();
        di as *mut ffi::BFI
    }
}

impl<'coeff> DiffusionIntegrator<'coeff> {
    pub fn new<QF>(qf: &'coeff mut QF) -> Self
    where QF: Coefficient {
        let qf = qf.pin_mut();
        let ir: *const mfem::IntegrationRule = ptr::null();
        let bfi = unsafe { mfem::DiffusionIntegrator::new1(qf, ir) };
        Self { inner: UniquePtr::emplace(bfi), marker: PhantomData }
    }
}


pub struct BilinearForm<'a> {
    inner: UniquePtr<mfem::BilinearForm>,
    marker: PhantomData<&'a ()>,
}

impl<'a> BilinearForm<'a> {
    pub fn new(fes: &FiniteElementSpaceBase<'a>) -> Self {
        // XXX Safety: the underlying `fes` is not modified so can take it
        // by ref only.
        // TODO: Submit a PR to mfem.
        let lf = unsafe {
            mfem::BilinearForm::new2(fes.as_const_ptr() as *mut _)
        };
        let inner = UniquePtr::emplace(lf);
        Self { inner,  marker: PhantomData }
    }

    /// Adds new Domain Integrator.
    pub fn add_domain_integrator<Bfi>(&mut self, bfi: Bfi)
    where Bfi: BilinearFormIntegrator + 'a {
        // Doc says: "Assumes ownership of `bfi`".
        let bfi = bfi.into();
        unsafe {
            self.inner.pin_mut().AddDomainIntegrator(bfi);
        }
    }

    /// Assembles the form i.e. sums over all domain/bdr integrators.
    pub fn assemble(&mut self, skip_zeros: bool) {
        let skip_zeros = if skip_zeros {1} else {0}; // TODO: option?
        self.inner.pin_mut().Assemble(c_int(skip_zeros));
    }

    pub fn form_linear_system(
        &self,
        ess_tdof_list: &ArrayInt,
        x: &VectorSlice,
        b: &VectorSlice,
        a_mat: &mut OperatorHandle,
        x_vec: &mut Vector,
        b_vec: &mut Vector,
    ) {
        todo!()
    }
}


pub struct OperatorHandle {
    // TODO
}

impl OperatorHandle {
    pub fn new() -> Self {
        todo!()
    }

    pub fn height(&self) -> usize {
        todo!()
    }

    pub fn get_type(&self) -> String {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_from_file1() {
        let mesh = Mesh::from_file("does-not-exists");
        assert!(mesh.is_err());
    }

    //// Not possible for now: the C++ code calls "abort".
    // #[test]
    // fn test_mesh_from_file2() {
    //     use std::{fs::File, io::Write};
    //     // Create a local file with the wrong format.
    //     let mut fh = File::create("m.mesh").unwrap();
    //     writeln!(fh, "Intentionally not a mesh file!");
    //     drop(fh);
    //     let mesh = Mesh::from_file("m.mesh");
    //     assert!(mesh.is_err());
    // }
}
