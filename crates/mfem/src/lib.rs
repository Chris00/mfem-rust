use autocxx::prelude::*;
use cxx::{let_cxx_string, UniquePtr};
use mfem_sys as mfem;
use std::{
    convert::{From, TryFrom}, fmt, i32, io, marker::PhantomData, path::Path, pin::Pin, ptr, slice
};

#[derive(Debug)]
pub enum Error {
    // TODO: be more precise.
    IO(io::Error),
    ConversionFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IO(e) => write!(f, "IO({e})"),
            Self::ConversionFailed => write!(f, "Conversion failed"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Error::IO(value)
    }
}

/// Wrap mfem::$ty into $tymfem.  This `Deref` to this base and not
/// inherit the C++ operations (we want to define our own to have a
/// sleek Rusty interface).
///
/// You want to define a base type if one of the two situations occurs:
/// - The type is returned by reference from functions;
/// - Multiple sub-classes exist (the subclass relationship is handled
///   by `Deref` to the base class — hence returning a ref to this type).
macro_rules! wrap_mfem_base {
    ($(#[$doc: meta])* $tymfem: ident, mfem $ty: ident) => {
        // Wrap the `mfem` value in order not to inherit `mfem`
        // methods (we want to redefine them with Rust conventions).
        //
        // When a C++ method return a pointer, we want be able to
        // interpret it as a (mutable) reference to this wrapper
        // struct.  The memory of this type is controlled by C++,
        // it should never be accessible as a Rust value.
        #[repr(transparent)]
        #[allow(non_camel_case_types)]
        $(#[$doc])*
        pub struct $tymfem {
            inner: mfem:: $ty,
        }
        #[allow(dead_code)]
        impl $tymfem {
            /// To wrap a *non-owned* pointer.
            ///
            /// *Safety*: Make sure the lifetime is constrained not to
            /// exceed the life of the pointed value.
            #[inline]
            unsafe fn ref_of_mfem<'a>(ptr: *const mfem::$ty) -> &'a Self {
                debug_assert!(! ptr.is_null());
                // `ptr` is a pointer to `$tymfem`.  Since the
                // representation of `Self` is transparent, `ptr` can
                // be seen as a reference to `Self`.
                &*(ptr as *const $tymfem)
            }
            #[inline]
            unsafe fn mut_of_mfem<'a>(ptr: *mut mfem::$ty) -> &'a mut Self {
                debug_assert!(! ptr.is_null());
                &mut *(ptr as *mut $tymfem)
            }
            #[inline]
            fn as_mfem(&self) -> & mfem:: $ty {
                &self.inner
            }
            // One cannot move (e.g. `Clone`) the returned value.
            #[inline]
            fn as_mut_mfem(&mut self) -> Pin<&mut mfem:: $ty> {
                unsafe { Pin::new_unchecked(&mut self.inner) }
            }
        }
    };
}

/// Define an struct owning the value `mfem::$ty`.
/// If a `$tybase` is given, it `Deref` to that type.
macro_rules! wrap_mfem {
    ($(#[$doc: meta])* $ty: ident < $($l: lifetime)? >,
        base $(#[$docmfem: meta])* $tymfem: ident
    ) => {
        wrap_mfem!(defstruct $(#[$doc])* $ty <$($l)?>);
        wrap_mfem_base!($(#[$docmfem])* $tymfem, mfem $ty);
        // Deref to the base.
        impl $(<$l>)? ::std::ops::Deref for $ty $(<$l>)? {
            type Target = $tymfem;

            #[inline]
            fn deref(&self) -> &Self::Target {
                let ptr: & mfem:: $ty = self.inner.as_ref().unwrap();
                unsafe { $tymfem::ref_of_mfem(ptr) }
            }
        }
        impl $(<$l>)? ::std::ops::DerefMut for $ty $(<$l>)? {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                let ptr: *mut mfem::$ty = self.inner.as_mut_ptr();
                unsafe { $tymfem::mut_of_mfem(ptr) }
            }
        }
    };

    ($(#[$doc: meta])* $ty: ident < $($l: lifetime)? >) => {
        wrap_mfem!(defstruct $(#[$doc])* $ty <$($l)?>);
        // No base type.  Provide the facilities to access the
        // underlying value.
        #[allow(dead_code)]
        impl $(<$l>)? $ty $(<$l>)? {
            #[inline]
            fn as_mfem(&self) -> & mfem:: $ty {
                &self.inner.as_ref().unwrap()
            }
            // One cannot move (e.g. `Clone`) the returned value.
            #[inline]
            fn as_mut_mfem(&mut self) -> Pin<&mut mfem:: $ty> {
                self.inner.pin_mut()
            }
        }
    };

    (defstruct $(#[$doc: meta])* $ty: ident < $($l: lifetime)? >) => {
        // This is basically a `UniquePtr` but with a nicer name and a
        // possible lifetime parameter to track dependencies.
        $(#[$doc])*
        #[must_use]
        #[repr(transparent)]
        pub struct $ty $(<$l>)? {
            inner: UniquePtr<mfem:: $ty>,
            marker: PhantomData<$(&$l)? ()>,
        }
        #[allow(dead_code)]
        impl $(<$l>)? $ty $(<$l>)? {
            /// Return an *owned* value, from the pointer `ptr`.
            #[inline]
            fn from_unique_ptr(ptr: UniquePtr<mfem:: $ty>) -> Self {
                Self { inner: ptr,  marker: PhantomData }
            }
            /// Consumes `self`, releasing its ownership, and returns
            /// the pointer (to the C++ heap).
            fn into_raw(self) -> *mut mfem::$ty {
                self.inner.into_raw()
            }
        }
    };
}

macro_rules! wrap_mfem_emplace {
    ($ty: ident < $($l: lifetime)? >) => {
        impl $(<$l>)? $ty $(<$l>)? {
            #[inline]
            fn emplace<N>(n: N) -> Self
            where N: New<Output = mfem:: $ty> {
                let inner = UniquePtr::emplace(n);
                Self { inner,  marker: PhantomData }
            }
        }
    };
}

/// Empty trait simply to be able to write a safety check as a bound.
#[allow(dead_code)]
trait Subclass {}

macro_rules! subclass {
    ($tysub0: ident < $($l: lifetime)? >, base $tysub: ident,
        $ty0: ident, $ty: ident
    ) => {
        // Use the fact that `autocxx` implements the `AsRef`
        // relationship for sub-classes to make sure it is safe.
        impl Subclass for $tysub
        where mfem::$tysub0 : AsRef<mfem::$ty0> {}
        subclass!(unsafe $tysub0 <$($l)?>, base $tysub, $ty0, $ty);
    };
    (unsafe $tysub0: ident < $($l: lifetime)? >, base $tysub: ident,
        $ty0: ident, $ty: ident
    ) => {
        subclass!(unsafe base $tysub, $ty);

        subclass!(unsafe $tysub0 <$($l)?>, into $ty0);
    };
    (unsafe base $tysub: ident, $ty: ident) => {
        impl ::std::ops::Deref for $tysub {
            type Target = $ty;

            #[inline]
            fn deref(&self) -> & Self::Target {
                unsafe {
                    // Safety: Since, on the C++ side, mfem::$subty is
                    // a subclass of mfem::$ty, the pointers can be
                    // cast.  As the representation is transparent, we
                    // can do the same with the wrapped types.
                    &*(self as *const _ as *const $ty)
                    // Remark: `$ty` implements `Drop`.  But since we
                    // only return a reference to it, it should not be
                    // triggered.
                }
            }
        }
        impl ::std::ops::DerefMut for $tysub {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe {
                    // Safety: see `Deref`.
                    &mut *(self as *mut _ as *mut $ty)
                }
            }
        }
    };
    (unsafe $tysub0: ident < $($l: lifetime)? >, into $ty0: ident) => {
        // For owned values, also define `Into`.
        impl $(<$l>)? ::std::convert::Into<$ty0> for $tysub0 $(<$l>)? {
            #[inline]
            fn into(self) -> $ty0 {
                unsafe {
                    // Since our types are transparent and pointers
                    // mfem::$tysub0 can be cast to mfem::$ty0, one can
                    // cast unique pointers.
                    ::std::mem::transmute::<Self, $ty0>(self)
                }
            }
        }
    };
    // For an owned only value (no "base" type), the `Deref` to the
    // base value of the parent class is different.
    (owned $tysub0: ident < $($l: lifetime)? >,
        into $ty0: ident, $ty: ident
    ) => {
        subclass!(unsafe $tysub0 <$($l)?>, into $ty0);
        subclass!(owned $tysub0 <$($l)?>, $ty);
    };
    (owned $tysub0: ident < $($l: lifetime)? >, $ty: ident) => {
        impl $(<$l>)? ::std::ops::Deref for $tysub0 $(<$l>)? {
            type Target = $ty;

            #[inline]
            fn deref(&self) -> & Self::Target {
                let ptr: & mfem::$tysub0 = self.inner.as_ref().unwrap();
                unsafe { &*(ptr as *const mfem::$tysub0 as *const $ty) }
            }
        }
        impl $(<$l>)? ::std::ops::DerefMut for $tysub0 $(<$l>)? {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                let ptr: *mut mfem::$tysub0 = self.inner.as_mut_ptr();
                unsafe { &mut *(ptr as *mut mfem::$tysub0 as *mut $ty) }
            }
        }
    }
}

// FIXME: Reverse: Array<T> if trait on T.
wrap_mfem!(
    /// An array of `i32`.
    ArrayInt<>,
    base
    /// Underlying data of [`ArrayInt`].
    AArrayInt);

impl Clone for ArrayInt {
    fn clone(&self) -> Self {
        Self::from_unique_ptr(mfem::ArrayInt_copy(self.as_mfem()))
    }
}

impl std::ops::Deref for AArrayInt {
    type Target = [i32];

    #[inline]
    fn deref(&self) -> &[i32] {
        let len = self.len();
        let data = mfem::ArrayInt_GetData(self.as_mfem());
        // autocxx::c_int is declared "transparent".
        let data = data as *const std::ffi::c_int;
        unsafe { slice::from_raw_parts(data, len) }
    }
}

impl std::ops::DerefMut for AArrayInt {
    #[inline]
    fn deref_mut(&mut self) -> &mut [i32] {
        let len = self.len();
        let data = mfem::ArrayInt_GetDataMut(self.as_mut_mfem());
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
        Self::from_unique_ptr(mfem::ArrayInt_with_len(c_int(0)))
    }

    pub fn with_len(len: usize) -> Self {
        let len = len.try_into().expect("Valid i32 len");
        Self::from_unique_ptr(mfem::ArrayInt_with_len(c_int(len)))
    }
}

impl AArrayInt {
    #[inline]
    pub fn len(&self) -> usize {
        let len: i32 = mfem::ArrayInt_len(self.as_mfem()).into();
        debug_assert!(len >= 0);
        len as usize
    }
}

wrap_mfem!(
    /// Vector of [`f64`] numbers.
    Vector<>,
    base
    /// Base type to which all structs that can be considered as vectors
    /// [`Deref`][::std::ops::Deref].
    AVector);
wrap_mfem_emplace!(Vector<>);

impl std::ops::Deref for AVector {
    type Target = [f64];

    #[inline]
    fn deref(&self) -> &Self::Target {
        let len = self.len();
        if len == 0 {
            &[]
        } else {
            let data = self.as_mfem().GetData();
            unsafe { slice::from_raw_parts(data, len) }
        }
    }
}

impl std::ops::DerefMut for AVector {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = self.len();
        if len == 0 {
            &mut []
        } else {
            let data = self.as_mut_mfem().GetData();
            unsafe { slice::from_raw_parts_mut(data, len) }
        }
    }
}

impl Default for Vector {
    fn default() -> Self {
        Self::new()
    }
}

impl Vector {
    /// Return a new, empty, vector.
    pub fn new() -> Self {
        Vector::emplace(mfem::Vector::new())
    }

    pub fn with_len(len: usize) -> Self {
        if len > std::i32::MAX as usize {
            panic!("mfem::Vector::with_len: size {len} too large");
        }
        Vector::emplace(mfem::Vector::new3(c_int(len as i32)))
    }
}

impl AVector {
    #[inline]
    pub fn len(&self) -> usize {
        let l: i32 = self.as_mfem().Size().into();
        l as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub use mfem::Operator_Type as OperatorType;

wrap_mfem_base!(Operator, mfem Operator);

impl Operator {
    pub fn height(&self) -> usize {
        let h = self.as_mfem().Height();
        debug_assert!(h >= 0);
        h as usize
    }

    pub fn width(&self) -> usize {
        let h = self.as_mfem().Width();
        debug_assert!(h >= 0);
        h as usize
    }

    pub fn get_type(&self) -> OperatorType {
        self.as_mfem().GetType()
    }

    /// This is a C++ virtual method.  One must only make it visible
    /// in types for which it is implemented in C++.
    unsafe fn recover_fem_solution(
        &mut self,
        from: &AVector,
        b: &AVector,
        x: &mut AVector,
    ) {
        unsafe {
            self.as_mut_mfem().RecoverFEMSolution(
                from.as_mfem(),
                b.as_mfem(),
                x.as_mut_mfem())
        }
    }
}

wrap_mfem!(
    /// A matrix.
    Matrix<>,
    base
    /// Matrix "slices" (many types [`Deref`][std::ops::Deref] to this one).
    AMatrix);

// XXX Ok to do with an abstract type?
subclass!(unsafe base AMatrix, Operator);


/// Algorithm for [`AMesh::uniform_refinement`].
pub enum RefAlgo {
    /// Algorithm "A".
    ///
    /// Currently used only for pure tetrahedral meshes.
    /// Produces elements with better quality
    A = 0,
    /// Algorithm "B".
    B = 1,
}

wrap_mfem!(Mesh<>, base AMesh);
wrap_mfem_emplace!(Mesh<>);

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesh {
    /// Return a new empty mesh.
    pub fn new() -> Self {
        Self::emplace(mfem::Mesh::new1())
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
        Ok(Self::emplace(mesh))
    }
}

impl AMesh {
    #[doc(alias = "Dimension")]
    pub fn dimension(&self) -> i32 {
        self.as_mfem().Dimension().into()
    }

    #[doc(alias = "GetNE")]
    pub fn get_num_elems(&self) -> usize {
        self.as_mfem().GetNE().0 as usize
    }

    /// Return the attribute of boundary element `i`.
    pub fn get_bdr_attribute(&self, i: i32) -> i32 {
        self.as_mfem().GetBdrAttribute(c_int(i)).into()
    }

    pub fn bdr_attributes(&self) -> &AArrayInt {
        // Return a pointer to a public attribute.
        let mesh = mfem::Mesh_bdr_attributes(self.as_mfem());
        unsafe { AArrayInt::ref_of_mfem(mesh) }
    }

    // TODO: optional argument?
    pub fn uniform_refinement(&mut self, ref_algo: RefAlgo) {
        self.as_mut_mfem()
            .UniformRefinement1(c_int(ref_algo as i32))
    }

    // TODO: overloaded, invent meaningful names
    pub fn get_nodes(&self) -> Option<&AGridFunction> {
        // GetNodes() returns an internal structure.  Since `self`
        // holds a `UniquePtr`, the memory pointed to does not move
        // (as is the rule for C++).  Thus the pointer `gf` does not
        // change location either.
        let gf: *const mfem::GridFunction = self.as_mfem().GetNodes2();
        if gf.is_null() {
            None
        } else {
            Some(unsafe { AGridFunction::ref_of_mfem(gf) })
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
    mesh: &'a AMesh,
    precision: i32,
}

impl MeshSave<'_> {
    // TODO: error?
    // FIXME: This is slower than the C++ version.
    // Rewriting in Rust may be nice (also for errors) but not a priority.
    // See Mesh::Printer.
    pub fn to_file(&self, path: impl AsRef<Path>) {
        let_cxx_string!(
            filename = path.as_ref().as_os_str().as_encoded_bytes()
        );
        self.mesh.as_mfem().Save(&filename, c_int(self.precision));
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
    assert_eq!(mfem::NumBasisTypes, 9);
}

impl TryFrom<c_int> for BasisType {
    type Error = i32;

    /// Try to convert the value into a [`BasisType`].  If it fails,
    /// it returns the number as an error.
    fn try_from(c_int(value): c_int) -> Result<Self, Self::Error> {
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
            _ => Err(value),
        }
    }
}

wrap_mfem!(
    /// Collection of finite elements from the same family in multiple
    /// dimensions.
    FiniteElementCollection<>,
    base AFiniteElementCollection);

impl AFiniteElementCollection {
    pub fn get_name(&self) -> String {
        let cstr = self.as_mfem().Name();
        let cstr = unsafe { std::ffi::CStr::from_ptr(cstr) };
        cstr.to_owned().into_string().expect("Name must be ASCII")
    }

    pub fn get_range_dim(&self, dim: i32) -> i32 {
        self.as_mfem().GetRangeDim(c_int(dim)).into()
    }
}

wrap_mfem!(
    /// Arbitrary order H1-conforming (continuous) finite element.
    /// Implements [`FiniteElementCollection`].
    #[allow(non_camel_case_types)]
    H1_FECollection<>,
    base AH1_FECollection);
wrap_mfem_emplace!(H1_FECollection<>);
subclass!(
    H1_FECollection<>,         base AH1_FECollection,
    FiniteElementCollection, AFiniteElementCollection);

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
}

impl AH1_FECollection {
    pub fn get_basis_type(&self) -> Option<BasisType> {
        self.as_mfem().GetBasisType().try_into().ok()
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

wrap_mfem!(
    /// Responsible for providing FEM view of the mesh, mainly managing
    /// the set of degrees of freedom.
    FiniteElementSpace<'deps>, // Single lifetime for deps.
    base AFiniteElementSpace);

impl<'a> FiniteElementSpace<'a> {
    pub fn new<'mesh: 'a, 'fec: 'a>(
        mesh: &'mesh Mesh,
        fec: &'fec AFiniteElementCollection,
        vdim: i32,
        ordering: Ordering,
    ) -> Self {
        let fes = mfem::FES_new(
            mesh.as_mfem(),
            fec.as_mfem(),
            c_int(vdim),
            ordering.into(),
        );
        Self::from_unique_ptr(fes)
    }

    pub fn get_true_vsize(&self) -> i32 {
        mfem::FES_GetTrueVSize(self.as_mfem()).into()
    }

    pub fn get_essential_true_dofs(
        &self,
        bdr_attr_is_ess: &ArrayInt,
        ess_tdof_list: &mut ArrayInt,
        component: Option<usize>,
    ) {
        mfem::FES_GetEssentialTrueDofs(
            self.as_mfem(),
            bdr_attr_is_ess.as_mfem(),
            ess_tdof_list.as_mut_mfem(),
            c_int(component.map(|c| c as i32).unwrap_or(-1)),
        )
    }
}

wrap_mfem!(
    /// Represent a [`Vector`] with associated FE space.
    GridFunction<'fes>,
    base AGridFunction);
wrap_mfem_emplace!(GridFunction<'fes>);
subclass!(GridFunction<'fes>, base AGridFunction, Vector, AVector);

impl<'fes> GridFunction<'fes> {
    // XXX Can `fes` be mutated?
    // Hopefully so because it is shared with e.g. `Linearform`
    // XXX can you pass a different `fes` than the one for `Linearform`?
    #[inline]
    pub fn with_fes(fes: &'fes FiniteElementSpace) -> Self {
        let gf = unsafe {
            mfem::GridFunction::new2(fes.as_mfem() as *const _ as *mut _)
        };
        GridFunction::emplace(gf)
    }
}

impl AGridFunction {
    pub fn fec(&self) -> &AFiniteElementCollection {
        let fec = mfem::GridFunction_OwnFEC(self.as_mfem());
        unsafe { AFiniteElementCollection::ref_of_mfem(fec) }
    }

    pub fn fec_mut(&mut self) -> &mut AFiniteElementCollection {
        let fec = self.as_mut_mfem().OwnFEC();
        unsafe { AFiniteElementCollection::mut_of_mfem(fec) }
    }

    // pub fn slice_mut(&mut self) -> &mut [f64] {
    //     let v = self.as_mut_mfem().GetTrueVector1();
    //     let len: i32 = v.Size().into();
    //     let data = v.GetData();
    //     unsafe { slice::from_raw_parts_mut(data, len as usize) }
    // }

    // pub fn fill(&mut self, c: f64) {
    //     self.slice_mut().fill(c);
    // }

    #[inline]
    pub fn save(&self) -> GridFunctionSave<'_> {
        GridFunctionSave {
            gf: self,
            precision: 8,
        }
    }
}

pub struct GridFunctionSave<'a> {
    gf: &'a AGridFunction,
    precision: i32,
}

impl GridFunctionSave<'_> {
    // TODO: error ?
    #[inline]
    pub fn to_file(&self, path: impl AsRef<Path>) {
        let_cxx_string!(
            filename = path.as_ref().as_os_str().as_encoded_bytes()
        );
        unsafe {
            let fname = filename.get_unchecked_mut().as_ptr();
            self.gf.as_mfem().Save1(
                fname as *const i8,
                c_int(self.precision));
        }
    }

    pub fn precision(&self, p: i32) -> Self {
        Self { gf: self.gf,  precision: p }
    }
}


wrap_mfem_base!(Coefficient, mfem Coefficient);

wrap_mfem!(ConstantCoefficient<>);
wrap_mfem_emplace!(ConstantCoefficient<>);
subclass!(owned ConstantCoefficient<>, Coefficient);

impl ConstantCoefficient {
    pub fn new(c: f64) -> Self {
        Self::emplace(mfem::ConstantCoefficient::new(c))
    }
}

pub use mfem::IntegrationRule;

// `LinearFormIntegrator` is an abstract class on the C// side.
// However, it will be taken by value by some functions such as
// `LinearForm::add_domain_integrator`.  Since
// `mfem::LinearFormIntegrator` lives on the C++ side and so can only
// be accessed through pointers, we define an owned type
// `LinearFormIntegrator`.  There will be no creation (`new`) function
// for that type but values can be produced by `Into` from owned
// values in sub-classes.
wrap_mfem!(
    /// Common capabilities of `LinarFormIntegrator`s.
    LinearFormIntegrator<>,
    base ALinearFormIntegrator);

wrap_mfem!(DomainLFIntegrator<'a>);
wrap_mfem_emplace!(DomainLFIntegrator<'a>);
subclass!(owned
    DomainLFIntegrator<'a>,
    into LinearFormIntegrator, ALinearFormIntegrator);

impl<'coeff> DomainLFIntegrator<'coeff> {
    pub fn new(qf: &'coeff mut Coefficient, a: i32, b: i32) -> Self {
        // Safety: The result does not seem to take ownership of `qf`.
        let qf = qf.as_mut_mfem();
        let lfi = mfem::DomainLFIntegrator::new(qf, c_int(a), c_int(b));
        Self::emplace(lfi)
    }
}

wrap_mfem!(
    /// Vector with associated FE space and [`LinearFormIntegrator`]s.
    LinearForm<'deps>,
    base ALinearForm);
wrap_mfem_emplace!(LinearForm<'deps>);
subclass!(LinearForm<'deps>, base ALinearForm, Vector, AVector);

impl<'a> LinearForm<'a> {
    pub fn new(fes: &FiniteElementSpace<'a>) -> Self {
        // XXX Safety: the underlying `fes` is not modified so can take it
        // by ref only.
        // TODO: Submit a PR to mfem.
        let lf = unsafe {
            mfem::LinearForm::new1(fes.as_mfem() as *const _ as *mut _)
        };
        LinearForm::emplace(lf)
    }

    pub fn fe_space(&self) -> &'a AFiniteElementSpace {
        let raw = self.as_mfem().FESpace1();
        unsafe { AFiniteElementSpace::ref_of_mfem(raw) }
    }

    pub fn assemble(&mut self) {
        self.as_mut_mfem().Assemble();
    }

    pub fn add_domain_integrator<Lfi>(&mut self, lfi: Lfi)
    where Lfi: Into<LinearFormIntegrator> {
        // The linear form "takes ownership of `lfi`".
        let lfi = lfi.into().into_raw();
        unsafe {
            self.as_mut_mfem().AddDomainIntegrator(lfi);
        }
    }
}

// See `LinearFormIntegrator` for the rationale.
wrap_mfem!(
    /// Common capabilities of `BilinearFormIntegrator`.
    BilinearFormIntegrator<>,
    base ABilinearFormIntegrator);

wrap_mfem!(DiffusionIntegrator<'a>);
subclass!(owned
    DiffusionIntegrator<'a>,
    into BilinearFormIntegrator, ABilinearFormIntegrator);

impl<'coeff> DiffusionIntegrator<'coeff> {
    pub fn new(qf: &'coeff mut Coefficient) -> Self {
        let qf = qf.as_mut_mfem();
        let ir: *const mfem::IntegrationRule = ptr::null();
        let bfi = unsafe { mfem::DiffusionIntegrator::new1(qf, ir) };
        Self {
            inner: UniquePtr::emplace(bfi),
            marker: PhantomData,
        }
    }
}

wrap_mfem!(
    BilinearForm<'a>,
    base ABilinearForm);
wrap_mfem_emplace!(BilinearForm<'a>);
subclass!(BilinearForm<'a>, base ABilinearForm, Matrix, AMatrix);

impl<'a> BilinearForm<'a> {
    pub fn new(fes: &FiniteElementSpace<'a>) -> Self {
        // XXX Safety: the underlying `fes` is not modified so can take it
        // by ref only.
        // TODO: Submit a PR to mfem.
        let lf = unsafe {
            mfem::BilinearForm::new2(fes.as_mfem() as *const _ as *mut _)
        };
        Self::emplace(lf)
    }

    /// Adds new Domain Integrator.
    pub fn add_domain_integrator<Bfi>(&mut self, bfi: Bfi)
    where
        Bfi: Into<BilinearFormIntegrator>,
    {
        // Doc says: "Assumes ownership of `bfi`".
        let bfi = bfi.into().into_raw();
        unsafe {
            self.as_mut_mfem().AddDomainIntegrator(bfi);
        }
    }

    /// Assembles the form i.e. sums over all domain/bdr integrators.
    pub fn assemble(&mut self, skip_zeros: bool) {
        let skip_zeros = if skip_zeros { 1 } else { 0 }; // TODO: option?
        self.inner.pin_mut().Assemble(c_int(skip_zeros));
    }

    pub fn form_linear_system(
        &mut self,
        ess_tdof_list: &AArrayInt,
        x: &AVector,
        b: &AVector,
        a_mat: &mut OperatorHandle,
        x_vec: &mut AVector,
        b_vec: &mut AVector,
    ) {
        let copy_interior = c_int(0);
        // self.as_mut_mfem().FormLinearSystem(
        //     ess_tdof_list, x, b,
        //     a_mat, x_vec, b_vec, copy_interior);
        mfem::BilinearForm_FormLinearSystem(
            self.as_mut_mfem(),
            ess_tdof_list.as_mfem(),
            x.as_mfem(),
            b.as_mfem(),
            // a_mat.as_mut_mfem(),
            a_mat.inner.as_mut().unwrap(),
            x_vec.as_mut_mfem(),
            b_vec.as_mut_mfem(),
            copy_interior);
    }

    // From `Operator`.
    pub fn recover_fem_solution(
        &mut self,
        from: &AVector,
        b: &AVector,
        x: &mut AVector,
    ) {
        let op: &mut Operator = self;
        unsafe { op.recover_fem_solution(from, b, x); }
    }
}

wrap_mfem!(
    /// Pointer to an Operator of a specified type.
    ///
    /// This provides a common type for global, matrix-type operators
    /// to be used in bilinear forms, gradients of nonlinear forms,
    /// static condensation, hybridization, etc.
    OperatorHandle<>, base AOperatorHandle);
wrap_mfem_emplace!(OperatorHandle<>);

// `OperatorHandle` is NOT a subclass of `Operator` but contains a
// pointer to an operator.  However, in C++, the operator *
// de-reference to `Operator` and `->` accesses `Operator` methods.
// In particular, where an `Operator&` is requested, a `*A`, where
// `A` is a `OperatorHandle`, can be provided.
//
// It thus makes sense to implement `Deref`.
impl ::std::ops::Deref for AOperatorHandle {
    type Target = Operator;

    #[inline]
    fn deref(&self) -> & Self::Target {
        let o = mfem::OperatorHandle_operator(self.as_mfem());
        unsafe { Operator::ref_of_mfem(o) }
    }
}

impl ::std::ops::DerefMut for AOperatorHandle {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let o = mfem::OperatorHandle_operator_mut(self.as_mut_mfem());
        unsafe { Operator::mut_of_mfem(o.get_unchecked_mut()) }
    }
}

impl Default for OperatorHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorHandle {
    pub fn new() -> Self {
        Self::emplace(mfem::OperatorHandle::new())
    }
}

impl AOperatorHandle {
    pub fn get_type(&self) -> OperatorType {
        self.as_mfem().Type()
    }
}

// FIXME: to any sub-class coercion (Deref), one must associate a try
// from the "bottom class" (following several Deref) to any type.  For
// this, a C++ templated code that has a test "is instance of" can be
// written.
impl<'a> TryFrom<&'a AOperatorHandle> for &'a ASparseMatrix {
    type Error = Error;

    fn try_from(oh: &'a AOperatorHandle) -> Result<Self, Self::Error> {
        if oh.get_type() == OperatorType::MFEM_SPARSEMAT {
            unsafe {
                let m = mfem::OperatorHandle_SparseMatrix(oh.as_mfem());
                Ok(&ASparseMatrix::ref_of_mfem(m))
            }
        } else {
            Err(Error::ConversionFailed)
        }
    }
}
// Deref not followed for traits.
impl<'a> TryFrom<&'a OperatorHandle> for &'a ASparseMatrix {
    type Error = Error;

    fn try_from(oh: &'a OperatorHandle) -> Result<Self, Self::Error> {
        let oh: &'a AOperatorHandle = oh;
        oh.try_into()
    }
}

wrap_mfem!(
    /// Data type sparse matrix.
    SparseMatrix<>, base ASparseMatrix);
// Sub-class of AbstractSparseMatrix, itself a subclass of Matrix.


wrap_mfem_base!(
    /// Base type for solvers.
    Solver, // No Rust constructor, only references.
    mfem Solver);

wrap_mfem!(GSSmoother<>);
wrap_mfem_emplace!(GSSmoother<>);
// FIXME: There are more subtle sub-class relationships but this one
// is needed for now.
subclass!(owned GSSmoother<>, Solver);


impl GSSmoother {
    pub fn new(a: &ASparseMatrix, t: usize, it: usize) -> Self {
        let t: i32 = t.try_into().unwrap_or(i32::MAX);
        let it: i32 = it.try_into().unwrap_or(i32::MAX);
        GSSmoother::emplace(mfem::GSSmoother::new1(
            a.as_mfem(), c_int(t), c_int(it)))
    }
}

// TODO: make a prelude and do not include this inside.
/// Preconditioned conjugate gradient method.
pub fn pcg<'a>(
    a: &'a Operator,
    solver: &'a mut Solver,
    b: &'a AVector,
    x: &'a mut AVector,
) -> PCG<'a> {
    PCG { a, solver, b, x,
        print_iter: false,
        // TODO: revisit defaults
        max_num_iter: 200,
        rtol: 1e-12,
        atol: 0.0,
    }
}

pub struct PCG<'a> {
    a: &'a Operator,
    solver: &'a mut Solver,
    b: &'a AVector,
    x: &'a mut AVector,
    print_iter: bool,
    max_num_iter: i32,
    rtol: f64,
    atol: f64,
}

impl PCG<'_> {
    /// Print a line for each iteration.
    pub fn print_iter(self, pr: bool) -> Self {
        Self { print_iter: pr, .. self }
    }

    /// Set the maximum number of iterations.
    pub fn max_num_iter(self, mx: usize) -> Self {
        let max_num_iter = mx.try_into().unwrap_or(i32::MAX);
        Self { max_num_iter, .. self }
    }

    /// Set the relative tolerance.  Note that tolerances are squared.
    pub fn rtol(self, rtol: f64) -> Self {
        Self { rtol: rtol.max(0.), .. self }
    }

    /// Set the absolute tolerance.  Note that tolerances are squared.
    pub fn atol(self, atol: f64) -> Self {
        Self { atol: atol.max(0.), .. self }
    }

    pub fn solve(&mut self) {
        let print_iter = if self.print_iter {1} else {0};
        mfem::PCG(
            self.a.as_mfem(),
            self.solver.as_mut_mfem(),
            self.b.as_mfem(),
            self.x.as_mut_mfem(),
            print_iter, self.max_num_iter,
            self.rtol, self.atol);
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
