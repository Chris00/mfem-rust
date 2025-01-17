use autocxx::{moveit::MakeCppStorage, prelude::*};
use cxx::{let_cxx_string, memory::UniquePtrTarget, UniquePtr};
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

trait WrapMfem {
    type Mfem;

    /// Reinterpret the pointer `ptr` as a mutable reference.
    unsafe fn mut_of_mfem<'a>(ptr: *mut Self::Mfem) -> &'a mut Self;

    fn as_mfem(&self) -> &Self::Mfem;
    fn as_mut_mfem(&mut self) -> Pin<&mut Self::Mfem>;

    unsafe fn ref_of_mfem<'a>(ptr: *const Self::Mfem) -> &'a Self {
        // Constraint to *mut is fine as the return is immutable.
        Self::mut_of_mfem(ptr as *mut _)
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
        impl WrapMfem for $tymfem {
            type Mfem = mfem::$ty;

            #[inline]
            unsafe fn mut_of_mfem<'a>(ptr: *mut mfem::$ty) -> &'a mut Self {
                debug_assert!(! ptr.is_null());
                // `ptr` is a pointer to `$tymfem`.  Since the
                // representation of `Self` is transparent, `ptr` can
                // be seen as a reference to `Self`.
                &mut *(ptr as *mut $tymfem)
            }
            #[inline]
            fn as_mfem(&self) -> & mfem:: $ty {
                &self.inner
            }
            #[inline]
            fn as_mut_mfem(&mut self) -> Pin<&mut mfem:: $ty> {
                unsafe { Pin::new_unchecked(&mut self.inner) }
            }
        }
    };
}

trait OwnedMfem {
    type Mfem: UniquePtrTarget;

    /// Return an *owned* value, from the pointer `ptr`.
    fn from_unique_ptr(ptr: UniquePtr<Self::Mfem>) -> Self;

    /// Consumes `self`, releasing its ownership, and returns the
    /// pointer (to the C++ heap).
    fn into_raw(self) -> *mut Self::Mfem;

    #[inline]
    fn emplace<N>(n: N) -> Self
    where
        N: New<Output = Self::Mfem>,
        Self::Mfem: MakeCppStorage,
        Self: Sized,
    {
        let ptr = UniquePtr::emplace(n);
        Self::from_unique_ptr(ptr)
    }
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
        impl $(<$l>)? ::std::ops::Deref for $ty $(<$l>)?
        where
            Self: OwnedMfem<Mfem = mfem::$ty>,
            $tymfem: WrapMfem<Mfem = mfem::$ty>
        {
            type Target = $tymfem;

            #[inline]
            fn deref(&self) -> &Self::Target {
                let ptr: & mfem::$ty = self.inner.as_ref().unwrap();
                unsafe { $tymfem::ref_of_mfem(ptr as *const _) }
            }
        }
        impl $(<$l>)? ::std::ops::DerefMut for $ty $(<$l>)?
        where
            Self: OwnedMfem<Mfem = mfem::$ty>,
            $tymfem: WrapMfem<Mfem = mfem::$ty>
        {
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
        #[allow(dead_code, private_bounds)]
        impl $(<$l>)? $ty $(<$l>)?
        where Self: OwnedMfem<Mfem = mfem::$ty> {
            #[inline]
            fn as_mfem(&self) -> & mfem::$ty {
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
        #[allow(non_camel_case_types)]
        #[repr(transparent)]
        pub struct $ty $(<$l>)? {
            inner: UniquePtr<mfem:: $ty>,
            marker: PhantomData<$(&$l)? ()>,
        }
        impl $(<$l>)? OwnedMfem for $ty $(<$l>)? {
            type Mfem = mfem::$ty;
            #[inline]
            fn from_unique_ptr(ptr: UniquePtr<mfem:: $ty>) -> Self {
                Self { inner: ptr,  marker: PhantomData }
            }
            #[inline]
            fn into_raw(self) -> *mut mfem::$ty { self.inner.into_raw() }
        }
    };
}

/// Empty trait simply to be able to write a safety check as a bound.
#[allow(dead_code)]
trait Subclass {}

// Let A (resp. B) be types owning classes A (resp. B).  Possibly A
// (resp. B) has a "base type" AA (resp. AB).
//
// If A is a subclass of B, we define:

// - If AA and AB exists (in which case there are deref(&A) -> &AA and
//   deref_mut(&mut A) -> &mut AA and the same for B), we implement:
//   - deref(&AA) -> &AB
//   - deref_mut(&mut AA) -> &mut AB
// - If A is a type without a "base" AA:
//   - deref(&A) -> &AB
//   - deref_mut(&mut A) -> &mut AB
// - For owned types:
//   - into(A) -> B
macro_rules! subclass {
    // If a lifetime is givent to parent value `$ty0` it must be the
    // same as `$l` and signals the the dependencies `$l` tracks are
    // still needed for `ty0`.
    ($tysub0: ident < $($l: lifetime)? >, base $tysub: ident,
        $ty0: ident < $($lparent: lifetime)? >, $ty: ident
    ) => {
        // Use the fact that `autocxx` implements the `AsRef`
        // relationship for sub-classes to make sure it is safe.
        impl Subclass for $tysub
        where mfem::$tysub0 : AsRef<mfem::$ty0> {}
        subclass!(unsafe $tysub0 <$($l)?>, base $tysub,
            $ty0 <$($lparent)?>, $ty);
    };
    (unsafe $tysub0: ident < $($l: lifetime)? >, base $tysub: ident,
        $ty0: ident < $($lparent: lifetime)? >, $ty: ident
    ) => {
        subclass!(unsafe base $tysub, $ty);
        subclass!(unsafe $tysub0 <$($l)?>, into $ty0 <$($lparent)?>);
    };
    // In the following case, the parent class does not have an owned
    // representation, only a thin wrapper.
    ($tysub0: ident < $($l: lifetime)? >, base $tysub: ident, $ty: ident) => {
        impl Subclass for $tysub
        where mfem::$tysub0 : AsRef<mfem::$ty> {}
        subclass!(unsafe base $tysub, $ty);
    };
    (unsafe base $tysub: ident, $ty: ident) => {
        impl ::std::ops::Deref for $tysub
        where $tysub: WrapMfem, $ty: WrapMfem {
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
        impl ::std::ops::DerefMut for $tysub
        where $tysub: WrapMfem, $ty: WrapMfem {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe {
                    // Safety: see `Deref`.
                    &mut *(self as *mut _ as *mut $ty)
                }
            }
        }
    };
    (unsafe $tysub0: ident < $($l: lifetime)? >,
        into $ty0: ident < $($lparent: lifetime)? >
    ) => {
        // For owned values, also define `Into`.  Note that the
        // lifetimes may of may not need be preserved depending on
        // whether the dependencies theiy track are still needed.
        impl $(<$l>)? ::std::convert::Into<$ty0 $(<$lparent>)?>
        for $tysub0 $(<$l>)?
        where $tysub0 $(<$l>)? : OwnedMfem, $ty0 $(<$lparent>)?: OwnedMfem {
            #[inline]
            fn into(self) -> $ty0 $(<$lparent>)? {
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
        into $ty0: ident < $($lparent: lifetime)? >, $ty: ident
    ) => {
        subclass!(unsafe $tysub0 <$($l)?>, into $ty0 <$($lparent)?>);
        subclass!(owned $tysub0 <$($l)?>, $ty);
    };
    (owned $tysub0: ident < $($l: lifetime)? >, $ty: ident) => {
        impl $(<$l>)? ::std::ops::Deref for $tysub0 $(<$l>)?
        where $tysub0 $(<$l>)? : OwnedMfem, $ty: WrapMfem {
            type Target = $ty;

            #[inline]
            fn deref(&self) -> & Self::Target {
                let ptr: & mfem::$tysub0 = self.inner.as_ref().unwrap();
                unsafe { &*(ptr as *const mfem::$tysub0 as *const $ty) }
            }
        }
        impl $(<$l>)? ::std::ops::DerefMut for $tysub0 $(<$l>)?
        where $tysub0 $(<$l>)? : OwnedMfem, $ty: WrapMfem {
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
    // Dereferencing to a mutable slice does not allow to move the
    // target because the slice is not `Sized` (so, for example,
    // `swap` cannot be used).
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

wrap_mfem_base!(Element, mfem Element);

#[derive(Debug, Clone, Copy)]
pub enum ElementType {
    Point, Segment, Triangle, Quadrilateral,
    Tetrahedron, Hexahedron, Wedge, Pyramid,
}

impl From<mfem::Element_Type> for ElementType {
    fn from(et: mfem::Element_Type) -> Self {
        use ElementType::*;
        match et {
            mfem::Element_Type::POINT => Point,
            mfem::Element_Type::SEGMENT => Segment,
            mfem::Element_Type::TRIANGLE => Triangle,
            mfem::Element_Type::QUADRILATERAL => Quadrilateral,
            mfem::Element_Type::TETRAHEDRON => Tetrahedron,
            mfem::Element_Type::HEXAHEDRON => Hexahedron,
            mfem::Element_Type::WEDGE => Wedge,
            mfem::Element_Type::PYRAMID => Pyramid,
        }
    }
}
impl From<ElementType> for mfem::Element_Type {
    fn from(et: ElementType) -> mfem::Element_Type {
        use ElementType::*;
        match et {
            Point => mfem::Element_Type::POINT,
            Segment => mfem::Element_Type::SEGMENT,
            Triangle => mfem::Element_Type::TRIANGLE,
            Quadrilateral => mfem::Element_Type::QUADRILATERAL,
            Tetrahedron => mfem::Element_Type::TETRAHEDRON,
            Hexahedron => mfem::Element_Type::HEXAHEDRON,
            Wedge => mfem::Element_Type::WEDGE,
            Pyramid => mfem::Element_Type::PYRAMID,
        }
    }
}

impl Element {
    pub fn get_type(&self) -> ElementType {
        self.as_mfem().GetType().into()
    }
}


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

    /// Creates 1D mesh for the interval divided into `n` equal intervals.
    pub fn make_cartesian1d(n: usize, sx: f64) -> Self {
        Self::emplace(mfem::Mesh::MakeCartesian1D(c_int(n as i32), sx))
    }

    pub fn cartesian2d(
        nx: usize,
        ny: usize,
        ty: ElementType,
    ) -> Cartesian2D {
        Cartesian2D {
            nx, ny, ty,
            generate_edges: true,
            sx: 1.,  sy: 1.,  scf_ordering: true,
        }
    }

    // revisit interface
    pub fn init(
        dim: usize,
        nvert: usize,
        nelem: usize,
        nbdr_elem: usize,
        space_dim: Option<usize>,
    ) -> MeshBuilder {
        let space_dim = space_dim.map(|x| x as i32).unwrap_or(-1);
        let inner = Self::emplace(mfem::Mesh::new5(
            c_int(dim as i32),
            c_int(nvert as i32),
            c_int(nelem as i32),
            c_int(nbdr_elem as i32),
            c_int(space_dim as i32)));
        MeshBuilder { inner, dim }
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

pub struct Cartesian2D {
    nx: usize,
    ny: usize,
    ty: ElementType,
    generate_edges: bool,
    sx: f64,
    sy: f64,
    scf_ordering: bool,
}

impl Cartesian2D {
    pub fn generate_edges(self, yes: bool) -> Self {
        Self { generate_edges: yes, .. self }
    }

    /// Set the length of the rectangle.
    pub fn sx(self, sx: f64) -> Self {
        Self { sx, .. self }
    }

    /// Set the height of the rectangle.
    pub fn sy(self, sy: f64) -> Self {
        Self { sy, .. self }
    }

    /// Set wether to use space-filling curve ordering.  Default: `true`.
    pub fn scf_ordering(self, yes: bool) -> Self {
        Self { scf_ordering: yes, .. self }
    }

    pub fn make(&self) -> Mesh {
        Mesh::emplace(mfem::Mesh::MakeCartesian2D(
            c_int(self.nx as i32), c_int(self.ny as i32),
            self.ty.into(), self.generate_edges,
            self.sx, self.sy, self.scf_ordering))
    }
}

pub struct MeshBuilder {
    inner: Mesh,
    dim: usize,
}

impl MeshBuilder {
    pub fn add_vertex<const N: usize>(&mut self, coord: [f64; N]) -> i32 {
        if N != self.dim {
            panic!("MeshBuilder::add_vertex: expected dim {}, got {N}",
                self.dim);
        }
        unsafe {
            self.inner.as_mut_mfem().AddVertex1(coord.as_ptr()).into()
        }
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

    // fn get_mut_nodes should wrap the return value (deref to
    // GridFunction) so that a custom Drop is executed that calls
    // Mesh::NodesUpdated()

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
    H1_FECollection<>,
    base AH1_FECollection);
subclass!(
    H1_FECollection<>, base AH1_FECollection,
    FiniteElementCollection<>, AFiniteElementCollection);

impl H1_FECollection {
    /// Return a H1-conforming (continuous) finite elements with order
    /// `p`, dimension `dim` and the default basis type
    /// [`GaussLobatto`][BasisType::GaussLobatto].
    pub fn new(p: i32, dim: i32) -> Self {
        Self::with_basis(p, dim, BasisType::GaussLobatto)
    }

    /// Return a H1-conforming (continuous) finite elements with
    /// positive basis functions with order `p` and dimension `dim`.
    #[doc(alias = "H1Pos_FECollection")]
    pub fn pos(p: i32, dim: i32) -> Self {
        // https://docs.mfem.org/html/fe__coll_8hpp_source.html#l00305
        Self::with_basis(p, dim, BasisType::Positive)
    }

    /// Return a H1-conforming (continuous) serendipity finite elements
    /// with order `p` and dimension `dim`.  Current implementation
    /// works in 2D only; 3D version is in development.
    #[doc(alias = "H1Ser_FECollection")]
    pub fn ser(p: i32, dim: i32) -> Self {
        Self::with_basis(p, dim, BasisType::Serendipity)
    }

    /// Return a "H^{1/2}-conforming" trace finite elements with order
    /// `p` and dimension `dim` defined on the interface between mesh
    /// elements (faces,edges,vertices); these are the trace FEs of
    /// the H1-conforming FEs.
    #[doc(alias = "H1_Trace_FECollection")]
    pub fn trace(p: i32, dim: i32, btype: BasisType) -> Self {
        Self::with_basis(p, dim - 1, btype)
    }

    /// Return a H1-conforming (continuous) finite elements with order
    /// `p`, dimension `dim` and basis type `btype`.
    pub fn with_basis(p: i32, dim: i32, btype: BasisType) -> Self {
        let h1 = mfem::H1_FECollection::new(
            c_int(p),
            c_int(dim),
            c_int(btype as i32),
        );
        H1_FECollection::emplace(h1)
    }

}

impl AH1_FECollection {
    pub fn get_basis_type(&self) -> Option<BasisType> {
        self.as_mfem().GetBasisType().try_into().ok()
    }
}

wrap_mfem!(
    /// Arbitrary order L²2-conforming discontinuous finite elements.
    L2_FECollection<>);
subclass!(owned L2_FECollection<>, AFiniteElementCollection);


wrap_mfem!(
    /// Arbitrary order "H^{1/2}-conforming" trace finite elements
    /// defined on the interface between mesh elements
    /// (faces,edges,vertices); these are the trace FEs of the
    /// H1-conforming FEs.
    H1_Trace_FECollection<>);
subclass!(owned H1_Trace_FECollection<>, AH1_FECollection);

wrap_mfem!(
    /// Arbitrary order H(div)-conforming Raviart-Thomas finite elements.
    RT_FECollection<>,
    base ART_FECollection);
subclass!(
    RT_FECollection<>, base ART_FECollection,
    FiniteElementCollection<>, AFiniteElementCollection);

wrap_mfem!(
    /// Arbitrary order H(curl)-conforming Nedelec finite elements.
    ND_FECollection<>,
    base AND_FECollection);
subclass!(
    ND_FECollection<>, base AND_FECollection,
    FiniteElementCollection<>, AFiniteElementCollection);

wrap_mfem!(
    /// Crouzeix-Raviart nonconforming elements in 2D.
    CrouzeixRaviartFECollection<>);
subclass!(owned CrouzeixRaviartFECollection<>, AFiniteElementCollection);

pub enum Ordering {
    /// This ordering arranges the DOFs by nodes first.  It is often
    /// used for continuous finite element spaces, such as those based
    /// on H¹ elements.  This ordering is beneficial for certain types
    /// of solvers and preconditioners that exploit the nodal
    /// structure of the problem.
    ByNodes = mfem::Ordering_Type::byNODES as isize,
    /// This ordering arranges the DOFs by vector dimension first.  It
    /// is typically used for vector-valued problems where the
    /// components of the vector field are stored consecutively.  This
    /// can be useful for problems in elasticity or fluid dynamics
    /// where the vector field represents physical quantities like
    /// displacement or velocity.
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
    /// Return a new space from the `mesh` an Finite Element
    /// Collection `fec`.
    pub fn new<'mesh: 'a, 'fec: 'a>(
        mesh: &'mesh Mesh,
        fec: &'fec AFiniteElementCollection,
    ) -> Self {
        Self::with_ordering(mesh, fec, 1, Ordering::ByNodes)
    }

    /// Return a new space from the `mesh` an Finite Element
    /// Collection `fec`.  This is a space of vectors with `vdim`
    /// components and arranges the degrees of freedom by `ordering`.
    pub fn with_ordering<'mesh: 'a, 'fec: 'a>(
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

    /// Return the number of vector true (conforming) dofs.
    pub fn get_true_vsize(&self) -> i32 {
        mfem::FES_GetTrueVSize(self.as_mfem()).into()
    }

    /// Mark degrees of freedom associated with boundary elements with
    /// the specified boundary attributes (marked in `bdr_attr_is_ess`).
    /// For spaces with `vdim` > 1, the `component` parameter can be
    /// used to restricts the marked vDOFs to the specified component.
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

    /// Get a list of all boundary true dofs, `boundary_dofs`.  For
    /// spaces with `vdim` > 1, the `component` parameter can be used
    /// to restrict the marked tDOFs to the specified component.
    /// Equivalent to [`SElf::get_essential_true_dofs` with all
    /// boundary attributes marked as essential.
    pub fn get_boundary_true_dofs(
        &self,
        boundary_dofs: &mut AArrayInt,
		component: Option<usize>
    ) {
        mfem::FES_GetBoundaryTrueDofs(
            self.as_mfem(),
            boundary_dofs.as_mut_mfem(),
            c_int(component.map(|c| c as i32).unwrap_or(-1)));
    }
}

wrap_mfem!(
    /// Represent a [`Vector`] with associated FE space.
    GridFunction<'fes>,
    base AGridFunction);
subclass!(
    GridFunction<'fes>, base AGridFunction,
    // FIXME: Does the reinterpretation as a vector no longer need the
    // space?  Freed correctly?
    Vector<>, AVector);

impl<'fes> GridFunction<'fes> {
    // XXX Can `fes` be mutated?
    // Hopefully so because it is shared with e.g. `Linearform`
    // XXX can you pass a different `fes` than the one for `Linearform`?
    /// Construct a GridFunction associated with the
    /// FiniteElementSpace `fes`.
    #[inline]
    pub fn new(fes: &'fes FiniteElementSpace) -> Self {
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
subclass!(owned ConstantCoefficient<>, Coefficient);

impl ConstantCoefficient {
    pub fn new(c: f64) -> Self {
        Self::emplace(mfem::ConstantCoefficient::new(c))
    }
}

wrap_mfem!(
    /// A general function coefficient.
    FunctionCoefficient<>);
subclass!(owned FunctionCoefficient<>, Coefficient);

impl FunctionCoefficient {
    pub fn new<F>(mut f: F) -> Self
    where F: FnMut(&AVector) -> f64 {
        let eval = |x: &mfem::Vector, d: *mut mfem::c_void| -> f64 {
            let f = unsafe { &mut *(d as *mut F) };
            let x = unsafe { AVector::ref_of_mfem(x) };
            f(x)
        };
        let d = &mut f as *mut F as *mut mfem::c_void;
        let fc = unsafe { mfem::new_FunctionCoefficient(eval, d) };
        FunctionCoefficient::from_unique_ptr(fc)
    }
}

wrap_mfem!(
    /// Coefficient defined by a GridFunction. This coefficient is
    /// mesh dependent.
    GridFunctionCoefficient<'gf>,
    base AGridFunctionCoefficient);
subclass!(
    GridFunctionCoefficient<'gf>, base AGridFunctionCoefficient,
    Coefficient);

impl<'gf> GridFunctionCoefficient<'gf> {
    pub fn new(gf: &'gf AGridFunction) -> Self {
        unsafe {
            Self::emplace(mfem::GridFunctionCoefficient::new1(
                gf.as_mfem(), c_int(1)))
        }
    }
}

wrap_mfem!(
    /// Scalar coefficient defined as the inner product of two vector
    /// coefficients.
    InnerProductCoefficient<>);
subclass!(owned InnerProductCoefficient<>, Coefficient);

wrap_mfem_base!(
    /// Base type for vector Coefficients that optionally depend on
    /// time and space.
    VectorCoefficient, mfem VectorCoefficient);

wrap_mfem!(
    /// Vector coefficient that is constant in space and time.
    VectorConstantCoefficient<>);
subclass!(owned VectorConstantCoefficient<>, VectorCoefficient);

impl VectorConstantCoefficient {
    pub fn new(x: &AVector) -> Self {
        Self::emplace(mfem::VectorConstantCoefficient::new(x.as_mfem()))
    }
}

wrap_mfem!(
    /// A general vector function coefficient.
    VectorFunctionCoefficient<>);
subclass!(owned VectorFunctionCoefficient<>, VectorCoefficient);



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
    LinearFormIntegrator<'deps>,
    base ALinearFormIntegrator);

wrap_mfem!(
    /// Represent a general integrator that supports delta coefficients.
    DeltaLFIntegrator<'deps>,
    base ADeltaLFIntegrator);
subclass!(
    DeltaLFIntegrator<'deps>, base ADeltaLFIntegrator,
    LinearFormIntegrator<'deps>, ALinearFormIntegrator);

wrap_mfem!(
    /// Type for domain integration L(v) := ∫ fv.
    DomainLFIntegrator<'deps>);
subclass!(owned
    DomainLFIntegrator<'deps>,
    into DeltaLFIntegrator<'deps>, ADeltaLFIntegrator);
subclass!(unsafe DomainLFIntegrator<'deps>, into LinearFormIntegrator<'deps>);


impl<'coeff> DomainLFIntegrator<'coeff> {
    /// Return a new linear form integrator v ↦  ∫ fv with order 2.
    pub fn new(qf: &'coeff mut Coefficient) -> Self {
        Self::with_order(qf, 2)
    }

    /// Return a new linear form integrator v ↦  ∫ fv with order `a`.
    pub fn with_order(qf: &'coeff mut Coefficient, a: usize) -> Self {
        // Safety: The result does not seem to take ownership of `qf`.
        let qf = qf.as_mut_mfem();
        let a = c_int(a as i32);
        let options = c_int(0);
        let lfi = mfem::DomainLFIntegrator::new(qf, a, options);
        Self::emplace(lfi)
    }
}

wrap_mfem!(
    /// Vector with associated FE space and [`LinearFormIntegrator`]s.
    LinearForm<'deps>,
    base ALinearForm);
subclass!(
    LinearForm<'deps>, base ALinearForm,
    // FIXME: Can drop the deps?
    Vector<>, AVector);

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

    pub fn add_domain_integrator<'b: 'a, Lfi>(&mut self, lfi: Lfi)
    where Lfi: Into<LinearFormIntegrator<'b>> {
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
    BilinearFormIntegrator<'a>,
    base ABilinearFormIntegrator);

wrap_mfem!(
    /// α(Q ∇u, ∇v)
    DiffusionIntegrator<'a>);
subclass!(owned
    DiffusionIntegrator<'a>,
    into BilinearFormIntegrator<'a>, ABilinearFormIntegrator);

impl<'coeff> DiffusionIntegrator<'coeff> {
    /// Return a diffusion integrator with coefficient 1:
    /// (u, v) ↦ ∫ ∇u∇v.
    pub fn new() -> Self {
        let bfi = unsafe {
            mfem::DiffusionIntegrator::new(ptr::null())
        };
        Self::emplace(bfi)
    }

    /// Return a diffusion integrator with coefficient `q`:
    /// (u, v) ↦ ∫ q ∇u∇v.
    pub fn with_coeff(q: &'coeff mut Coefficient) -> Self {
        let q = q.as_mut_mfem();
        let ir: *const mfem::IntegrationRule = ptr::null();
        let bfi = unsafe { mfem::DiffusionIntegrator::new1(q, ir) };
        Self::emplace(bfi)
    }
}

wrap_mfem!(
    /// α(Q·∇u, v)
    ConvectionIntegrator<'a>);
subclass!(owned ConvectionIntegrator<'a>,
    into BilinearFormIntegrator<'a>, ABilinearFormIntegrator);


wrap_mfem!(
    BilinearForm<'a>,
    base ABilinearForm);
subclass!(
    BilinearForm<'a>, base ABilinearForm,
    // FIXME: Ok to drop the dependencies lifetime?
    Matrix<>, AMatrix);

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
    pub fn add_domain_integrator<'b: 'a, Bfi>(&mut self, bfi: Bfi)
    where Bfi: Into<BilinearFormIntegrator<'b>> {
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

    /// Form the linear system A X = B, corresponding to this bilinear
    /// form and the linear form `b`(.).
    ///
    /// This method applies any necessary transformations to the
    /// linear system such as: eliminating boundary conditions;
    /// applying conforming constraints for non-conforming AMR;
    /// parallel assembly; static condensation; hybridization.
    ///
    /// The [`GridFunction`]-size vector `x` must contain the
    /// essential b.c.  The BilinearForm and the LinearForm-size
    /// vector `b` must be assembled.
    ///
    /// The vector `x_vec` is initialized with a suitable initial
    /// guess: when using hybridization, the vector `x_vec` is set to
    /// zero; otherwise, the essential entries of `x_vec` are set to
    /// the corresponding b.c. and all other entries are set to zero
    /// (`copy_interior` == 0) or copied from `x` (copy_interior != 0).
    ///
    /// This method can be called multiple times (with the same
    /// `ess_tdof_list` array) to initialize different right-hand
    /// sides and boundary condition values.
    ///
    /// After solving the linear system, the finite element solution x can
    /// be recovered by calling [`recover_fem_solution`] (with the same
    /// vectors `x_vec`, `b`, and `x`).
    ///
    /// NOTE: If there are no transformations, `x_vec` simply reuses the
    /// data of `x`.
    pub fn form_linear_system(
        &mut self,
        ess_tdof_list: &AArrayInt,
        x: &AVector,
        b: &AVector,
    ) -> (OperatorHandle<'a>, Vector, Vector) {
        // FIXME: The possible dependency of `x_vec` and `b_vec` on
        // `x` and `b` must be expressed.
        let copy_interior = c_int(0);
        // self.as_mut_mfem().FormLinearSystem(
        //     ess_tdof_list, x, b,
        //     a_mat, x_vec, b_vec, copy_interior);
        // FIXME: is there a point to allow passing `a_mat`, `x_vec`
        // and `b_vec` as optional arguments (moving them)?  Their
        // data may be wiped out and redirected to the one of `self`,
        // `x` and `b`.
        let mut a_mat = OperatorHandle::new();
        let mut x_vec = Vector::new();
        let mut b_vec = Vector::new();
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
        (a_mat, x_vec, b_vec)
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
    OperatorHandle<'deps>, base AOperatorHandle);

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

impl Default for OperatorHandle<'static> {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorHandle<'static> {
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
                let m = mfem::OperatorHandle_ref_SparseMatrix(oh.as_mfem());
                Ok(&ASparseMatrix::ref_of_mfem(m))
            }
        } else {
            Err(Error::ConversionFailed)
        }
    }
}
// Deref not followed for traits.
impl<'a> TryFrom<&'a OperatorHandle<'a>> for &'a ASparseMatrix {
    type Error = Error;

    fn try_from(oh: &'a OperatorHandle<'a>) -> Result<Self, Self::Error> {
        let oh: &'a AOperatorHandle = oh;
        oh.try_into()
    }
}

wrap_mfem!(
    /// Data type sparse matrix.
    SparseMatrix<>, base ASparseMatrix);
// Sub-class of AbstractSparseMatrix, itself a subclass of Matrix.
subclass!(unsafe
    SparseMatrix<>, base ASparseMatrix,
    Matrix<>, AMatrix);

impl SparseMatrix {
    pub fn new() -> Self {
        Self::emplace(mfem::SparseMatrix::new())
    }
}

impl ASparseMatrix {
    pub fn as_oper(&mut self) -> OperatorHandle<'_> {
        let x = unsafe { self.as_mut_mfem().get_unchecked_mut() };
        OperatorHandle::from_unique_ptr(
            unsafe { mfem::SparseMatrix_to_OperatorHandle(x) })
    }
}



wrap_mfem_base!(
    /// Base type for solvers.
    Solver, // No Rust constructor, only references.
    mfem Solver);

wrap_mfem!(GSSmoother<>);
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

    #[test]
    fn test_grid_function_lifetime() -> Result<(), Box<dyn std::error::Error>> {
        let mesh = Mesh::cartesian2d(50, 50, ElementType::Triangle).make();
        let fec = H1_FECollection::new(1, 2);
        let fespace = FiniteElementSpace::new(&mesh, &fec);
        let mut x = GridFunction::new(&fespace);
        x.fill(1.);
        let v: Vector = x.into();
        drop(fespace);
        assert_eq!(v[0], 1.);
        Ok(())
    }
}
