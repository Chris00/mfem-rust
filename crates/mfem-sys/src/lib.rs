#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

use autocxx::prelude::*;

include_cpp! {
    #include "mfem.hpp"
    #include "extra.hpp"
    safety!(unsafe)
    generate!("extra::Mfem_MFEM_USE_EXCEPTIONS")
    generate!("extra::NumBasisTypes")
    generate!("mfem::ErrorAction")
    generate!("mfem::set_error_action")
    generate!("mfem::Vector")
    generate!("extra::ArrayInt") // Alias for Array<int>
    generate!("extra::ArrayInt_with_len")
    generate!("extra::ArrayInt_copy")
    generate!("extra::ArrayInt_from_slice")
    generate!("extra::ArrayInt_len")
    generate!("extra::ArrayInt_GetData")
    generate!("extra::ArrayInt_GetDataMut")

    // mfem::Operator is abstract, no automatic binding.
    generate!("mfem::Operator_Type") // Operator::Type
    generate!("mfem::Matrix")

    generate!("mfem::Element_Type")

    generate!("mfem::Mesh")
    generate!("extra::Mesh_bdr_attributes")
    generate!("mfem::Mesh_Operation")       // Mesh::Operation
    generate!("mfem::Mesh_FaceTopology")    // Mesh::FaceTopology
    generate!("mfem::Mesh_ElementLocation") // Mesh::ElementLocation
    generate!("mfem::Mesh_ElementConformity") // Mesh::ElementConformity
    generate!("mfem::Mesh_FaceInfoTag")     // Mesh::FaceInfoTag

    generate!("mfem::FiniteElement")
    generate!("mfem::FiniteElementCollection")
    generate!("mfem::L2_FECollection")
    generate!("mfem::H1_FECollection")
    generate!("mfem::H1_Trace_FECollection")
    generate!("mfem::RT_FECollection")
    generate!("mfem::ND_FECollection")
    generate!("mfem::CrouzeixRaviartFECollection")

    generate!("mfem::Ordering")
    // `FiniteElementSpace` contains protected types which are not
    // handled well.  Thus, one must bind by hand.
    // generate!("mfem::FiniteElementSpace")
    generate!("extra::FES")
    generate!("extra::FES_new")
    generate!("extra::FES_GetEssentialTrueDofs")
    generate!("extra::FES_GetTrueVSize")

    generate!("mfem::GridFunction")
    generate!("extra::GridFunction_OwnFEC") // immutable version
    generate!("mfem::LinearFormIntegrator")
    generate!("mfem::BilinearFormIntegrator")
    generate!("mfem::LinearForm")
    generate!("mfem::BilinearForm")
    generate!("extra::BilinearForm_FormLinearSystem")
    generate!("mfem::MixedBilinearForm")

    generate!("mfem::Coefficient")
    generate!("mfem::ConstantCoefficient")
    generate!("mfem::FunctionCoefficient")
    generate!("mfem::GridFunctionCoefficient")
    generate!("mfem::InnerProductCoefficient")
    generate!("mfem::VectorCoefficient")
    generate!("mfem::VectorConstantCoefficient")
    generate!("mfem::VectorFunctionCoefficient")

    generate!("mfem::OperatorHandle")
    // mfem::LinearFormIntegrator is an abstract class.
    generate!("extra::LFI")
    generate!("mfem::DeltaLFIntegrator")
    generate!("mfem::DomainLFIntegrator")
    // mfem::BilinearFormIntegrator is an abstract class.
    generate!("extra::BFI")
    generate!("mfem::DiffusionIntegrator")
    generate!("mfem::ConvectionIntegrator")

    generate!("mfem::SparseMatrix")
    generate!("mfem::Solver")
    generate!("mfem::GSSmoother")
    generate!("mfem::PermuteFaceL2")
}

// We handle abstract classes in this module.  One can convert
// pointers to these classes and use the methods.  This avoids writing
// C++ code to apply all the abstract methods to the sub-classes.
#[cxx::bridge]
mod ffi_cxx {
    unsafe extern "C++" {
        include!("ffi_cxx.hpp");
        // mfem::Operator::Type.  mfem::Operator is not really a
        // namespace but this is needed for the type Id to coincide with
        // autocxx.  (We want that compatibility because, say,
        // `OperatorHandle::Type` also return that type.)
        #[namespace = "mfem::Operator"]
        type Type = crate::Operator_Type;
        #[namespace = "mfem"]
        type Vector = crate::Vector;
        type Operator;

        fn GetType(self: &Operator) -> Type;
        fn Height(self: &Operator) -> i32;
        fn Width(self: &Operator) -> i32;
        /// Safety: Virtual method.  Make sure it applied only to proper
        /// sub-classes
        unsafe fn RecoverFEMSolution(
            self: Pin<&mut Operator>,
            X: &Vector, b: &Vector, x: Pin<&mut Vector>);

        #[namespace = "mfem"]
        type Element;
        #[namespace = "mfem::Element"]
        #[cxx_name = "Type"]
        type Element_Type = crate::Element_Type;
        fn GetType(self: &Element) -> Element_Type;

        // autocxx does not bind any constructor of `FunctionCoefficient`.
        // Moreover, we "enhance" the interface to allow to pass closures.
        #[namespace = "mfem"]
        type FunctionCoefficient = crate::FunctionCoefficient;
        type c_void;
        unsafe fn new_FunctionCoefficient(
            f: unsafe fn(&Vector, data: *mut c_void) -> f64,
            data: *mut c_void,
        ) -> UniquePtr<FunctionCoefficient>;

        #[namespace = "mfem"]
        type Matrix = crate::Matrix;
        #[cxx_name = "upcast_to_operator"]
        fn Matrix_to_operator<'a>(m: &'a Matrix) -> &'a Operator;
        #[cxx_name = "upcast_to_operator_mut"]
        fn Matrix_to_operator_mut<'a>(
            m: Pin<&'a mut Matrix>) -> Pin<&'a mut Operator>;

        #[namespace = "mfem"]
        type OperatorHandle = crate::OperatorHandle;
        fn OperatorHandle_operator<'a>(
            o: &'a OperatorHandle) -> &'a Operator;
        fn OperatorHandle_operator_mut<'a>(
            o: Pin<&'a mut OperatorHandle>) -> Pin<&'a mut Operator>;

        #[namespace = "mfem"]
        type SparseMatrix = crate::SparseMatrix;
        unsafe fn OperatorHandle_SparseMatrix<'a>(
            o: &'a OperatorHandle) -> &'a SparseMatrix;

        #[namespace = "mfem"]
        type Solver = crate::Solver;
        fn PCG(
            a: &Operator,
            solver: Pin<&mut Solver>,
            b: &Vector,
            x: Pin<&mut Vector>,
            print_iter: i32,
            max_num_iter: i32,
            rtol: f64,
            atol: f64);
    }
    impl UniquePtr<Operator> {}
}

// Import into scope all C++ symbols defined above.
pub use ffi::extra::*;
pub use ffi::mfem::*;
pub use ffi_cxx::{
    Matrix_to_operator, Matrix_to_operator_mut,
    c_void, new_FunctionCoefficient,
    Operator, OperatorHandle_operator, OperatorHandle_operator_mut,
    Element,
    OperatorHandle_SparseMatrix,
    PCG,
};

use std::fmt::{Debug, Error, Formatter};

impl Debug for Operator_Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use Operator_Type::*;
        write!(f, "{}", match self {
            ANY_TYPE => "ANY_TYPE",
            MFEM_SPARSEMAT => "MFEM_SPARSEMAT",
            Hypre_ParCSR => "Hypre_ParCSR",
            PETSC_MATAIJ => "PETSC_MATAIJ",
            PETSC_MATIS => "PETSC_MATIS",
            PETSC_MATSHELL => "PETSC_MATSHELL",
            PETSC_MATNEST => "PETSC_MATNEST",
            PETSC_MATHYPRE => "PETSC_MATHYPRE",
            PETSC_MATGENERIC => "PETSC_MATGENERIC",
            Complex_Operator => "Complex_Operator",
            MFEM_ComplexSparseMat => "MFEM_ComplexSparseMat",
            Complex_Hypre_ParCSR => "Complex_Hypre_ParCSR",
            Complex_DenseMat => "Complex_DenseMat",
            MFEM_Block_Matrix => "MFEM_Block_Matrix",
            MFEM_Block_Operator => "MFEM_Block_Operator",
        })
    }
}

#[ctor::ctor]
fn init() {
    if Mfem_MFEM_USE_EXCEPTIONS == 1 {
        set_error_action(ErrorAction::MFEM_ERROR_THROW);
    } else {
        eprintln!(
            "warning: It is recommended to compile `mfem` with \
            MFEM_USE_EXCEPTIONS=YES"
        );
    }
}
