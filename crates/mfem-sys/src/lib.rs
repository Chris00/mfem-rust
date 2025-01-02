#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

use autocxx::prelude::*;

include_cpp! {
    #include "mfem.hpp"
    #include "extra.hpp"
    safety!(unsafe)
    generate!("Mfem_MFEM_USE_EXCEPTIONS")
    generate!("NumBasisTypes")
    generate!("mfem::ErrorAction")
    generate!("mfem::set_error_action")
    generate!("mfem::Vector")
    concrete!("mfem::Array<int>", ArrayInt)
    generate!("Arrayint_with_len")
    generate!("Arrayint_copy")
    generate!("Arrayint_from_slice")
    generate!("Arrayint_len")
    generate!("Arrayint_GetData")
    generate!("Arrayint_GetDataMut")
    generate!("Mesh_bdr_attributes")

    generate!("mfem::Matrix")
    generate!("mfem::Ordering")
    generate!("mfem::Mesh")
    generate!("mfem::FiniteElement")
    generate!("mfem::FiniteElementCollection")
    generate!("mfem::H1_FECollection")
    // `FiniteElementSpace` contains protected types which are not
    // handled well.  Thus, one must bind by hand.
    // generate!("mfem::FiniteElementSpace")
    generate!("FES")
    generate!("FES_new")
    generate!("FES_GetEssentialTrueDofs")
    generate!("FES_GetTrueVSize")

    generate!("mfem::GridFunction")
    generate!("mfem::LinearFormIntegrator")
    generate!("mfem::BilinearFormIntegrator")
    generate!("mfem::LinearForm")
    generate!("mfem::BilinearForm")
    generate!("mfem::MixedBilinearForm")
    generate!("mfem::Coefficient")
    generate!("mfem::ConstantCoefficient")
    generate!("mfem::OperatorHandle")
    // mfem::LinearFormIntegrator is an abstract class.
    generate!("LFI")
    generate!("mfem::DomainLFIntegrator")
    // mfem::BilinearFormIntegrator is an abstract class.
    generate!("BFI")
    generate!("mfem::DiffusionIntegrator")
}

pub use ffi::*;


#[ctor::ctor]
fn init() {
    if Mfem_MFEM_USE_EXCEPTIONS == 1 {
        mfem::set_error_action(mfem::ErrorAction::MFEM_ERROR_THROW);
    } else {
        eprintln!("warning: It is recommended to compile `mfem` with \
            MFEM_USE_EXCEPTIONS=YES");
    }
}
