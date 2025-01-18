// Disable the surious warnings for the mfem header file.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#include "mfem.hpp"
#pragma GCC diagnostic pop

#ifndef RUST_EXTRA_H
#define RUST_EXTRA_H

using namespace mfem;

// Easily extract what we have defined.
namespace extra {
#ifdef MFEM_USE_EXCEPTIONS
    const int Mfem_MFEM_USE_EXCEPTIONS = 1;
#else
    const int Mfem_MFEM_USE_EXCEPTIONS = 0;
#endif

    const int NumBasisTypes = mfem::BasisType::NumBasisTypes;

    using FES = mfem::FiniteElementSpace;

#define ARRAY(T)                                                       \
    using Array##T = Array<T>;                                         \
    std::unique_ptr<Array##T> Array##T##_with_len(int len) {           \
        return std::make_unique<Array##T>(len);                        \
    }                                                                  \
    std::unique_ptr<Array##T> Array##T##_copy(Array##T const& src) {   \
        return std::make_unique<Array##T>(src);                        \
    }                                                                  \
    std::unique_ptr<Array##T> Array##T##_from_slice(                   \
        T* data, int len) {                                            \
        return std::make_unique<Array##T>(data, len);                  \
    }                                                                  \
    int Array##T##_len(Array##T const& a) {                            \
        return a.Size();                                               \
    }                                                                  \
    const T* Array##T##_GetData(Array##T const& a) {                   \
        return a.GetData();                                            \
    }                                                                  \
    T* Array##T##_GetDataMut(Array##T& a) {                            \
        return a.GetData();                                            \
    }

    using Int = int;
    ARRAY(Int)

    ArrayInt const& Mesh_bdr_attributes(Mesh const& mesh) {
        return mesh.bdr_attributes;
    }

    // Immutable version
    FiniteElementCollection const* GridFunction_OwnFEC(GridFunction const& gf)
    {
        return const_cast<GridFunction&>(gf).OwnFEC();
    }

    std::unique_ptr<FES> FES_new(
        Mesh const& mesh,
        FiniteElementCollection const& fec,
        int vdim,
        Ordering::Type ordering)
    {
        // FIXME: Make sure mesh is not modified.
        auto& mut_mesh = const_cast<Mesh&>(mesh);
        return std::make_unique<FES>(&mut_mesh, &fec, vdim, ordering);
    }

    int FES_GetTrueVSize(FiniteElementSpace const& fes) {
        return fes.GetTrueVSize();
    }

    void FES_GetEssentialTrueDofs(
        FiniteElementSpace const& fespace,
        ArrayInt const& bdr_attr_is_ess,
        ArrayInt& ess_tdof_list,
        int component)
    {
        // FIXME: make sure fespace is not modified
        auto& mut_fespace = const_cast<FiniteElementSpace&>(fespace);
        mut_fespace.GetEssentialTrueDofs(
            bdr_attr_is_ess, ess_tdof_list, component);
    }

    using LFI = mfem::LinearFormIntegrator;
    using BFI = mfem::BilinearFormIntegrator;

    // Array<int> detected.
    inline void BilinearForm_FormLinearSystem(
        BilinearForm &bf,
        const ArrayInt &ess_tdof_list,
        Vector &x,
        Vector &b,
        OperatorHandle &A,
        Vector &X,
        Vector &B,
        int copy_interior)
    {
        bf.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
    }
}

#endif // RUST_EXTRA_H
