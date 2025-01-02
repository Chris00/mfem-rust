#include "mfem.hpp"

#ifndef RUST_EXTRA_H
#define RUST_EXTRA_H

#ifdef MFEM_USE_EXCEPTIONS
const int Mfem_MFEM_USE_EXCEPTIONS = 1;
#else
const int Mfem_MFEM_USE_EXCEPTIONS = 0;
#endif

const int NumBasisTypes = mfem::BasisType::NumBasisTypes;

using FES = mfem::FiniteElementSpace;

using namespace mfem;

#define ARRAY(T) \
    std::unique_ptr<Array<T>> Array##T##_with_len(int len) {      \
        return std::make_unique<Array<T>>(len);                        \
    }                                                                  \
    std::unique_ptr<Array<T>> Array##T##_copy(Array<T> const& src) { \
        return std::make_unique<Array<T>>(src);                        \
    }                                                                  \
    std::unique_ptr<Array<T>> Array##T##_from_slice(              \
        T* data, int len) {                                            \
        return std::make_unique<Array<T>>(data, len);                  \
    }                                                                  \
    int Array##T##_len(Array<T> const& a) {                       \
        return a.Size();                                               \
    }                                                                  \
    const T* Array##T##_GetData(Array<T> const& a) {              \
        return a.GetData();                                            \
    }                                                                  \
    T* Array##T##_GetDataMut(Array<T>& a) {                       \
        return a.GetData();                                            \
    }

ARRAY(int)

Array<int> const& Mesh_bdr_attributes(Mesh const& mesh) {
    return mesh.bdr_attributes;
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
    Array<int> const& bdr_attr_is_ess,
    Array<int>& ess_tdof_list,
    int component)
{
    // FIXME: make sure fespace is not modified
    auto& mut_fespace = const_cast<FiniteElementSpace&>(fespace);
    mut_fespace.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list, component);
}

using LFI = mfem::LinearFormIntegrator;
using BFI = mfem::BilinearFormIntegrator;

#endif // RUST_EXTRA_H
