#include "mfem.hpp"
#include "cxx.h"

#ifndef FFI_CXX_H
#define FFI_CXX_H

using namespace mfem;

template<typename T>
Operator const& upcast_to_operator(T const& x) {
    return x;
}

template<typename T>
Operator& upcast_to_operator_mut(T& x) {
    return x;
}

Operator const& OperatorHandle_operator(OperatorHandle const& x) {
    return *x;
}

Operator& OperatorHandle_operator_mut(OperatorHandle& x) {
    return *x;
}

SparseMatrix const& OperatorHandle_SparseMatrix(OperatorHandle const& x) {
    return *x.As<SparseMatrix>();
}

using c_void = void;

std::unique_ptr<FunctionCoefficient>
new_FunctionCoefficient(rust::Fn<double(mfem::Vector const &, void*)> f,
                        void *d)
{
  std::function<real_t(const Vector &)> F =
      [d = std::move(d), f = std::move(f)](mfem::Vector const &x) {
      return f(x, d);
  };
  return std::make_unique<FunctionCoefficient>(F);
}


#endif // FFI_CXX_H
