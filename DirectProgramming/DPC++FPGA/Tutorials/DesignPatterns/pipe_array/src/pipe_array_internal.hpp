//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef __PIPE_ARRAY_INTERNAL_HPP__
#define __PIPE_ARRAY_INTERNAL_HPP__

namespace {
template <size_t dim1, size_t... dims>
struct VerifierDimLayer {
  template <size_t idx1, size_t... idxs>
  struct VerifierIdxLayer {
    static constexpr bool IsValid() {
      return idx1 < dim1 &&
             (VerifierDimLayer<dims...>::template VerifierIdxLayer<
                 idxs...>::IsValid());
    }
  };
};
template <size_t dim>
struct VerifierDimLayer<dim> {
  template <size_t idx>
  struct VerifierIdxLayer {
    static constexpr bool IsValid() { return idx < dim; }
  };
};
}  // namespace

#endif /* __PIPE_ARRAY_INTERNAL_HPP__ */
