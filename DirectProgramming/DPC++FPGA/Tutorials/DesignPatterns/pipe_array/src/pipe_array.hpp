//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#ifndef __PIPE_ARRAY_HPP__
#define __PIPE_ARRAY_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <utility>

#include "pipe_array_internal.hpp"

template <class Id, typename BaseTy, size_t depth, size_t... dims>
struct PipeArray {
  PipeArray() = delete;

  template <size_t... idxs>
  struct StructId;

  template <size_t... idxs>
  struct VerifyIndices {
    static_assert(sizeof...(idxs) == sizeof...(dims),
                  "Indexing into a PipeArray requires as many indices as "
                  "dimensions of the PipeArray.");
    static_assert(VerifierDimLayer<dims...>::template VerifierIdxLayer<
                      idxs...>::IsValid(),
                  "Index out of bounds");
    using VerifiedPipe =
        cl::sycl::INTEL::pipe<StructId<idxs...>, BaseTy, depth>;
  };

  template <size_t... idxs>
  using PipeAt = typename VerifyIndices<idxs...>::VerifiedPipe;
};

#endif /* __PIPE_ARRAY_HPP__ */
