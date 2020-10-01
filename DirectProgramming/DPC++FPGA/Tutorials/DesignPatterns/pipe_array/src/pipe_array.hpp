//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <utility>

#include "pipe_array_internal.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
  using namespace sycl;
#endif

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
        INTEL::pipe<StructId<idxs...>, BaseTy, depth>;
  };

  template <size_t... idxs>
  using PipeAt = typename VerifyIndices<idxs...>::VerifiedPipe;
};
