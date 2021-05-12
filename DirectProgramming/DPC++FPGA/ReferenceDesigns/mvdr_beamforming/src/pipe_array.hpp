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
//#include <variant>

#include "pipe_array_internal.hpp"

template <class Id,          // identifier for the pipe array
          typename BaseTy,   // type to write/read for each pipe
          size_t min_depth,  // minimum capacity of each pipe
          size_t... dims     // depth of each dimension in the array
                             // any number of dimensions are supported
          >
struct PipeArray {
  PipeArray() = delete;  // ensure we cannot create an instance

  template <size_t... idxs>
  struct StructId;  // the ID of each pipe in the array

  // VerifyIndices checks that we only access pipe indicies that are in range
  template <size_t... idxs>
  struct VerifyIndices {
    static_assert(sizeof...(idxs) == sizeof...(dims),
                  "Indexing into a PipeArray requires as many indices as "
                  "dimensions of the PipeArray.");
    static_assert(VerifierDimLayer<dims...>::template VerifierIdxLayer<
                      idxs...>::IsValid(),
                  "Index out of bounds");
    using VerifiedPipe =
        cl::sycl::INTEL::pipe<StructId<idxs...>, BaseTy, min_depth>;
  };

  // helpers for accessing the dimensions of the pipe array
  // usage:
  //  MyPipeArray::GetNumDims() - number of dimensions in this pipe array
  //  MyPipeArray::GetDimSize<3>() - size of dimension 3 in this pipe array
  static constexpr size_t GetNumDims() { return (sizeof...(dims)); }
  template <int dim_num>
  static constexpr size_t GetDimSize() {
    return std::get<dim_num>(dims...);
  }

  // PipeAt<idxs...> is used to reference a pipe at a particular index
  template <size_t... idxs>
  using PipeAt = typename VerifyIndices<idxs...>::VerifiedPipe;

  // functor to impllement blocking write to all pipes in the array
  template <std::size_t... I>
  struct BlockingWriteFunc {
    void operator()(const BaseTy &data, bool &success) const {
      PipeAt<I...>::write(data);
    }
  };
  // functor to impllement non-blocking write to all pipes in the array
  template <std::size_t... I>
  struct NonBlockingWriteFunc {
    void operator()(const BaseTy &data, bool &success) const {
      PipeAt<I...>::write(data, success);
    }
  };
  // helper function for implementing write() call to all pipes in the array
  template <template <std::size_t...> class WriteFunc,
            typename... IndexSequences>
  static void write_currying_helper(const BaseTy &data, bool &success,
                                    IndexSequences...) {
    write_currying<WriteFunc, BaseTy, std::index_sequence<>,
                   IndexSequences...>()(data, success);
  }

  // blocking write
  // write the same data to all pipes in the array using blocking writes
  static void write(const BaseTy &data) {
    bool success;  // temporary variable, ignored in BlockingWriteFunc
    write_currying_helper<BlockingWriteFunc>(
        data, success, std::make_index_sequence<dims>()...);
  }

  // non-blocking write
  // write the same data to all pipes in the array using non-blocking writes
  static void write(const BaseTy &data, bool &success) {
    write_currying_helper<NonBlockingWriteFunc>(
        data, success, std::make_index_sequence<dims>()...);
  }

};  // end of struct PipeArray

#endif /* __PIPE_ARRAY_HPP__ */
