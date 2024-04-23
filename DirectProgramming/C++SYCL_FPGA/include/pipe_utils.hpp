//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#ifndef __PIPE_UTILS_HPP__
#define __PIPE_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <utility>

/*

This header defines the following utilities for use with pipes in SYCL FPGA
designs.

1. PipeArray

      Create a collection of pipes that can be indexed like an array.

      template <class Id,          // identifier for the pipe array
                typename BaseTy,   // type to write/read for each pipe
                size_t min_depth,  // minimum capacity of each pipe
                size_t... dims     // depth of each dimension in the array
                                   // any number of dimensions are supported
                >
      struct PipeArray

      Example usage:
    
      class PipeArrayId;
      constexpr int min_depth = 0;
      constexpr int num_pipes = 4;
      using MyPipeArray = PipeArray<PipeArrayId, int, min_depth, num_pipes>;
      ...
      constexpr int pipe_idx = 1;
      MyPipeArray::PipeAt<pipe_idx>::read(); 

2. PipeDuplicator

      Fan-out a single pipe write to multiple pipe instances,
      each of which will receive the same data.
      A blocking write will perform a blocking write to each pipe.
      A non-blocking write will perform a non-blocking write to each pipe,
      and set success to true only if ALL writes were successful.

      Note that the special case of 0 pipe instances is supported, which can 
      be useful as a stub for writes to pipes that are not needed in your particular 
      design.

      template <class Id,          // name of this PipeDuplicator
                typename T,        // data type to transfer
                typename... Pipes  // all pipes to send duplicated writes to
                >
      struct PipeDuplicator

      Example usage:

      class PipeID1;
      class PipeID2;
      using MyPipe1 = sycl::ext::intel::pipe<PipeID1, int>;
      using MyPipe2 = sycl::ext::intel::pipe<PipeID2, int>;

      class PipeDuplicatorID;
      using MyPipeDuplicator = PipeDuplicator<PipeDuplicatorID, int, MyPipe1, MyPipe2>;
      ...
      MyPipeDuplicator::write(1); // write the value 1 to both MyPipe1 and MyPipe2

*/

// =============================================================
// Internal Helper Functions/Structs
// =============================================================

namespace fpga_tools {
namespace detail {

// Templated classes for verifying dimensions when accessing elements in the
// pipe array.
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

// Templated classes to perform 'currying' write to all pipes in the array
// Primary template, dummy
template <template <std::size_t...> class WriteFunc, typename BaseTy,
          typename PartialSequence, typename... RemainingSequences>
struct write_currying {};
// Induction case
template <template <std::size_t...> class WriteFunc, typename BaseTy,
          std::size_t... I, std::size_t... J, typename... RemainingSequences>
struct write_currying<WriteFunc, BaseTy, std::index_sequence<I...>,
                      std::index_sequence<J...>, RemainingSequences...> {
  void operator()(const BaseTy &data, bool &success) const {
    (write_currying<WriteFunc, BaseTy, std::index_sequence<I..., J>,
                    RemainingSequences...>()(data, success),
     ...);
  }
};
// Base case
template <template <std::size_t...> class WriteFunc, typename BaseTy,
          std::size_t... I>
struct write_currying<WriteFunc, BaseTy, std::index_sequence<I...>> {
  void operator()(const BaseTy &data, bool &success) const {
    WriteFunc<I...>()(data, success);
  }
};

}  // namespace detail

// =============================================================
// PipeArray
// =============================================================

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
    static_assert(fpga_tools::detail::VerifierDimLayer<dims...>::template
                  VerifierIdxLayer<idxs...>::IsValid(),
                  "Index out of bounds");
    using VerifiedPipe =
        sycl::ext::intel::pipe<StructId<idxs...>, BaseTy, min_depth>;
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
    fpga_tools::detail::write_currying<WriteFunc, BaseTy,
                   std::index_sequence<>, IndexSequences...>()(data, success);
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

// =============================================================
// PipeDuplicator
// =============================================================

// Connect a kernel that writes to a single pipe to multiple pipe instances,
// each of which will receive the same data.
// A blocking write will perform a blocking write to each pipe.  A non-blocking
// write will perform a non-blocking write to each pipe, and set success to
// true only if ALL writes were successful.

// primary template, dummy
template <class Id,          // name of this PipeDuplicator
          typename T,        // data type to transfer
          typename... Pipes  // all pipes to send duplicated writes to
          >
struct PipeDuplicator {};

// recursive case, write to each pipe
template <class Id,                   // name of this PipeDuplicator
          typename T,                 // data type to transfer
          typename FirstPipe,         // at least one output pipe
          typename... RemainingPipes  // additional copies of the output pipe
          >
struct PipeDuplicator<Id, T, FirstPipe, RemainingPipes...> {
  PipeDuplicator() = delete;  // ensure we cannot create an instance

  // Non-blocking write
  static void write(const T &data, bool &success) {
    bool local_success;
    FirstPipe::write(data, local_success);
    success = local_success;
    PipeDuplicator<Id, T, RemainingPipes...>::write(data, local_success);
    success &= local_success;
  }

  // Blocking write
  static void write(const T &data) {
    FirstPipe::write(data);
    PipeDuplicator<Id, T, RemainingPipes...>::write(data);
  }
};

// base case for recursion, no pipes to write to
// also useful as a 'null' pipe, writes don't do anything
template <class Id,   // name of this PipeDuplicator
          typename T  // data type to transfer
          >
struct PipeDuplicator<Id, T> {
  PipeDuplicator() = delete;  // ensure we cannot create an instance

  // Non-blocking write
  static void write(const T & /*data*/, bool &success) { success = true; }

  // Blocking write
  static void write(const T & /*data*/) {
    // do nothing
  }
};

} // namespace fpga_tools

#endif /* __PIPE_UTILS_HPP__ */
