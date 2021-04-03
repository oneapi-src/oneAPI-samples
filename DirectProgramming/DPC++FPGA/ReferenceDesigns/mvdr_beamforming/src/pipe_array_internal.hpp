//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef __PIPE_ARRAY_INTERNAL_HPP__
#define __PIPE_ARRAY_INTERNAL_HPP__

namespace {

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

}  // namespace

#endif /* __PIPE_ARRAY_INTERNAL_HPP__ */
