#ifndef __UNROLLEDLOOP_HPP__
#define __UNROLLEDLOOP_HPP__

#include <type_traits>
#include <utility>

#include "metaprogramming_utils.hpp"

namespace fpga_tools {
///////////////////////////////////////////////////////////////////////////////
//
// Example usage for UnrolledLoop constexpr:
//
// Base
//    UnrolledLoop(std::integer_sequence<int,5,2,7,8>{},[&](auto i) {
//      /* i = 5,2,7,8 */
//    });
//
// Case A
//    UnrolledLoop<10>([&](auto i) {
//      /* i = 0,1,...,9 */
//    });
//
// Case B
//    UnrolledLoop<10>([&](auto i) {
//      /* i = 0,1,...,9 */
//    });
//
// Case C
//    UnrolledLoop<char, 1, 10>([&](auto i) {
//      /* i = 1,2,...,9 */
//    });
//    UnrolledLoop<char, 10, 1>([&](auto i) {
//      /* i = 10,9,...,2 */
//    });
//
// Case D
//    UnrolledLoop<1, 10>([&](auto i) {
//      /* i = 1,2,...,9 */
//    });
//    UnrolledLoop<10, 1>([&](auto i) {
//      /* i = 10,9,...,2 */
//    });
//
///////////////////////////////////////////////////////////////////////////////

//
// Base implementation
// Templated on:
//    ItType    - the type of the iterator (size_t, int, char, ...)
//    ItType... - the indices to iterate on
//    F         - the function to run for each index (i.e. the lambda)
//
template <class ItType, ItType... inds, class F>
constexpr void UnrolledLoop(std::integer_sequence<ItType, inds...>, F&& f) {
  (f(std::integral_constant<ItType, inds>{}), ...);
}

//
// Convience implementation (A)
// performs UnrolledLoop in range [0,n) with iterator of type ItType
//
template <class ItType, ItType n, class F>
constexpr void UnrolledLoop(F&& f) {
  UnrolledLoop(std::make_integer_sequence<ItType, n>{}, std::forward<F>(f));
}

//
// Convenience implementation (B)
// performs UnrolledLoop in range [0,n) with an iterator of type std::size_t
//
template <std::size_t n, class F>
constexpr void UnrolledLoop(F&& f) {
  UnrolledLoop(std::make_index_sequence<n>{}, std::forward<F>(f));
}

//
// Convenience implementation (C)
// performs UnrolledLoop from start...end with an iterator of type ItType
// NOTE:  start is INCLUSIVE, end is EXCLUSIVE
// NOTE:  if start<=end, sequence is start,start+1,...,end-1
//        if end<=start, sequence is start,start-1,...,end+1
//
template <class ItType, ItType start, ItType end, class F>
constexpr void UnrolledLoop(F&& f) {
  UnrolledLoop(make_integer_range<ItType, start, end>{}, std::forward<F>(f));
}

//
// Convenience implementation (D)
// performs UnrolledLoop from start...end with an iterator of type size_t
// NOTE:  start is INCLUSIVE, end is EXCLUSIVE
// NOTE:  if start<=end, sequence is start,start+1,...,end-1
//        if end<=start, sequence is start,start-1,...,end+1
//
template <std::size_t start, std::size_t end, class F>
constexpr void UnrolledLoop(F&& f) {
  UnrolledLoop(make_index_range<start, end>{}, std::forward<F>(f));
}

}  // namespace fpga_tools

#endif /* __UNROLLEDLOOP_HPP__ */