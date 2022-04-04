#ifndef __METAPROGRAMMING_UTILS_HPP__
#define __METAPROGRAMMING_UTILS_HPP__

#include <type_traits>

namespace fpga_tools {

//
// The code below creates the constexprs 'make_integer_range'
// and 'make_index_range' these are akin to 'std::make_integer_sequence'
// and 'std::make_index_sequence', respectively.
// However they allow you to specificy a range and can either increment
// or decrement, rather than a strict increasing sequence
//
template <typename T, typename, T begin, bool increase>
struct integer_range_impl;

// incrementing case
template <typename T, T... N, T begin>
struct integer_range_impl<T, std::integer_sequence<T, N...>, begin, true> {
  using type = std::integer_sequence<T, N + begin...>;
};

// decrementing case
template <typename T, T... N, T begin>
struct integer_range_impl<T, std::integer_sequence<T, N...>, begin, false> {
  using type = std::integer_sequence<T, begin - N...>;
};

// integer_range
template <typename T, T begin, T end>
using integer_range = typename integer_range_impl<
    T, std::make_integer_sequence<T, (begin < end) ? end - begin : begin - end>,
    begin, (begin < end)>::type;

//
// make_integer_range
//
// USAGE:
//    make_integer_range<int,1,10>{} ==> 1,2,...,9
//    make_integer_range<int,10,1>{} ==> 10,9,...,2
//
template <class T, T begin, T end>
using make_integer_range = integer_range<T, begin, end>;

//
// make_index_range
//
// USAGE:
//    make_index_range<1,10>{} ==> 1,2,...,9
//    make_index_range<10,1>{} ==> 10,9,...,2
//
template <std::size_t begin, std::size_t end>
using make_index_range = integer_range<std::size_t, begin, end>;

//
// The code below creates the constexprs 'make_integer_pow2_sequence'
// and 'make_index_pow2_sequence'. These generate the sequence
// 2^0, 2^1, 2^2, ... , 2^(N-1) = 1,2,4,...,2^(N-1)
//
template <typename T, typename>
struct integer_pow2_sequence_impl;

template <typename T, T... Pows>
struct integer_pow2_sequence_impl<T, std::integer_sequence<T, Pows...>> {
  using type = std::integer_sequence<T, (T(1) << Pows)...>;
};

// integer_pow2_sequence
template <typename T, T N>
using integer_pow2_sequence =
    typename integer_pow2_sequence_impl<T,
                                        std::make_integer_sequence<T, N>>::type;

//
// make_integer_pow2_sequence
//
// USAGE:
//    make_integer_pow2_sequence<int,5>{} ==> 1,2,4,8,16
//
template <class T, T N>
using make_integer_pow2_sequence = integer_pow2_sequence<T, N>;

//
// make_index_pow2_sequence
//
// USAGE:
//    make_index_pow2_sequence<5>{} ==> 1,2,4,8,16
//
template <std::size_t N>
using make_index_pow2_sequence = integer_pow2_sequence<std::size_t, N>;

//
// Checks for existence of subscript operator
//
namespace detail {
template <typename... >
using void_t = void;

template<class T, typename = void>
struct has_subscript_impl : std::false_type { };

template<typename T>
struct has_subscript_impl<T, void_t<decltype(std::declval<T>()[1])>> 
  : std::true_type { };
}  // namespace detail

template <typename T>
struct has_subscript {
  static constexpr bool value =
    std::is_same_v<typename detail::has_subscript_impl<T>::type, std::true_type>;
};

template <typename T>
inline constexpr bool has_subscript_v = has_subscript<T>::value;

//
// checks if a type is any instance of SYCL pipe
//
namespace detail {

template<typename T>
struct is_sycl_pipe_impl : std::false_type {};

template<typename Id, typename T, std::size_t N>
struct is_sycl_pipe_impl<sycl::ext::intel::pipe<Id, T, N>> : std::true_type {};

}  // namespace detail

template <typename T>
struct is_sycl_pipe {
  static constexpr bool value = detail::is_sycl_pipe_impl<T>{};
};

template <typename T>
inline constexpr bool is_sycl_pipe_v = is_sycl_pipe<T>::value;

} // namespace fpga_tools

#endif  /* __METAPROGRAMMING_UTILS_HPP__ */