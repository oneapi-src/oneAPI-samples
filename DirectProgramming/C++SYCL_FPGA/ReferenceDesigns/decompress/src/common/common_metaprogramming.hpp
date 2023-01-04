#ifndef __COMMON_METAPROGRAMMING_HPP__
#define __COMMON_METAPROGRAMMING_HPP__

//
// Metaprogamming utility to check if a class has a boolean member named 'flag'
//
namespace detail {
template <typename T, typename = bool>
struct has_flag_bool_impl : std::false_type {};

template <typename T>
struct has_flag_bool_impl<T, decltype(T::flag)> : std::true_type {};
}  // namespace detail

template <typename T>
struct has_flag_bool {
  static constexpr bool value = detail::has_flag_bool_impl<T>{};
};

template <typename T>
inline constexpr bool has_flag_bool_v = has_flag_bool<T>::value;

//
// Metaprogamming utility to check if a class has any member named 'data'
//
namespace detail {
template <typename T, typename = int>
struct has_data_member_impl : std::false_type {};

template <typename T>
struct has_data_member_impl<T, decltype((void)T::data, 0)> : std::true_type {};
}  // namespace detail

template <typename T>
struct has_data_member {
  static constexpr bool value = detail::has_data_member_impl<T>{};
};

template <typename T>
inline constexpr bool has_data_member_v = has_data_member<T>::value;

//
// Metaprogamming utility to check if a class has any member named 'valid_count'
//
namespace detail {
template <typename T, typename = int>
struct has_valid_count_member_impl : std::false_type {};

template <typename T>
struct has_valid_count_member_impl<T, decltype((void)T::valid_count, 0)>
    : std::true_type {};
}  // namespace detail

template <typename T>
struct has_valid_count_member {
  static constexpr bool value = detail::has_valid_count_member_impl<T>{};
};

template <typename T>
inline constexpr bool has_valid_count_member_v =
    has_valid_count_member<T>::value;

#endif /* __COMMON_METAPROGRAMMING_HPP__ */
