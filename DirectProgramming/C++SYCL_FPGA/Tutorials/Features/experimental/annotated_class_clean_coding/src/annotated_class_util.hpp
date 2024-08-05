#ifndef __ANNOTATED_CLASS_UTIL_HPP__
#define __ANNOTATED_CLASS_UTIL_HPP__

#include <sycl/sycl.hpp>

namespace fpga_tools {
///////////////////////////////////////////////////////////////////////////////
//
// This header provides a utility function: "alloc_annotated", which improves
// "malloc_shared" in allocating host/share memory in the following aspects:
//   (1) better code brevity
//   (2) support compile-time type check
//
// "alloc_annotated" function takes an annotated_arg type as the only template
// parameter, and returns an instance of the same template type, with memory
// allocated for the underlying pointer.
// This provides a compile-time guarantee that the properties of the allocated
// memory (for example, buffer location, alignment) match with the annotations
// on the kernel arguments.
//
// To use "alloc_annotated",
//
// 1. include header "annotated_class_util.hpp"
// 2. create a type alias for the "annotated_arg" type, e.g.
//
//  using annotated_arg_t =
//      sycl::ext::oneapi::experimental::annotated_arg<
//          int *, decltype(sycl::ext::oneapi::experimental::properties{
//                          sycl::ext::intel::experimental::buffer_location<1>,
//                          sycl::ext::intel::experimental::dwidth<32>,
//                          sycl::ext::intel::experimental::latency<0>,
//                          sycl::ext::intel::experimental::read_write_mode_write,
//                          sycl::ext::oneapi::experimental::alignment<4>})>;
//
// Furthermore, if you add the "-std=c++20" compiler flag, the type alias
// declaration above can be simplified as:
//
//  using annotated_arg_t = sycl::ext::oneapi::experimental::annotated_arg<
//      int *, fpga_tools::properties_t<
//                  sycl::ext::intel::experimental::buffer_location<1>,
//                  sycl::ext::intel::experimental::dwidth<32>,
//                  sycl::ext::intel::experimental::latency<0>,
//                  sycl::ext::intel::experimental::read_write_mode_write,
//                  sycl::ext::oneapi::experimental::alignment<4>>;
//
// 3. in the host code, replace the USM allocation call (e.g. malloc_shared,
//    aligned_alloc_shared, etc) with:
//
//  annotated_arg_t c = fpga_tools::alloc_annotated<annotated_arg_t>(count, q);
//
//
// NOTE: The template parameter for the "alloc_annotated" function must be
// either an "annotated_arg" with a pointer type, or an "annotated_ptr". Other
// types cause a compiler error.
//
///////////////////////////////////////////////////////////////////////////////

#if __cplusplus >= 202002L

template <auto... Props>
using properties_t =
    decltype(sycl::ext::oneapi::experimental::properties{Props...});

#endif

// Type traits to check if a type is annotated_ptr or
// annotated_arg
template <typename T>
struct is_annotated_class : std::false_type {};

template <typename T, typename... Props>
struct is_annotated_class<sycl::ext::oneapi::experimental::annotated_ptr<
    T, sycl::ext::oneapi::experimental::detail::properties_t<Props...>>>
    : std::true_type {};

template <typename T, typename... Props>
struct is_annotated_class<sycl::ext::oneapi::experimental::annotated_arg<
    T, sycl::ext::oneapi::experimental::detail::properties_t<Props...>>>
    : std::true_type {};

// Type traits to get the underlying raw type of annotated_arg/annotated_ptr
template <typename T>
struct get_raw_type {};

template <typename T, typename... Props>
struct get_raw_type<sycl::ext::oneapi::experimental::annotated_ptr<
    T, sycl::ext::oneapi::experimental::detail::properties_t<Props...>>> {
  using type = T;
};

template <typename T, typename... Props>
struct get_raw_type<sycl::ext::oneapi::experimental::annotated_arg<
    T, sycl::ext::oneapi::experimental::detail::properties_t<Props...>>> {
  static constexpr bool is_annotated_arg_for_pointer = false;
  static_assert(is_annotated_arg_for_pointer,
                "'alloc_annotated' cannot be specified with annotated_arg<T> "
                "as template parameter if T is a non-pointer type");
};

template <typename T, typename... Props>
struct get_raw_type<sycl::ext::oneapi::experimental::annotated_arg<
    T *, sycl::ext::oneapi::experimental::detail::properties_t<Props...>>> {
  using type = T;
};

// Type traits to get the type of the property list in
// annotated_arg/annotated_ptr
template <typename T>
struct get_property_list {};

template <typename T, typename... Props>
struct get_property_list<sycl::ext::oneapi::experimental::annotated_ptr<
    T, sycl::ext::oneapi::experimental::detail::properties_t<Props...>>> {
  using type = sycl::ext::oneapi::experimental::detail::properties_t<Props...>;
};

template <typename T, typename... Props>
struct get_property_list<sycl::ext::oneapi::experimental::annotated_arg<
    T, sycl::ext::oneapi::experimental::detail::properties_t<Props...>>> {
  using type = sycl::ext::oneapi::experimental::detail::properties_t<Props...>;
};

// Type traits to remove alignment from a property list. This is needed for
// because the annotated malloc API does not support compile-time alignment
// property
template <typename T>
struct remove_align_from {};

template <>
struct remove_align_from<
    sycl::ext::oneapi::experimental::empty_properties_t> {
  using type = sycl::ext::oneapi::experimental::empty_properties_t;
};

template <typename Prop, typename... Props>
struct remove_align_from<
    sycl::ext::oneapi::experimental::detail::properties_t<Prop, Props...>> {
  using type = std::conditional_t<
      sycl::ext::oneapi::experimental::detail::HasAlign<
          sycl::ext::oneapi::experimental::detail::properties_t<Prop>>::value,
      sycl::ext::oneapi::experimental::detail::properties_t<Props...>,
      sycl::ext::oneapi::experimental::detail::merged_properties_t<
          sycl::ext::oneapi::experimental::detail::properties_t<Prop>,
          typename remove_align_from<sycl::ext::oneapi::experimental::detail::
                                         properties_t<Props...>>::type>>;
};

template <typename T>
struct split_annotated_type {
  static constexpr bool is_valid_annotated_type = is_annotated_class<T>::value;
  static_assert(is_valid_annotated_type,
                "alloc_annotated function only takes 'annotated_ptr' or "
                "'annotated_arg' type as a template parameter");

  using raw_type = typename get_raw_type<T>::type;
  using all_properties = typename get_property_list<T>::type;
  static constexpr size_t alignment =
      sycl::ext::oneapi::experimental::detail::GetAlignFromPropList<
          all_properties>::value;
  using properties = typename remove_align_from<all_properties>::type;
};

// Wrapper function that allocates USM host memory with compile-time properties
// and returns annotated_ptr
template <typename T>
T alloc_annotated(size_t count, const sycl::queue &q,
                  sycl::usm::alloc usm_kind = sycl::usm::alloc::host) {
  auto ann_ptr = sycl::ext::oneapi::experimental::aligned_alloc_annotated<
      typename split_annotated_type<T>::raw_type,
      typename split_annotated_type<T>::properties>(
      split_annotated_type<T>::alignment, count, q, usm_kind);

  if (ann_ptr.get() == nullptr) {
    std::cerr << "Memory allocation returns null" << std::endl;
    std::terminate();
  }

  return T{ann_ptr.get()};
}

}  // namespace fpga_tools

#endif