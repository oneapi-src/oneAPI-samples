#ifndef __TUPLE_HPP__
#define __TUPLE_HPP__
#pragma once

#include <type_traits>

//
// Generic tuple
//
// USAGE EXAMPLE:
//    Tuple<char,short,int,long> my_tuple;
//    char a = my_tuple.get<0>();
//    short b = my_tuple.get<1>();
//    int c = my_tuple.get<2>();
//    long d = my_tuple.get<3>();
//
template <typename... Tys>
struct Tuple {
  Tuple(Tys... Args) : values(Args...) {}
  Tuple() {}

  //
  // get the index'th item in the tuple of values
  //
  template <int index>
  auto& get() {
    static_assert(index < NumTys, "index out of bounds");
    return get_impl<index, Tys...>(values);
  }

  //
  // helper to get the first element in the tuple
  //
  auto& first() { return get<0>(); }

  //
  // helper to get the last element in the tuple
  //
  auto& last() { return get<NumTys - 1>(); }

 private:
  //
  // generic tuple implementation: recursive case
  //
  template <typename CurrentTy, typename... OtherTys>
  struct tuple_impl {
    tuple_impl(CurrentTy& current, OtherTys... others)
        : value(current), other_values(others...) {}
    tuple_impl() {}

    using ValueTy = CurrentTy;
    ValueTy value;
    tuple_impl<OtherTys...> other_values;
  };

  //
  // generic tuple implementation: base case
  //
  template <typename FinalTy>
  struct tuple_impl<FinalTy> {
    tuple_impl(FinalTy& current) : value(current) {}
    tuple_impl() {}

    using ValueTy = FinalTy;
    ValueTy value;
  };

  // the tuple values
  tuple_impl<Tys...> values;

  // the number of tuple values
  constexpr static auto NumTys = sizeof...(Tys);

  //
  // implementation of 'get' for general tuple
  //
  template <int index, typename HeadTy, typename... TailTys>
  static auto& get_impl(tuple_impl<HeadTy, TailTys...>& sub_tuple) {
    if constexpr (index == 0) {
      // base case
      return sub_tuple.value;
    } else {
      // recursive case
      return get_impl<index - 1, TailTys...>(sub_tuple.other_values);
    }
  }
};

//
// NTuple implementation
// This is convenient way to have N elements of the same type
// somewhat like an std::array
//
template <int, typename Type>
using NTupleElem = Type;

template <typename Type, std::size_t... Idx>
static auto make_NTupleImpl(std::index_sequence<Idx...>)
    -> Tuple<NTupleElem<Idx, Type>...>;

template <int N, typename Type>
using make_NTuple =
    decltype(make_NTupleImpl<Type>(std::make_index_sequence<N>()));

//
// convenience alias for a tuple of N elements of the same type
//
// USAGE EXAMPLE:
//    NTuple<10,int> elements;
//    elements.get<3>() = 17;
//
template <int N, typename Type>
using NTuple = make_NTuple<N, Type>;

#endif /* __TUPLE_HPP__ */
