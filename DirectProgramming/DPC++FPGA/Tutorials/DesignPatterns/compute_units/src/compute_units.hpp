//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <utility>

namespace {
template <typename Func, template <std::size_t> typename Name,
          std::size_t Index>
class SubmitOneComputeUnit {
public:
  SubmitOneComputeUnit(Func &&f, sycl::queue &q) {

    q.submit([&](sycl::handler &h) {
      h.single_task<Name<Index>>([=] {
        static_assert(
            std::is_invocable<
                Func, std::integral_constant<std::size_t, Index>>::value,
            "The callable Func passed to SubmitComputeUnits must take a single "
            "argument of type auto");
        f(std::integral_constant<std::size_t, Index>());
      });
    });
  }
};

template <template <std::size_t> typename Name, typename Func,
          std::size_t... Indices>
inline constexpr void ComputeUnitUnroller(sycl::queue &q, Func &&f,
                                          std::index_sequence<Indices...>) {
  (SubmitOneComputeUnit<Func, Name, Indices>(f, q), ...); // fold expression
}
} // namespace

template <std::size_t N,                           // Number of compute units
          template <std::size_t ID> typename Name, // Name for the compute units
          typename Func>                           // Callable defining compute
                                                   // units' functionality

// Func must take a single argument. This argument is the compute unit's ID.
// The compute unit ID is a constexpr, and it can be used to specialize
// the kernel's functionality.
// Note: the type of Func's single argument must be 'auto', because Func
// will be called with various indices (i.e., the ID for each compute unit)
constexpr void SubmitComputeUnits(sycl::queue &q, Func &&f) {
  std::make_index_sequence<N> indices;
  ComputeUnitUnroller<Name>(q, f, indices);
}
