//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <utility>

namespace {
template <class Func, template <std::size_t> class Name, std::size_t index>
class SubmitOneComputeUnit {
public:
  SubmitOneComputeUnit(Func &&f, sycl::queue &q) {

    q.submit([&](sycl::handler &h) {
      h.single_task<Name<index>>([=] {
        // verifies that f only takes a single argument
        f(std::integral_constant<std::size_t, index>());
      });
    });
  }
};

template <template <std::size_t> class Name, class Func, std::size_t... index>
inline constexpr void ComputeUnitUnroller(sycl::queue &q, Func &&f,
                                          std::index_sequence<index...>) {
  (SubmitOneComputeUnit<Func, Name, index>(f, q), ...); // fold expression
}
} // namespace

// N is the number of compute units
// Name is the kernel's name
template <std::size_t N, template <std::size_t ID> class Name, class Func>
constexpr void submit_compute_units(sycl::queue &q, Func &&f) {
  std::make_index_sequence<N> indices;
  ComputeUnitUnroller<Name>(q, f, indices);
}
