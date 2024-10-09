// REQUIRES: linux
// RUN: %{build} %device_sanitizer_flags -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_ENABLE_LAYERS=UR_LAYER_ASAN %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK %s
#include <sycl/sycl.hpp>

constexpr size_t N = 64;

int main() {
  sycl::queue Q;
  auto *data = new int[N];
  sycl::free(data, Q);
  return 0;
}
