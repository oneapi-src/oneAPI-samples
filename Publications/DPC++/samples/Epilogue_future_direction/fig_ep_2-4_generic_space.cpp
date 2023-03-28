// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>

using namespace sycl;

#if defined(EXPLICIT_ADDRESS_SPACE)
// Pointers in structs must be explicitly decorated with address space
// Supporting both address spaces requires a template parameter
template <access::address_space AddressSpace>
struct Particles {
  multi_ptr<float, AddressSpace> x;
  multi_ptr<float, AddressSpace> y;
  multi_ptr<float, AddressSpace> z;
};
#elif defined(GENERIC_ADDRESS_SPACE)
// Pointers in structs default to the generic address space
struct Particles {
  float* x;
  float* y;
  float* z;
};
#elif defined(OPTIONAL_ADDRESS_SPACE)
// Template parameter defaults to generic address space
// User of class can override address space for performance tuning
template <access::address_space AddressSpace =
          access::address_space::generic_space>
struct Particles {
  multi_ptr<float, AddressSpace> x;
  multi_ptr<float, AddressSpace> y;
  multi_ptr<float, AddressSpace> z;
};
#else
#error "Must define one of: EXPLICIT_ADDRESS_SPACE, GENERIC_ADDRESS_SPACE, OPTIONAL_ADDRESS_SPACE
#endif

int main() {

  queue Q;
  constexpr int N = 1024;

  float* x = malloc_shared<float>(N, Q);
  float* y = malloc_shared<float>(N, Q);
  float* z = malloc_shared<float>(N, Q);
#if defined(EXPLICIT_ADDRESS_SPACE)
  Particles<access::address_space::global_space> particles{x, y, z};
#elif defined(GENERIC_ADDRESS_SPACE) || defined(OPTIONAL_ADDRESS_SPACE)
  Particles particles{x, y, z};
#endif

  Q.parallel_for(range{N}, [=](id<1> idx) {
    x[idx] = 1;
    y[idx] = 2;
    z[idx] = 3;
  }).wait();

  free(x, Q);
  free(y, Q);
  free(z, Q);
}
