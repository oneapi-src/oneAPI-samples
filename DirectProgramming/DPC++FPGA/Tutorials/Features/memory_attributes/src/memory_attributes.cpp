//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// constants for this tutorial
constexpr size_t kRows = 8;
constexpr size_t kVec = 4;
constexpr size_t kMaxVal = 512;
constexpr size_t kNumTests = 64;
constexpr size_t kMaxIter = 8;

// the kernel class name
// templating allows us to easily instantiate different versions of the kernel
template<int AttrType>
class Kernel;

// The shared compute function for host and device code
size_t Compute(unsigned init, unsigned dict_offset[][kVec]) {
  // We do not provide any attributes for compare_offset and hash;
  // we let the compiler decide what's best based on the access pattern
  // and their size.
  unsigned compare_offset[kVec][kVec];
  unsigned hash[kVec];

  #pragma unroll
  for (size_t i = 0; i < kVec; i++) {
    hash[i] = (++init) & (kRows - 1);
  }

  size_t count = 0, iter = 0;
  do {
    // After unrolling both loops, we have kVec*kVec reads from dict_offset
    #pragma unroll
    for (size_t i = 0; i < kVec; i++) {
      #pragma unroll
      for (size_t k = 0; k < kVec; ++k) {
        compare_offset[k][i] = dict_offset[hash[i]][k];
      }
    }

    // After unrolling, we have kVec writes to dict_offset
    #pragma unroll
    for (size_t k = 0; k < kVec; ++k) {
      dict_offset[hash[k]][k] = (init << k);
    }
    init++;

    #pragma unroll
    for (size_t i = 0; i < kVec; i++) {
      #pragma unroll
      for (size_t k = 0; k < kVec; ++k) {
        count += compare_offset[i][k];
      }
    }
  } while (++iter < kMaxIter);
  return count;
}

// We use partial template specialization to apply different attributes to the
// 'dict_offset' variable in the kernel.
// This serves as a baseline implementation where no attributes are applied
// to the variable. The compiler uses heuristics to try and find the best
// configuration
template<int AttrType>
event submitKernel(queue& q, unsigned init, buffer<unsigned, 1>& d_buf,
                      buffer<unsigned, 1>& r_buf) {
  auto e = q.submit([&](handler &h) {
    accessor d_accessor { d_buf, h, read_only };
    accessor r_accessor { r_buf, h, write_only, noinit };

    h.single_task<Kernel<AttrType>>([=]() [[intel::kernel_args_restrict]] {
      // Declare 'dict_offset' whose attributes are applied based on AttrType
      unsigned dict_offset[kRows][kVec];

      // Initialize 'dict_offset' with values from global memory.
      for (size_t i = 0; i < kRows; ++i) {
        #pragma unroll
        for (size_t k = 0; k < kVec; ++k) {
          // After unrolling, we end up with kVec writes to dict_offset.
          dict_offset[i][k] = d_accessor[i * kVec + k];
        }
      }

      // compute the result
      r_accessor[0] = Compute(init, dict_offset);
    });
  });

  return e;
}

// Define version 1 of the kernel - using a single pumped memory 
template<>
event submitKernel<1>(queue& q, unsigned init, buffer<unsigned, 1>& d_buf,
                      buffer<unsigned, 1>& r_buf) {
  auto e = q.submit([&](handler &h) {
    accessor d_accessor { d_buf, h, read_only };
    accessor r_accessor { r_buf, h, write_only, noinit };

    h.single_task<Kernel<1>>([=]() [[intel::kernel_args_restrict]] {
      // Declare 'dict_offset' whose attributes are applied based on AttrType
      [[intelfpga::singlepump,
        intelfpga::memory("MLAB"),
        intelfpga::numbanks(kVec),
        intelfpga::max_replicates(kVec)]]
      unsigned dict_offset[kRows][kVec];

      // Initialize 'dict_offset' with values from global memory.
      for (size_t i = 0; i < kRows; ++i) {
        #pragma unroll
        for (size_t k = 0; k < kVec; ++k) {
          // After unrolling, we end up with kVec writes to dict_offset.
          dict_offset[i][k] = d_accessor[i * kVec + k];
        }
      }

      // compute the result
      r_accessor[0] = Compute(init, dict_offset);
    });
  });

  return e;
}

// Define version 2 of the kernel - using a double pumped memory 
template<>
event submitKernel<2>(queue& q, unsigned init, buffer<unsigned, 1>& d_buf,
                      buffer<unsigned, 1>& r_buf) {
  auto e = q.submit([&](handler &h) {
    accessor d_accessor { d_buf, h, read_only };
    accessor r_accessor { r_buf, h, write_only, noinit };

    h.single_task<Kernel<2>>([=]() [[intel::kernel_args_restrict]] {
      // Declare 'dict_offset' whose attributes are applied based on AttrType
      [[intelfpga::doublepump,
        intelfpga::memory("MLAB"),
        intelfpga::numbanks(kVec),
        intelfpga::max_replicates(kVec)]]
      unsigned dict_offset[kRows][kVec];

      // Initialize 'dict_offset' with values from global memory.
      for (size_t i = 0; i < kRows; ++i) {
        #pragma unroll
        for (size_t k = 0; k < kVec; ++k) {
          // After unrolling, we end up with kVec writes to dict_offset.
          dict_offset[i][k] = d_accessor[i * kVec + k];
        }
      }

      // compute the result
      r_accessor[0] = Compute(init, dict_offset);
    });
  });

  return e;
}

template<int AttrType>
unsigned RunKernel(unsigned init, const unsigned dict_offset_init[]) {
  unsigned result = 0;

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  try {
    queue q(device_selector, dpc_common::exception_handler);

    // Flatten the 2D array to a 1D buffer, because the
    // buffer constructor requires a pointer to input data
    // that is contiguous in memory.
    buffer<unsigned, 1> d_buf(dict_offset_init, range<1>(kRows * kVec));
    buffer<unsigned, 1> r_buf(&result, 1);

    // submit the kernel
    auto e = submitKernel<AttrType>(q, init, d_buf, r_buf);

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  return result;
}

// This host side function performs the same computation as the device side
// kernel, and is used to verify functional correctness.
unsigned GoldenRun(unsigned init, unsigned const dict_offset_init[]) {
  unsigned dict_offset[kRows][kVec];
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t k = 0; k < kVec; ++k) {
      dict_offset[i][k] = dict_offset_init[i * kVec + k];
    }
  }
  return Compute(init, dict_offset);
}

int main() {
  srand(0);

  unsigned dict_offset_init[kRows*kVec];

  bool passed = true;

  for (size_t j = 0; j < kNumTests; j++) {
    unsigned init = rand() % kMaxVal;
    unsigned int dict_offset_init[kRows * kVec];

    // initialize input data with random values
    for (size_t i = 0; i < kRows; ++i) {
      for (size_t k = 0; k < kVec; ++k) {
        dict_offset_init[i * kVec + k] = rand() % kMaxVal;
      }
    }

    // compute the golden result
    unsigned golden_result = GoldenRun(init, dict_offset_init);

    // run the kernel with 'singlepump' memory attribute
    unsigned result_sp = RunKernel<1>(init, dict_offset_init);

    if (!(result_sp == golden_result)) {
      passed = false;
      std::cout << "  Test#" << j
                << ": mismatch: " << result_sp << " != " << golden_result
                << " (result_sp != golden_result)\n";
    }

    // run the kernel with 'doublepump' memory attribute
    unsigned result_dp = RunKernel<2>(init, dict_offset_init);

    if (!(result_dp == golden_result)) {
      passed = false;
      std::cout << "  Test#" << j
                << ": mismatch: " << result_dp << " != " << golden_result
                << " (result_dp != golden_result)\n";
    }

    // run the kernel with no memory attributes
    unsigned result_na = RunKernel<0>(init, dict_offset_init);

    if (!(result_na == golden_result)) {
      passed = false;
      std::cout << "  Test#" << j
                << ": mismatch: " << result_na << " != " << golden_result
                << " (result_na != golden_result)\n";
    }
  }

  if (passed) {
    std::cout << "PASSED: all kernel results are correct.\n";
  } else {
    std::cout << "FAILED\n";
    return 1;
  }

  return 0;
}
