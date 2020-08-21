//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

using namespace sycl;

constexpr size_t kRows = 8;
constexpr size_t kVec = 4;
constexpr size_t kMaxVal = 512;
constexpr size_t kNumTests = 64;
constexpr size_t kMaxIter = 8;

// Forward declaration of the kernel name
// (This will become unnecessary in a future compiler version.)
template<int attr_type>
class KernelCompute;

using UintArray = std::array<unsigned, kVec>;
using Uint2DArray = std::array<std::array<unsigned, kVec>, kRows>;
using UintSQArray = std::array<std::array<unsigned, kVec>, kVec>; // square

// The shared compute function for host and device code
size_t Compute(unsigned init, Uint2DArray &dict_offset) {

  // We do not provide any attributes for compare_offset and hash;
  // we let the compiler decide what's best based on the access pattern
  // and their size.
  UintSQArray compare_offset;
  UintArray hash;

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

// Declare a 2D array with memory attribute 'doublepump' if
// attr_type=2, attribute 'singlepump' if attr_type=1,
// and no memory attributes otherwise
template<int attr_type>
Uint2DArray CreateDictOffset() {
  if (attr_type == 1) {

    // The memory attributes apply to the array's declaration
    [[intelfpga::singlepump, intelfpga::memory("MLAB"),
      intelfpga::numbanks(kVec), intelfpga::max_replicates(kVec)]]
    Uint2DArray dict_offset;

    return dict_offset;

  } else if (attr_type == 2) {

    [[intelfpga::doublepump, intelfpga::memory("MLAB"),
      intelfpga::numbanks(kVec), intelfpga::max_replicates(kVec)]]
    Uint2DArray dict_offset;

    return dict_offset;
  }

  return Uint2DArray{};
}

template<int attr_type>
unsigned RunKernel(unsigned init, const unsigned dict_offset_init[]) {
  unsigned result = 0;

#if defined(FPGA_EMULATOR)
  intel::fpga_emulator_selector device_selector;
#else
  intel::fpga_selector device_selector;
#endif

  try {
    queue q(device_selector, dpc_common::exception_handler);

    // Flatten the 2D array to a 1D buffer, because the
    // buffer constructor requires a pointer to input data
    // that is contiguous in memory.
    buffer<unsigned, 1> buffer_d(dict_offset_init,
                                 range<1>(kRows * kVec));
    buffer<unsigned, 1> buffer_r(&result, 1);

    auto e = q.submit([&](handler &h) {
      auto accessor_d = buffer_d.get_access<access::mode::read>(h);
      auto accessor_r = buffer_r.get_access<access::mode::discard_write>(h);

      h.single_task<KernelCompute<attr_type>>(
                    [=]() [[intel::kernel_args_restrict]] {

        // Declare 'dict_offset' to be single or double pumped
        Uint2DArray dict_offset = CreateDictOffset<attr_type>();

        // Initialize 'dict_offset' with values from global memory.
        for (size_t i = 0; i < kRows; ++i) {
          #pragma unroll
          for (size_t k = 0; k < kVec; ++k) {
            // After unrolling, we end up with kVec writes to dict_offset.
            dict_offset[i][k] = accessor_d[i * kVec + k];
          }
        }
        accessor_r[0] = Compute(init, dict_offset);
      });
    });

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
  Uint2DArray dict_offset;
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t k = 0; k < kVec; ++k) {
      dict_offset[i][k] = dict_offset_init[i * kVec + k];
    }
  }
  return Compute(init, dict_offset);
}

int main() {
  srand(0);

  Uint2DArray dict_offset_init;

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
