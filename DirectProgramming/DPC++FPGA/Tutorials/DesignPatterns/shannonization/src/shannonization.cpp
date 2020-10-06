//==============================================================
//Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

#include "IntersectionKernel.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

//
// print the usage
//
void Usage() {
  std::cout
      << "USAGE: ./intersection [--A=<size of list A>] [--B=<size of list B>]"
      << "[--iters=<number of times to run the kernel>] [-h --help]\n";
}

//
// helper to check if string 'str' starts with 'prefix'
//
bool StrStartsWith(std::string& str, std::string prefix) {
  return str.find(prefix) == 0;
}

//
// Helper to count instances of an element 'x' in a sorted vector 'v'.
// Since the vector is sorted, this algorithm has O(logn) complexity,
// rather than the naive O(n) complexity.
//
unsigned int CountSorted(std::vector<unsigned int>& v, int x) {
  // find first occurrence of 'x' in 'v'
  auto low = std::lower_bound(v.begin(), v.end(), x);

  // check if element was present
  if (low == v.end() || *low != x) {
    return 0;
  }

  // find last occurrence of x in array
  auto high = std::upper_bound(low, v.end(), x);

  // return count
  return high - low;
}

//
// Submit the three kernels that make up the whole design
//
template <int Version, int II>
event SubmitKernels(queue& q, std::vector<unsigned int>& a,
                    std::vector<unsigned int>& b, int& n) {
  // static asserts
  static_assert(Version >= 0 && Version <= 3, "Invalid kernel version");
  static_assert(II > 0, "II target must be positive and non-zero");

  // the pipes for this Version of the design
  using ProduceAPipe = pipe<ProduceAPipeClass<Version>, unsigned int>;
  using ProduceBPipe = pipe<ProduceBPipeClass<Version>, unsigned int>;

  // input sizes
  const int a_size = a.size();
  const int b_size = b.size();

  // setup the input buffers
  buffer a_buf(a);
  buffer b_buf(b);

  // setup the output buffer
  buffer<int,1> n_buf(&n, 1);

  // submit the kernel that produces table A
  q.submit([&](handler& h) {
    accessor a_accessor { a_buf, h, read_only };
    h.single_task<ProducerA<Version>>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < a_size; i++) {
        ProduceAPipe::write(a_accessor[i]);
      }
    });
  });

  // submit the kernel that produces table B
  q.submit([&](handler& h) {
    accessor b_accessor { b_buf, h, read_only };
    h.single_task<ProducerB<Version>>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < b_size; i++) {
        ProduceBPipe::write(b_accessor[i]);
      }
    });
  });

  // submit the kernel that performs the intersection
  event e = q.submit([&](handler& h) {
    // output accessor
    accessor n_accessor { n_buf, h, write_only, noinit };

    h.single_task<Worker<Version>>([=]() [[intel::kernel_args_restrict]] {
      // The 'Version' template parameter will choose between the different 
      // versions of the kernel defined in IntersectionKernel.hpp.
      // The operator() of the IntersectionKernel object will return the
      // size of the intersection of A and B.
      IntersectionKernel<Version, II, ProduceAPipe, ProduceBPipe> K;
      n_accessor[0] = K(a_size, b_size);
    });
  });

  return e;
}

//
// This function performs the intersection by submitting the 
// different kernels. This method also validates the output
// of the kernels and prints performance information
//
template <int Version, int II>
bool Intersection(queue& q, std::vector<unsigned int>& a, 
                  std::vector<unsigned int>& b, int golden_n) {
  // For emulation, just do a single iteration.
  // For hardware, perform multiple iterations for a more
  // accurate throughput measurement
#if defined(FPGA_EMULATOR)
  int iterations = 1;
#else
  int iterations = 5;
#endif

  std::cout << "Running " << iterations 
            << ((iterations == 1) ? " iteration" : " iterations")
            << " of kernel " << Version
            << " with |A|=" << a.size() 
            << " and |B|=" << b.size() << "\n";
  
  bool success = true;
  std::vector<double> kernel_latency(iterations);

  // perform multiple iterations of the kernel to get a more accurate
  // throughput measurement
  for (size_t i = 0; i < iterations && success; i++) {
    // run kernel
    int n = 0;
    event e = SubmitKernels<Version,II>(q, a, b, n);

    // check output
    if (golden_n != n) {
      success = false;
      std::cerr << "ERROR: Kernel version " << Version << " output is incorrect"
                << " (Expected=" << golden_n << ", Result=" << n << ")\n";
    }

    // get profiling info
    auto start = e.get_profiling_info<info::event_profiling::command_start>();
    auto end = e.get_profiling_info<info::event_profiling::command_end>();
    kernel_latency[i] = (end - start) / 1e9;
  }

  // If all the iterations were successful, print the throughput results.
  // The FPGA emulator does not accurately represent the hardware performance
  // so we don't print performance results when running with the emulator
  if (success) {
#ifndef FPGA_EMULATOR
    // Compute the average throughput across all iterations.
    // We use the first iteration as a 'warmup' for the FPGA,
    // so we ignore its results.
    double avg_kernel_latency =
        std::accumulate(kernel_latency.begin() + 1, kernel_latency.end(), 0.0) /
        (double)(iterations - 1);

    double input_size_megabytes = 
        ((a.size() + b.size()) * sizeof(unsigned int)) / (1024.0 * 1024.0);

    const double avg_throughput = input_size_megabytes / avg_kernel_latency;

    std::cout << "Kernel " << Version 
              << " average throughput: " << avg_throughput << " MB/s\n";
#endif
  }

  return success;
}


int main(int argc, char** argv) {
  // parse the command line arguments
#if defined(FPGA_EMULATOR)
  unsigned int a_size = 128;
  unsigned int b_size = 256;
#else
  unsigned int a_size = 131072;
  unsigned int b_size = 262144;
#endif
  bool need_help = false;

  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      need_help = true;
    } else {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (StrStartsWith(arg, "--A=")) {
        a_size = std::stoi(str_after_equals);
      } else if (StrStartsWith(arg, "--B=")) {
        b_size = std::stoi(str_after_equals);
      } else {
        std::cout << "WARNING: ignoring unknown argument '" << arg << "'\n";
      }
    }
  }

  // ensure the arrays have more than 3 elements
  if (a_size <= 3) {
    std::cout << "WARNING: array A must have more than 3 "
                  "elements, increasing its size\n";
    a_size = 4;
  }
  if (b_size <= 3) {
    std::cout << "WARNING: array A must have more than 3"
                  "elements, increasing its size\n";
    b_size = 4;
  }

  // print help if needed or asked
  if (need_help) {
    Usage();
    return 0;
  }

  std::cout << "Generating input data\n";

  // seed the random number generator
  srand(777);

  // initialize input data
  std::vector<unsigned int> a(a_size), b(b_size);
  std::iota(a.begin(), a.end(), 0);
  std::generate(b.begin(), b.end(), [=] { return rand() % a_size; });
  std::sort(b.begin(), b.end());

  std::cout << "Computing golden result\n";

  // compute the golden result
  int golden_n = 0;
  for (int i = 0; i < a_size; i++) {
    golden_n += CountSorted(b, a[i]);
  }

  try {
    // queue properties to enable profiling
    auto props = property_list{property::queue::enable_profiling()};

    // the device selector
#ifdef FPGA_EMULATOR
    INTEL::fpga_emulator_selector device_selector;
#else
    INTEL::fpga_selector device_selector;
#endif

    // create the device queue
    queue q(device_selector, props);

    bool success = true;

  // Instantiate multiple versions of the kernel
  // The II achieved by the compiler can differ between FPGA architectures
  //
  // On Arria 10, we are able to achieve an II of 1 for versions 1 and 2 of
  // the kernel (not version 0).
  // Version 2 of the kernel can achieve the highest Fmax with 
  // an II of 1 (and therefore has the highest throughput).
  // Since this tutorial compiles to a single FPGA image, this is not
  // reflected in the final design (that is, version 1 bottlenecks the Fmax
  // of the entire design, which contains versions 0, 1 and 2).
  // However, the difference between versions 1 and 2
  // can be seen in the "Block Scheduled Fmax" columns in the 
  // "Loop Analysis" tab of the HTML reports.
  //
  // On Stratix 10, the same discussion applies, but version 0
  // can only achieve an II of 3 while versions 1 and 2 can only achieve
  // an II of 2. On Stratix 10, we can achieve an II of 1 if we use non-blocking
  // pipe reads in the IntersectionKernel, which is shown in version 3 of the
  // kernel.
  //
#if defined(A10)
    success &= Intersection<0,2>(q, a, b, golden_n);
    success &= Intersection<1,1>(q, a, b, golden_n);
    success &= Intersection<2,1>(q, a, b, golden_n);
    success &= Intersection<3,1>(q, a, b, golden_n);
#elif defined(S10)
    success &= Intersection<0,3>(q, a, b, golden_n);
    success &= Intersection<1,2>(q, a, b, golden_n);
    success &= Intersection<2,2>(q, a, b, golden_n);
    success &= Intersection<3,1>(q, a, b, golden_n);
#else
      static_assert(false, "Unknown FPGA architecture!");
#endif

    if (success) {
      std::cout << "PASSED\n";
    } else {
      std::cout << "FAILED\n";
    }

  } catch (exception const& e) {
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

  return 0;
}
