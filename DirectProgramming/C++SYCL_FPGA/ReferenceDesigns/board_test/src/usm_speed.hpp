// Forward declare the kernel names
class USMMemCopy;
class USMMemRead;
class USMMemWrite;

// NOTE: sycl::ulong8 was picked for this test because it is 64 bytes in size
// and that is the width of the interconnect to global memory.

enum USMTest { MEMCOPY, READ, WRITE };

constexpr size_t kIterations = 1;
constexpr size_t kNumBytes = 1024 * 1024 * 1024;
constexpr size_t kNumItems = kNumBytes / sizeof(sycl::ulong8);

// Arbitrary value used for testing/initialization
constexpr long kTestValue = 5;

// MEMCOPY test: launches a kernel to copy data from one USM pointer to another.
sycl::event memcopy_kernel(sycl::queue &q, sycl::ulong8 *in,
                           sycl::ulong8 *out) {
  return q.single_task<USMMemCopy>([=]() [[intel::kernel_args_restrict]] {
    sycl::host_ptr<sycl::ulong8> in_h(in);
    sycl::host_ptr<sycl::ulong8> out_h(out);
    for (size_t i = 0; i < kNumItems; i++) {
      out_h[i] = in_h[i];
    }
  });
}

// READ test: launches a kernel to read data from a USM pointer, sum it up, and
// store to an output pointer.
sycl::event read_kernel(sycl::queue &q, sycl::ulong8 *in, sycl::ulong8 *out) {
  return q.single_task<USMMemRead>([=]() {
    sycl::host_ptr<sycl::ulong8> in_h(in);
    sycl::host_ptr<sycl::ulong8> out_h(out);
    sycl::ulong8 sum{0};
    for (size_t i = 0; i < kNumItems; i++) {
      sum += in_h[i];
    }
    // This prevents the reads from being optimized away
    out_h[0] = sum;
  });
}

// WRITE test: launches a kernel to write data to a USM pointer.
sycl::event write_kernel(sycl::queue &q, sycl::ulong8 *in, sycl::ulong8 *out) {
  return q.single_task<USMMemWrite>([=]() {
    sycl::host_ptr<sycl::ulong8> out_h(out);
    sycl::ulong8 answer{kTestValue};
    for (size_t i = 0; i < kNumItems; i++) {
      out_h[i] = answer;
    }
  });
}

// Function to check the answer of the memcopy, read or write test.
bool verify(sycl::ulong8 *actual, sycl::ulong8 *expected) {
  // Verify all values are equal
  for (int i = 0; i < kNumItems; i++) {
    for (int j = 0; j < 8; j++) {
      if (actual[i][j] != expected[i][j]) {
        std::cerr << "ERROR: Values do not match, "
                  << "out[" << i << "][" << j << "] = " << actual[i][j]
                  << "; expected " << expected[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

// Parameterized function to perform one of the tests defined above. Allocates
// and initializes host USM, runs the test, then verifies the answer. Then
// re-runs the test several times to measure bandwidth.
int run_test(sycl::queue &q, USMTest test) {

  double kNumBytes_gb = kNumBytes / kGB;

  // USM host allocation
  sycl::ulong8 *in = sycl::malloc_host<sycl::ulong8>(kNumItems, q);
  sycl::ulong8 *out = sycl::malloc_host<sycl::ulong8>(kNumItems, q);
  if (in == nullptr || out == nullptr) {
    std::cerr << "Error: Out of memory, can't allocate " << kNumBytes
              << " bytes" << std::endl;
    return 1;
  }

  // Initialize the input with random values and output with zero.
  for (int i = 0; i < kNumItems; i++) {
    in[i] = rand();
    out[i] = 0;
  }

  std::function<sycl::event(sycl::queue &, sycl::ulong8 *, sycl::ulong8 *)> kernel;
  sycl::ulong8 *expected = new sycl::ulong8[kNumItems];

  // Test selection: set the function that will be run, and fill the "expected"
  // array with the expected answer.
  switch (test) {
  case MEMCOPY: {
    std::cout << "Case: Full Duplex" << std::endl;
    kernel = memcopy_kernel;
    // Full duplex transfers twice the amount of data
    kNumBytes_gb *= 2;
    // When verifying, the output pointer should match the input pointer.
    for (int i = 0; i < kNumItems; i++) {
      expected[i] = in[i];
    }
    break;
  }
  case READ: {
    std::cout << "Case: From Host to Device" << std::endl;
    kernel = read_kernel;
    // When verifying, the first index of the output pointer should contain
    // the expected answer (sum of all values at the input pointer); at all
    // other indices it should contain zero.
    sycl::ulong8 read_answer{0};
    for (int i = 0; i < kNumItems; i++) {
      read_answer += in[i];
    }
    for (int i = 0; i < kNumItems; i++) {
      expected[i] = (i == 0) ? read_answer : (sycl::ulong8)(0);
    }
    break;
  }
  case WRITE: {
    std::cout << "Case: From Device to Host" << std::endl;
    kernel = write_kernel;
    // When verifying, each index of the output pointer should contain the
    // known answer that was written.
    for (int i = 0; i < kNumItems; i++) {
      expected[i] = kTestValue;
    }
    break;
  }
  default:
    std::cout << "Error: Failed to launch test" << std::endl;
    return 1;
  }

  std::cout << "Iterations: " << kIterations << std::endl;
  std::cout << "Data size: " << kNumBytes / kMB << " MB" << std::endl;
  std::cout << "Data type size: " << sizeof(sycl::ulong8) << " bytes"
            << std::endl;

  // The first iteration is slow due to one time tasks like buffer creation,
  // program creation and device programming, bandwidth measured in subsequent
  // iterations.
  kernel(q, in, out).wait();
  if (!verify(out, expected)) {
    std::cerr << "FAILED" << std::endl << std::endl;
    return 1;
  }
  float time = 0;
  for (int i = 0; i < kIterations; i++) {
    sycl::event e = kernel(q, in, out);
    e.wait();
    time += SyclGetQStExecTimeNs(e);
  }

  // Free USM
  sycl::free(in, q);
  sycl::free(out, q);

  delete[] expected;

  // Report throughput
  time /= kIterations;
  std::cout << "Average Time: " << time / 1000.0 << " ns\t" << std::endl;
  std::cout << "Average Throughput: "
            << (kNumBytes_gb / (time / (1000.0 * 1000.0 * 1000.0))) << " GB/s\t"
            << std::endl << std::endl;

  return 0;
}