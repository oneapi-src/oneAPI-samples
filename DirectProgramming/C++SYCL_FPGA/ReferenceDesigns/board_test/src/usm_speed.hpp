#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// NOTE: sycl::ulong8 was picked for this test because it is 64 bytes in size
// and that is the width of the interconnect to global memory.

// Arbitrary value used for testing/initialization
#define TEST_VAL 5

// Forward declare the kernel names
class USMMemCopy;
class USMMemRead;
class USMMemWrite;

// Available tests to be run
enum USMTest { MEMCOPY, READ, WRITE };

// MEMCOPY test: launches a kernel to copy data from one USM pointer to another.
sycl::event memcopy_kernel(sycl::queue &q, sycl::ulong8 *in, sycl::ulong8 *out,
                           size_t num_items) {
  return q.single_task<USMMemCopy>([=]() [[intel::kernel_args_restrict]] {
    sycl::host_ptr<sycl::ulong8> in_h(in);
    sycl::host_ptr<sycl::ulong8> out_h(out);
    for (size_t i = 0; i < num_items; i++) {
      out_h[i] = in_h[i];
    }
  });
}

// READ test: launches a kernel to read data from a USM pointer, sum it up, and
// store to an output pointer.
sycl::event read_kernel(sycl::queue &q, sycl::ulong8 *in, sycl::ulong8 *out,
                        size_t num_items) {
  return q.single_task<USMMemRead>([=]() {
    sycl::host_ptr<sycl::ulong8> in_h(in);
    sycl::host_ptr<sycl::ulong8> out_h(out);
    sycl::ulong8 sum{0};
    for (size_t i = 0; i < num_items; i++) {
      sum += in_h[i];
    }
    // This prevents the reads from being optimized away
    out_h[0] = sum;
  });
}

// WRITE test: launches a kernel to write data to a USM pointer.
sycl::event write_kernel(sycl::queue &q, sycl::ulong8 *in, sycl::ulong8 *out,
                         size_t num_items) {
  return q.single_task<USMMemWrite>([=]() {
    sycl::host_ptr<sycl::ulong8> out_h(out);
    sycl::ulong8 answer{TEST_VAL};
    for (size_t i = 0; i < num_items; i++) {
      out_h[i] = answer;
    }
  });
}

// Function to check output against expected answer.
bool verify(sycl::ulong8 *actual, sycl::ulong8 *expected, size_t num_items) {
  // Verify all values are equal
  for (int i = 0; i < num_items; i++) {
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

  size_t iterations = 1;
  size_t num_bytes = 1024 * 1024 * 1024;
  size_t num_items = num_bytes / sizeof(sycl::ulong8);
  size_t num_bytes_GB = num_bytes / kGB;

  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "Data size: " << num_bytes / kMB << " MB" << std::endl;
  std::cout << "Data type size: " << sizeof(sycl::ulong8) << " bytes"
            << std::endl;

  // USM host allocation
  sycl::ulong8 *in = sycl::malloc_host<sycl::ulong8>(num_items, q);
  sycl::ulong8 *out = sycl::malloc_host<sycl::ulong8>(num_items, q);
  if (in == nullptr || out == nullptr) {
    std::cerr << "Error: Out of memory, can't allocate " << num_bytes
              << " bytes" << std::endl;
    return 1;
  }

  // Initialize the input with random values and output with zero.
  for (int i = 0; i < num_items; i++) {
    in[i] = rand();
    out[i] = 0;
  }

  std::function<sycl::event(sycl::queue &, sycl::ulong8 *, sycl::ulong8 *,
                            size_t)>
      kernel;
  sycl::ulong8 *expected = new sycl::ulong8[num_items];

  // Test selection: set the test function that will be run, and fill the
  // "expected" array with the expected answer.
  switch (test) {
  case MEMCOPY: {
    std::cout << "Case: Full Duplex" << std::endl;
    kernel = memcopy_kernel;
    // Full duplex transfers twice the amount of data
    num_bytes_GB *= 2;
    // When verifying, the output pointer should match the input pointer.
    for (int i = 0; i < num_items; i++) {
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
    for (int i = 0; i < num_items; i++) {
      read_answer += in[i];
    }
    for (int i = 0; i < num_items; i++) {
      expected[i] = (i == 0) ? read_answer : (sycl::ulong8)(0);
    }
    break;
  }
  case WRITE: {
    std::cout << "Case: From Device to Host" << std::endl;
    kernel = write_kernel;
    // When verifying, each index of the output pointer should contain the
    // known answer that was written.
    for (int i = 0; i < num_items; i++) {
      expected[i] = TEST_VAL;
    }
    break;
  }
  default:
    std::cout << "Error: Failed to launch test" << std::endl;
    sycl::free(in, q);
    sycl::free(out, q);
    delete[] expected;
    return 1;
  }

  // The first iteration is slow due to one time tasks like buffer creation,
  // program creation and device programming, bandwidth measured in subsequent
  // iterations.
  kernel(q, in, out, num_items).wait();
  if (!verify(out, expected, num_items)) {
    std::cerr << "FAILED" << std::endl << std::endl;
    sycl::free(in, q);
    sycl::free(out, q);
    delete[] expected;
    return 1;
  }
  float time = 0;
  for (int i = 0; i < iterations; i++) {
    sycl::event e = kernel(q, in, out, num_items);
    e.wait();
    time += SyclGetQStExecTimeNs(e);
  }

  sycl::free(in, q);
  sycl::free(out, q);
  delete[] expected;

  // Report throughput
  time /= iterations;
  std::cout << "Average Time: " << time / 1000.0 << " ns\t" << std::endl;
  std::cout << "Average Throughput: "
            << (num_bytes_GB / (time / (1000.0 * 1000.0 * 1000.0))) << " GB/s\t"
            << std::endl
            << std::endl;

  return 0;
}