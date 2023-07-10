class USMMemCopy;
class USMMemRead;
class USMMemWrite;

using long8 = sycl::vec<long, 8>;

enum USMTest { MEMCOPY, READ, WRITE };
constexpr size_t kIterations = 1;
constexpr size_t kNumBytes = 1024 * 1024 * 1024;
constexpr size_t kNumItems = kNumBytes / sizeof(long8);

// MEMCOPY test: launches a kernel to copy data from one USM pointer to another.
sycl::event memcopy_kernel(sycl::queue &q, long8 *in, long8 *out) {
  return q.single_task<USMMemCopy>([=]() [[intel::kernel_args_restrict]] {
    sycl::host_ptr<long8> in_h(in);
    sycl::host_ptr<long8> out_h(out);
    for (size_t i = 0; i < kNumItems; i++) {
      out_h[i] = in_h[i];
    }
  });
}

// READ test: launches a kernel to read data from a USM pointer, sum it up, and
// store to an output pointer.
sycl::event read_kernel(sycl::queue &q, long8 *in, long8 *out) {
  return q.single_task<USMMemRead>([=]() {
    sycl::host_ptr<long8> in_h(in);
    sycl::host_ptr<long8> out_h(out);
    long8 sum{0};
    for (size_t i = 0; i < kNumItems; i++) {
      sum += in_h[i];
    }
    // This prevents the reads from being optimized away
    out_h[0] = sum;
  });
}

// WRITE test: launches a kernel to write data to a USM pointer.
sycl::event write_kernel(sycl::queue &q, long8 *in, long8 *out) {
  return q.single_task<USMMemWrite>([=]() {
    sycl::host_ptr<long8> out_h(out);
    long8 answer{5};
    for (size_t i = 0; i < kNumItems; i++) {
      out_h[i] = answer;
    }
  });
}

// Function to check the answer of the memcopy, read or write test.
bool verify(long8 *in, long8 *out, USMTest test) {

  // The write kernel writes a known value to every index of "out".
  long8 write_answer{5};

  // The read kernel calculates a sum of all the values at "in" and stores it
  // at out[0]. First calculate a reference to compare to. The rest of the
  // indices should retain their intial value, 0. So, also define a zero vector
  // to compare to.
  long8 zero_vec{0};
  long8 read_answer{0};
  for (int i = 0; i < kNumItems; i++) {
    read_answer += in[i];
  }

  for (int i = 0; i < kNumItems; i++) {
    // for each element of the output pointer, the value we expect it to take.
    long8 reference;

    switch (test) {
    case MEMCOPY:
      // output pointer should match the input pointer.
      reference = in[i];
      break;
    case READ:
      // first index should contain the expected answer (sum of all values at
      // the input pointer); at all other it should contain zero.
      reference = (i == 0) ? read_answer : zero_vec;
      break;
    case WRITE:
      // each index should contain the known answer that was written.
      reference = write_answer;
      break;
    }

    // "compare" will be a vector containing an element-wise "==" of the
    // vectors we are checking equality for.
    long8 compare = (out[i] == reference);
    for (int j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, "
                  << "out[" << i << "][" << j << "] = " << out[i][j]
                  << "; expected " << reference[j] << std::endl;
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

  double kNumBytes_gb;

  // USM host allocation
  long8 *in = sycl::malloc_host<long8>(kNumItems, q);
  long8 *out = sycl::malloc_host<long8>(kNumItems, q);
  if (in == nullptr || out == nullptr) {
    std::cerr << "Error: Out of memory, can't allocate " << kNumBytes
              << " bytes" << std::endl;
    return 1;
  }

  // Initialize the input with random values and output with zero
  for (int i = 0; i < kNumItems; i++) {
    in[i] = 1; //{rand() % 2};
    out[i] = {0};
  }

  // Test selection
  std::function<sycl::event(sycl::queue &, long8 *, long8 *)> kernel;
  switch (test) {
  case MEMCOPY:
    std::cout << "Case: Full Duplex" << std::endl;
    kernel = memcopy_kernel;
    // full duplex transfers twice the amount of data
    kNumBytes_gb = kNumBytes * 2 / kGB;
    break;
  case READ:
    std::cout << "Case: From Host to Device" << std::endl;
    kernel = read_kernel;
    kNumBytes_gb = kNumBytes / kGB;
    break;
  case WRITE:
    std::cout << "Case: From Device to Host" << std::endl;
    kernel = write_kernel;
    kNumBytes_gb = kNumBytes / kGB;
    break;
  default:
    std::cout << "Error: Failed to launch test" << std::endl;
    return 1;
  }

  std::cout << "Iterations: " << kIterations << std::endl;
  std::cout << "Data size: " << kNumBytes / kMB << " MB" << std::endl;
  std::cout << "Data type size: " << sizeof(long8) << " bytes" << std::endl;

  // The first iteration is slow due to one time tasks like buffer creation,
  // program creation and device programming, bandwidth measured in subsequent
  // iterations.
  kernel(q, in, out).wait();
  if (!verify(in, out, test)) {
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

  // Report throughput
  time /= kIterations;
  std::cout << "Average Time: " << time / 1000.0 << " ns\t" << std::endl;
  std::cout << "Average Throughput: "
            << (kNumBytes_gb / (time / (1000.0 * 1000.0 * 1000.0))) << " GB/s\t"
            << std::endl << std::endl;

  return 0;
}