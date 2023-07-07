class USMMemCopy;
class USMMemRead;
class USMMemWrite;

using long8 = sycl::vec<long, 8>;

enum USMTest { MEMCOPY, READ, WRITE };
constexpr size_t kIterations = 1;
constexpr size_t kNumBytes = 1024 * 1024 * 1024;
constexpr size_t kNumItems = kNumBytes / sizeof(long8);

// Launches a kernel to copy data from one USM pointer to another.
sycl::event memcopy_kernel(sycl::queue &q, long8 *in, long8 *out) {
  return q.single_task<USMMemCopy>([=]() [[intel::kernel_args_restrict]] {
    sycl::host_ptr<long8> in_h(in);
    sycl::host_ptr<long8> out_h(out);
    for (size_t i = 0; i < kNumItems; i++) {
      out_h[i] = in_h[i];
    }
  });
}

bool verify_memcopy(long8 *in, long8 *out) {
  for (int i = 0; i < kNumItems; i++) {
    // "compare" is vector containing an element-wise "==" of in[i] and out[i]
    long8 compare = in[i] == out[i];
    for (int j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, in[" << i << "][" << j
                  << "]:" << in[i][j] << " != out[" << i << "][" << j
                  << "]:" << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

// Launches a kernel to read data from a USM pointer, sum it up, and store to an
// output pointer.
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

bool verify_read(long8 *in, long8 *out) {
  // The read kernel calculates a sum of all the values at "in" and stores it
  // at out[0]. First calculate a reference to compare to.
  long8 answer{0};
  for (int i = 0; i < kNumItems; i++) {
    answer += in[i];
  }
  // The rest of the indices should retain their intial value, 0. Define a
  // zero vector to compare to.
  long8 zero_vec{0};

  for (int i = 0; i < kNumItems; i++) {
    // "compare" will be a vector containing an element-wise "==" of the
    // vectors we are checking equality for
    long8 compare{0};
    if (i == 0) {
      // out[0] should contain the answer (sum of all values at "in")
      compare = out[i] == answer;
    } else {
      // all other elements should be zero
      compare = out[i] == zero_vec;
    }
    for (int j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        if (i == 0) {
          std::cerr << "ERROR: Values do not match, answer[" << j
                    << "]:" << answer[j];
        } else {
          std::cerr << "ERROR: Values do not match, 0";
        }
        std::cerr << " != out[" << i << "][" << j << "]:" << out[i][j]
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

// Launches a kernel to write data to a USM pointer.
sycl::event write_kernel(sycl::queue &q, long8 *in, long8 *out) {
  return q.single_task<USMMemWrite>([=]() {
    sycl::host_ptr<long8> out_h(out);
    long8 answer{5};
    for (size_t i = 0; i < kNumItems; i++) {
      out_h[i] = answer;
    }
  });
}

bool verify_write(long8 *in, long8 *out) {
  // The write kernel writes a known value to every index of "out".
  long8 answer{5};
  for (int i = 0; i < kNumItems; i++) {
    // "compare" is vector containing an element-wise "==" of out[i] and
    // answer
    long8 compare = out[i] == answer;
    for (int j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, answer[" << j
                  << "]:" << answer[j] << " != out[" << i << "][" << j
                  << "]:" << out[i][j] << std::endl;
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

  std::cout << "Iterations: " << kIterations << std::endl;
  std::cout << "Data size: " << kNumBytes / kMB << " MB" << std::endl;
  std::cout << "Data type size: " << sizeof(long8) << " bytes" << std::endl;

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
    in[i] = {rand()};
    out[i] = {0};
  }

  // Test selection
  std::function<sycl::event(sycl::queue &, long8 *, long8 *)> kernel;
  std::function<bool(long8 *, long8 *)> verify;
  switch (test) {
  case MEMCOPY:
    std::cout << std::endl << "Case: Full Duplex" << std::endl;
    kernel = memcopy_kernel;
    verify = verify_memcopy;
    // full duplex transfers twice the amount of data
    kNumBytes_gb = kNumBytes * 2 / kGB;
    break;
  case READ:
    std::cout << std::endl << "Case: From Host to Device" << std::endl;
    kernel = read_kernel;
    verify = verify_read;
    kNumBytes_gb = kNumBytes / kGB;
    break;
  case WRITE:
    std::cout << std::endl << "Case: From Device to Host" << std::endl;
    kernel = write_kernel;
    verify = verify_write;
    kNumBytes_gb = kNumBytes / kGB;
    break;
  default:
    std::cout << "Error: Failed to launch test" << std::endl;
    return 1;
  }

  // The first iteration is slow due to one time tasks like buffer creation,
  // program creation and device programming, bandwidth measured in subsequent
  // iterations.
  kernel(q, in, out).wait();
  if (!verify(in, out)) {
    std::cerr << "FAILED" << std::endl;
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
            << std::endl;

  return 0;
}