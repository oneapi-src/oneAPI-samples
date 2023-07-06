#include <sycl/sycl.hpp>

class USMMemCopy;
class USMMemRead;
class USMMemWrite;

// Launches a kernel to copy data from one USM pointer to another.
sycl::event memcopy_kernel(sycl::queue &q,           // device queue
                           sycl::vec<long, 8> *in,   // input pointer
                           sycl::vec<long, 8> *out,  // output pointer
                           int num_items             // num items to copy
) {
  return q.single_task<USMMemCopy>([=]() [[intel::kernel_args_restrict]] {
    sycl::host_ptr<sycl::vec<long, 8>> in_h(in);
    sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    for (size_t i = 0; i < num_items; i++) {
      out_h[i] = in_h[i];
    }
  });
}

bool verify_memcopy(sycl::vec<long, 8> *in,   // input pointer
                    sycl::vec<long, 8> *out,  // output pointer
                    int num_items             // num items to verify
) {
  for (int i = 0; i < num_items; i++) {
    // "compare" is vector containing an element-wise "==" of in[i] and out[i]
    sycl::vec<long, 8> compare = in[i] == out[i];
    for (int j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, in[" << i << "][" << j
                  << "]:" << in[i][j] << " != out[" << i << "][" << j << "]:"
                  << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

// Launches a kernel to read data from a USM pointer, sum it up, and store to an
// output pointer.
sycl::event read_kernel(sycl::queue &q,           // device queue
                        sycl::vec<long, 8> *in,   // input pointer
                        sycl::vec<long, 8> *out,  // output pointer
                        int num_items             // num items to copy
) {
  return q.single_task<USMMemRead>([=]() {
    sycl::host_ptr<sycl::vec<long, 8>> in_h(in);
    sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    sycl::vec<long, 8> sum{0};
    for (size_t i = 0; i < num_items; i++) {
      sum += in_h[i];
    }
    // This prevents the reads from being optimized away
    out_h[0] = sum;
  });
}

bool verify_read(sycl::vec<long, 8> *in,   // input pointer
                 sycl::vec<long, 8> *out,  // output pointer
                 int num_items             // num items to verify
) {
  // The read kernel calculates a sum of all the values at "in" and stores it
  // at out[0]. First calculate a reference to compare to.
  sycl::vec<long, 8> answer{0};
  for (int i = 0; i < num_items; i++) {
    answer += in[i];
  }
  // The rest of the indices should retain their intial value, 0. Define a
  // zero vector to compare to.
  sycl::vec<long, 8> zero_vec{0};

  for (int i = 0; i < num_items; i++) {
    // "compare" will be a vector containing an element-wise "==" of the
    // vectors we are checking equality for
    sycl::vec<long, 8> compare{0};
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
          std::cerr << "ERROR: Values do not match, answer[" << j << "]:"
                    << answer[j];
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
sycl::event write_kernel(sycl::queue &q,           // device queue
                         sycl::vec<long, 8> *in,   // input pointer (unused)
                         sycl::vec<long, 8> *out,  // output pointer
                         int num_items             // num items to copy
) {
  return q.single_task<USMMemWrite>([=]() {
    sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    sycl::vec<long, 8> answer{5};
    for (size_t i = 0; i < num_items; i++) {
      out_h[i] = answer;
    }
  });
}

bool verify_write(sycl::vec<long, 8> *in,   // input pointer (unused)
                  sycl::vec<long, 8> *out,  // output pointer
                  int num_items             // num items to verify
) {
  // The write kernel writes a known value to every index of "out".
  sycl::vec<long, 8> answer{5};
  for (int i = 0; i < num_items; i++) {
    // "compare" is vector containing an element-wise "==" of out[i] and
    // answer
    sycl::vec<long, 8> compare = out[i] == answer;
    for (int j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, answer[" << j << "]:"
                  << answer[j]
                  << " != out[" << i << "][" << j << "]:"
                  << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

// Parameterized function to perform one of the tests defined above. Allocates
// and initializes host USM, runs the test, then verifies the answer. Then
// re-runs the test several times to measure bandwidth.
int run_test(sycl::queue &q,         // device queue
             const size_t num_bytes, // number of bytes of memory to allocate
             int iterations,         // number of times to repeat the test
             std::function<sycl::event(sycl::queue&, sycl::vec<long, 8>*,
                                       sycl::vec<long, 8>*, int)> kernel,
             std::function<bool(sycl::vec<long, 8> *, sycl::vec<long, 8> *,
                                int)> verify,
             float &time            // variable to store time take by operation
) {
  // USM host allocation
  int num_items = num_bytes / sizeof(sycl::vec<long, 8>);
  sycl::vec<long, 8> *in  = sycl::malloc_host<sycl::vec<long, 8>>(num_items, q);
  sycl::vec<long, 8> *out = sycl::malloc_host<sycl::vec<long, 8>>(num_items, q);
  if (in == nullptr || out == nullptr) {
    std::cerr << "Error: Out of memory, can't allocate " << num_bytes
              << " bytes" << std::endl;
    return 1;
  }

  // Initialize the input with random values and output with zero
  for (int i = 0; i < num_items; i++) {
    in[i] = {rand()};
    out[i] = {0};
  }

  // The first iteration is slow due to one time tasks like buffer creation,
  // program creation and device programming, bandwidth measured in subsequent
  // iterations.
  kernel(q, in, out, num_items);
  q.wait();

  if (!verify(in, out, num_items)) {
    std::cerr << "FAILED" << std::endl;
    return 1;
  }

  for (int i = 0; i < iterations; i++) {
    sycl::event evt = kernel(q, in, out, num_items);
    q.wait();
    time += SyclGetQStExecTimeNs(evt);
  }

  // Free USM
  sycl::free(in,  q);
  sycl::free(out, q);

  return 0;
}