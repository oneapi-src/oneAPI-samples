#include <random>
#include <sycl/sycl.hpp>

class USMMemCopy;
class USMMemRead;
class USMMemWrite;

void memcopy_kernel(sycl::queue &q, sycl::vec<long, 8> *in, sycl::vec<long, 8> *out, const sycl::range<1> numItems) {
  q.single_task<USMMemCopy>([=]() [[intel::kernel_args_restrict]] {
    // sycl::host_ptr<sycl::vec<long, 8>> in_h(in);
    // sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    for (size_t i = 0; i < numItems.get(0); i++) {
      out[i] = in[i];
    }
  });
}

void read_kernel(sycl::queue &q, sycl::vec<long, 8> *in, sycl::vec<long, 8> *out, const sycl::range<1> numItems) {
  q.single_task<USMMemRead>([=]() {
    sycl::vec<long, 8> sum{0};
    // sycl::host_ptr<sycl::vec<long, 8>> in_h(in);
    // sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    for (size_t i = 0; i < numItems.get(0); i++) {
      sum += in[i];
    }
    // This prevents the reads from being optimized away
    out[0] = sum;
  });
}

void write_kernel(sycl::queue &q, sycl::vec<long, 8> *in, sycl::vec<long, 8> *out, const sycl::range<1> numItems) {
  q.single_task<USMMemWrite>([=]() {
    sycl::vec<long, 8> anws{5};
    // sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    for (size_t i = 0; i < numItems.get(0); i++) {
      out[i] = anws;
    }
  });
}

bool verify_memcopy_kernel(sycl::vec<long, 8> *in, sycl::vec<long, 8> *out, const sycl::range<1> numItems) {
  for (auto i = 0; i < numItems.get(0); i++) {
    sycl::vec<long, 8> compare = in[i] == out[i];
    for (auto j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, in[" << i << "][" << j << "]:" << in[i][j] << " != out[" << i << "][" << j << "]:" << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verify_read_kernel(sycl::vec<long, 8> *in, sycl::vec<long, 8> *out, const sycl::range<1> numItems) {
  sycl::vec<long, 8> anwser{0};
  for (auto i = 0; i < numItems.get(0); i++) {
    anwser += in[i];
  }
  for (auto i = 0; i < numItems.get(0); i++) {
    sycl::vec<long, 8> compare{0};
    if (i == 0) {
      compare = anwser == out[i];
    } else {
      compare = compare == out[i];
    }
    for (auto j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        if (i == 0) {
          std::cerr << "ERROR: Values do not match, anwser[" << j << "]:" << anwser[j];
        } else {
          std::cerr << "ERROR: Values do not match, anwser[" << j << "]:" << compare[j];
        }
        std::cerr << " != out[" << i << "][" << j << "]:" << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verify_write_kernel(sycl::vec<long, 8> *in, sycl::vec<long, 8> *out, const sycl::range<1> numItems) {
  sycl::vec<long, 8> anwser{5};
  for (auto i = 0; i < numItems.get(0); i++) {
    sycl::vec<long, 8> compare = anwser == out[i];
    for (auto j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, anwser[" << j << "]:" << anwser[j] << " != out[" << i << "][" << j << "]:" << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

void run_test(sycl::queue &q, const size_t numBytes, int iterations,
              std::function<void(sycl::queue &, sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> kernel,
              std::function<bool(sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> verify,
              std::chrono::microseconds &time) {

  const sycl::range<1> numItems{numBytes / sizeof(sycl::vec<long, 8>)};
  sycl::vec<long, 8> *in  = sycl::malloc_host<sycl::vec<long, 8>>(numItems.get(0), q.get_context());
  sycl::vec<long, 8> *out = sycl::malloc_host<sycl::vec<long, 8>>(numItems.get(0), q.get_context());

  if (in == nullptr || out == nullptr) {
    std::cerr << "Error: Out of memory, can't allocate " << numBytes << " bytes" << std::endl;
    exit(1);
  }

  // initialize the input
  std::random_device rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<long> distrib(0, 1024);
  for (auto i = 0; i < numItems.get(0); i++) {
    in[i] = {distrib(gen)};
    out[i] = {0};
  }

  // The first invocation is slow because the following calls are called when
  // the kernel is submitted to the queue: clCreateBuffer,
  // clCreateProgramWithBinary, clBuildProgram, clEnqueueWriteBuffer(implicit),
  // clEnqueueNDsycl::rangeKernel.
  // Buffer creation, program creation, and device programming does not happen
  // on subsequent calls.
  kernel(q, in, out, numItems);
  q.wait();

  if (!verify(in, out, numItems)) {
    std::cerr << "FAILED" << std::endl;
    exit(1);
  }

  std::array<std::chrono::high_resolution_clock::time_point, 3> t;
  for (auto i = 0; i < iterations; i++) {
    t[0] = std::chrono::high_resolution_clock::now();
    kernel(q, in, out, numItems);
    q.wait();
    t[1] = std::chrono::high_resolution_clock::now();
    time += std::chrono::duration_cast<std::chrono::microseconds>(t[1] - t[0]);
  }

  sycl::free(in, q.get_context());
  sycl::free(out, q.get_context());
}