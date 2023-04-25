//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
constexpr int MAXTHREADS = 16;
constexpr int NUM = 1024;
constexpr int MATRIXTILESIZE = 16;
constexpr int WPT = 8;

#include <CL/sycl.hpp>
// exception handler
/*
The exception_list parameter is an iterable list of std::exception_ptr objects.
But those pointers are not always directly readable.
So, we rethrow the pointer, catch it,  and then we have the exception itself.
Note: depending upon the operation there may be several exceptions.
*/
auto exception_handler = [](sycl::exception_list exceptionList) {
  for (std::exception_ptr const &e : exceptionList) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::terminate(); // exit the process immediately.
    }
  }
};

typedef float TYPE;
typedef TYPE Array[NUM];

class TimeInterval {
public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

// Select which multiply kernel to use via the following macro so that the
// kernel being used can be reported when the test is run.
#define MULTIPLY multiply1_2

extern void multiply1(int msize, int tidx, int numt, TYPE a[][NUM],
                      TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_1(int msize, int tidx, int numt, TYPE a[][NUM],
                        TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_2(int msize, int tidx, int numt, TYPE a[][NUM],
                        TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);

extern void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM],
                             TYPE c[][NUM], TYPE t[][NUM]);
