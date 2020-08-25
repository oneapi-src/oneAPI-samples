//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <array>
#include <iostream>

#include <CL/sycl.hpp>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include "tbb/task_group.h"

using namespace cl::sycl;

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;


// exception handler
/*
The exception_list parameter is an iterable list of std::exception_ptr objects.
But those pointers are not always directly readable.
So, we rethrow the pointer, catch it,  and then we have the exception itself.
Note: depending upon the operation there may be several exceptions.
*/
auto exception_handler = [](exception_list exceptionList) {
  for (std::exception_ptr const& e : exceptionList) {
    try {
      std::rethrow_exception(e);
    } catch (exception const& e) {
      std::terminate();  // exit the process immediately.
    }
  }
};

#define VERBOSE

const float alpha = 0.5;  // coeff for triad calculation

const size_t array_size = 16;
std::array<float, array_size> a_array;      // input
std::array<float, array_size> b_array;      // input
std::array<float, array_size> c_array;      // output
std::array<float, array_size> c_array_tbb;  // output

class ExecuteOnGpu {
  const char* message;

 public:
  ExecuteOnGpu(const char* str) : message(str) {}
  void operator()() const {
      std::cout << message << "\n";

    // By including all the SYCL work in a {} block, we ensure
    // all SYCL tasks must complete before exiting the block
    {  // starting SYCL code

      const float coeff = alpha;  // coeff is a local varaible
      range<1> n_items{array_size};
      buffer<cl_float, 1> a_buffer(a_array.data(), n_items);
      buffer<cl_float, 1> b_buffer(b_array.data(), n_items);
      buffer<cl_float, 1> c_buffer(c_array.data(), n_items);

      queue q;
      q.submit([&](handler& h) {
            auto a_accessor = a_buffer.get_access<sycl_read>(h);
            auto b_accessor = b_buffer.get_access<sycl_read>(h);
            auto c_accessor = c_buffer.get_access<sycl_write>(h);

            h.parallel_for(n_items, [=](id<1> index) {
              c_accessor[index] = a_accessor[index] + b_accessor[index] * coeff;
            });  // end of the kernel -- parallel for
          })
     .wait_and_throw();  // end of the commands for the SYCL queue

    }  // end of the scope for SYCL code; wait unti queued work completes
  }    // operator
};

class ExecuteOnCpu {
  const char* message;

 public:
  ExecuteOnCpu(const char* str) : message(str) {}
  void operator()() const {
      std::cout << message << "\n";

    tbb::parallel_for(tbb::blocked_range<int>(0, a_array.size()),
                      [&](tbb::blocked_range<int> r) {
                        for (int index = r.begin(); index < r.end(); ++index) {
                          c_array_tbb[index] =
                              a_array[index] + b_array[index] * alpha;
                        }
                      });
  }  // operator()
};

void PrintArr(const char* text, const std::array<float, array_size>& array) {
    std::cout << text;
  for (const auto& s : array) std::cout << s << ' ';
  std::cout << "\n";
}

int main() {
  // init input arrays
  for (int i = 0; i < array_size; i++) {
    a_array[i] = i;
    b_array[i] = i;
  }

  // start tbb task group
  tbb::task_group tg;
  //  tbb::task_scheduler_init init(2);
  int nth = 4;  // number of threads
  auto mp = tbb::global_control::max_allowed_parallelism;
  tbb::global_control gc(mp, nth);

  tg.run(ExecuteOnGpu("executing on GPU"));  // spawn task and return
  tg.run(ExecuteOnCpu("executing on CPU"));  // spawn task and return

  tg.wait();  // wait for tasks to complete

  // Serial execution
  std::array<float, array_size> c_gold;

  for (size_t i = 0; i < array_size; ++i)
    c_gold[i] = a_array[i] + alpha * b_array[i];

  // Compare golden triad with heterogeneous triad
  if (!std::equal(std::begin(c_array), std::end(c_array), std::begin(c_gold)))
    std::cout << "Heterogenous triad error.\n";
  else
    std::cout << "Heterogenous triad correct.\n";

  // Compare golden triad with TBB triad
  if (!std::equal(std::begin(c_array_tbb), end(c_array_tbb),
                  std::begin(c_gold)))
    std::cout << "TBB triad error.\n";
  else
    std::cout << "TBB triad correct.\n";

#ifdef VERBOSE
  PrintArr("input array a_array: ", a_array);
  PrintArr("input array b_array: ", b_array);
  PrintArr("output array c_array on GPU: ", c_array);
  PrintArr("output array c_array_tbb on CPU: ", c_array_tbb);
#endif

}  // main
