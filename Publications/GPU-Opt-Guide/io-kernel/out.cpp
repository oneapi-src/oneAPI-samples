//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

void out1() {
  constexpr int N = 16;
  sycl::queue q;
  q.submit([&](auto &cgh) {
     sycl::stream str(8192, 1024, cgh);
     cgh.parallel_for(N, [=](sycl::item<1> it) {
       int id = it[0];
       /* Send the identifier to a stream to be printed on the console */
       str << "ID=" << id << sycl::endl;
     });
   }).wait();
} // end out1

void out2() {
  sycl::queue q;
  q.submit([&](auto &cgh) {
     sycl::stream str(8192, 4, cgh);
     cgh.parallel_for(1, [=](sycl::item<1>) {
       str << "ABC" << sycl::endl;     // Print statement 1
       str << "ABCDEFG" << sycl::endl; // Print statement 2
     });
   }).wait();
} // end out2

void out3() {
  sycl::queue q;
  q.submit([&](auto &cgh) {
     sycl::stream str(8192, 10, cgh);
     cgh.parallel_for(1, [=](sycl::item<1>) {
       str << "ABC" << sycl::endl;     // Print statement 1
       str << "ABCDEFG" << sycl::endl; // Print statement 2
     });
   }).wait();
} // end out3

void out4() {
  sycl::queue q;
  q.submit([&](auto &cgh) {
     sycl::stream str(8192, 1024, cgh);
     cgh.parallel_for(sycl::nd_range<1>(32, 4), [=](sycl::nd_item<1> it) {
       int id = it.get_global_id();
       str << "ID=" << id << sycl::endl;
     });
   }).wait();
} // end out4

void out5() {
  int *m = NULL;
  sycl::queue q;
  q.submit([&](auto &cgh) {
     sycl::stream str(8192, 1024, cgh);
     cgh.parallel_for(sycl::nd_range<1>(32, 4), [=](sycl::nd_item<1> it) {
       int id = it.get_global_id();
       str << "ID=" << id << sycl::endl;
       if (id == 31)
         *m = id;
     });
   }).wait();
} // end out5

int main() {
  out1();
  out2();
  out3();
  out4();
  out5();
  return 0;
}
