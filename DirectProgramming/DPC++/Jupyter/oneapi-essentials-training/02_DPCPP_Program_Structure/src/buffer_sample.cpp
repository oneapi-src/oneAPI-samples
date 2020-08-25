//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

constexpr int num=16;
using namespace sycl;

  int main() {
  auto R = range<1>{ num };
  //Create Buffers A and B
  buffer<int> A{ R }, B{ R };
  //Create a device queue
  queue Q;
  //Submit Kernel 1
  Q.submit([&](handler& h) {
    //Accessor for buffer A
    auto out = A.get_access<access::mode::write>(h);
    h.parallel_for(R, [=](id<1> idx) {
      out[idx] = idx[0]; }); });
  //Submit Kernel 2
  Q.submit([&](handler& h) {
    //This task will wait till the first queue is complete
    auto out = A.get_access<access::mode::write>(h);
    h.parallel_for(R, [=](id<1> idx) {
      out[idx] += idx[0]; }); });
  //Submit Kernel 3
  Q.submit([&](handler& h) { 
    //Accessor for Buffer B
    auto out = B.get_access<access::mode::write>(h);
    h.parallel_for(R, [=](id<1> idx) {
      out[idx] = idx[0]; }); });
  //Submit task 4
  Q.submit([&](handler& h) {
   //This task will wait till kernel 2 and 3 are complete
   auto in = A.get_access<access::mode::read>(h);
   auto inout =
    B.get_access<access::mode::read_write>(h);
  h.parallel_for(R, [=](id<1> idx) {
    inout[idx] *= in[idx]; }); }); 
      
 // And the following is back to device code
 auto result =
    B.get_access<access::mode::read>();
  for (int i=0; i<num; ++i)
    std::cout << result[i] << "\n";      
  return 0;
}