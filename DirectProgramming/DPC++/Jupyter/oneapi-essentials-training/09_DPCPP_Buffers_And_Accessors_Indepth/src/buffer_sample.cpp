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
    accessor out(A,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      out[idx] = idx[0]; }); });
  //Submit Kernel 2
  Q.submit([&](handler& h) {
    //This task will wait till the first queue is complete
    accessor out(A,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      out[idx] += idx[0]; }); });
  //Submit Kernel 3
  Q.submit([&](handler& h) { 
    //Accessor for Buffer B
    accessor out(B,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      out[idx] = idx[0]; }); });
  //Submit task 4
  Q.submit([&](handler& h) {
   //This task will wait till kernel 2 and 3 are complete
   accessor in (A,h,read_only);
   accessor inout(B,h);
  h.parallel_for(R, [=](auto idx) {
    inout[idx] *= in[idx]; }); }); 
      
 // And the following is back to device code
 host_accessor result(B,read_only);
  for (int i=0; i<num; ++i)
    std::cout << result[i] << "\n";      
  return 0;
}
