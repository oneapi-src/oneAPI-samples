#ifndef __RESTRICTED_USM_KERNEL_HPP__
#define __RESTRICTED_USM_KERNEL_HPP__
#pragma once

#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;
using namespace std::chrono;

//
// The structure of the kernels in this design is shown in the diagram below.
// The Producer kernel reads the data from CPU memory (via PCIe), producing it
// for the RestrictedUSM via a pipe. The Worker does the computation on the
// input data and writes it to the ConsumePipe. The consumer reads the data
// from this pipe and writes the output back to the CPU memory (via PCIe).
//
//                                |----------------------------------|
//                                |              FPGA                |
//               |-------------|  |                                  |
//               |             |  | |- --------|   |---------------| |
//  |-------|    |             |--->| Producer |==>|               | |
//  |       |    |             |  | |----------|   |               | |
//  |  CPU  |<-->| Host Memory |  |                | RestrictedUSM | |
//  |       |    |             |  | |----------|   |               | |
//  |-------|    |             |<---| Consumer |<==|               | |
//               |             |  | |----------|   |---------------| |
//               |-------------|  |                                  |
//                                |----------------------------------|
//
//
// As shown in the image above and the code below, we have split this design
// into three kernels:
//    1) Producer
//    2) RestrictedUSM
//    3) Consumer
// We do this to decouple the reads/writes from/to the Host Memory over PCIe.
// Decoupling the memory accesses and using SYCL pipes with a substantial
// depth ('kPipeDepth' below) allows the kernel to be more resilient against
// stalls while reading/writing from/to the Host Memory over PCIe. 
//

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class RestrictedUSM;
class Producer;
class Consumer;

// pipes
constexpr size_t kPipeDepth = 64;
template <typename T>
using ProducePipe = pipe<class ProducePipeClass, T, kPipeDepth>;
template <typename T>
using ConsumePipe = pipe<class ConsumePipeClass, T, kPipeDepth>;

//
// reads the input data from the hosts memory
// and writes it to the ProducePipe
//
template <typename T>
event SubmitProducer(queue& q, T* in_data, size_t size) {
  auto e = q.submit([&](handler& h) {
    h.single_task<Producer>([=]() [[intel::kernel_args_restrict]] {
      // using a host_ptr tells the compiler that this pointer lives in the
      // hosts address space
      host_ptr<T> h_in_data(in_data);

      for (size_t i = 0; i < size; i++) {
        T data_from_host_memory = *(h_in_data + i);
        ProducePipe<T>::write(data_from_host_memory);
      }
    });
  });

  return e;
}

//
// The worker kernel in the device:
//  1) read input data from the ProducePipe
//  2) perform computation
//  3) write the output data to the ConsumePipe
//
template <typename T>
event SubmitWorker(queue& q, size_t size) {
  auto e = q.submit([&](handler& h) {
    h.single_task<RestrictedUSM>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < size; i++) {
        T data = ProducePipe<T>::read();
        T value = data * i; // perform computation
        ConsumePipe<T>::write(value);
      }
    });
  });

  return e;
}

//
// reads output data from the device via ConsumePipe
// and writes it to the hosts memory
//
template <typename T>
event SubmitConsumer(queue& q, T* out_data, size_t size) {
  auto e = q.submit([&](handler& h) {
    // using a host_ptr tells the compiler that this pointer lives in the
    // hosts address space
    host_ptr<T> h_out_data(out_data);

    h.single_task<Consumer>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < size; i++) {
        T data_to_host_memory = ConsumePipe<T>::read();
        *(h_out_data + i) = data_to_host_memory;
      }
    });
  });

  return e;
}

template <typename T>
double RestrictedUSMKernel(queue& q, T* in, T* out, size_t size) {
  // start the timer
  auto start = high_resolution_clock::now();

  // start the kernels
  auto worker_event = SubmitWorker<T>(q, size);
  auto producer_event = SubmitProducer<T>(q, in, size);
  auto consumer_event = SubmitConsumer<T>(q, out, size);

  // wait for all the kernels to finish
  q.wait();

  // stop the timer
  auto end = high_resolution_clock::now();
  duration<double, std::milli> diff = end - start;

  return diff.count();
}

#endif /* __RESTRICTED_USM_KERNEL_HPP__ */
