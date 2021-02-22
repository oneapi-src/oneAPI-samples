//
// This file contains all of the FPGA device code for the multi-kernel design
//

#ifndef __MULTI_KERNEL_HPP__
#define __MULTI_KERNEL_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

// Forward declare the kernel names to reduce name mangling
class K0;
class K1;
class K2;
class P;
class C;

//
// A generic kernel to produce data from host memory to a SYCL pipe.
//
// The following is a block diagram of this kernel:
//
// |-----------------|         |--------------------------|
// | CPU    |-----|  | in_ptr  | FPGA  |---| InPipe       |
// |        | RAM |--|---------|------>| P |========> ... |
// |        |-----|  |         |       |---|              |
// |-----------------|         |--------------------------|
//
template<typename T, typename InPipe>
event SubmitProducer(queue &q, T* in_ptr, size_t size) {
  auto e = q.submit([&](handler& h) {
    h.single_task<P>([=]() [[intel::kernel_args_restrict]] {
      host_ptr<T> in(in_ptr);
      for (size_t i = 0; i < size; i++) {
        auto data = in[i];
        InPipe::write(data);
      }
    });
  });

  return e;
}

//
// A generic kernel to consume data from a SYCL pipe and write it to host memory
//
// The following is a block diagram of this kernel:
//
// |-----------------|         |--------------------------|
// | CPU    |-----|  | out_ptr | FPGA  |---| OutPipe      |
// |        | RAM |<-|---------|-------| C |<======== ... |
// |        |-----|  |         |       |---|              |
// |-----------------|         |--------------------------|
//
template<typename T, typename OutPipe>
event SubmitConsumer(queue &q, T* out_ptr, size_t size) {
  auto e = q.submit([&](handler& h) {
    h.single_task<C>([=]() [[intel::kernel_args_restrict]] {
      host_ptr<T> out(out_ptr);
      for (size_t i = 0; i < size; i++) {
        auto data = OutPipe::read();
        *(out + i) = data;
      }
    });
  });

  return e;
}

// A generic kernel that reads from an input pipe and writes to an output pipe
//
// The following is a block diagram of this kernel:
//
//      InPipe  |--------| OutPipe
// ... ========>| Kernel |=========> ...
//              |--------|
template<typename KernelClass, typename T, typename InPipe, typename OutPipe>
event SubmitSinglePipeWorker(queue &q, size_t size) {
  auto e = q.submit([&](handler& h) {
    h.single_task<KernelClass>([=]() [[intel::kernel_args_restrict]] {
      for (size_t i = 0; i < size; i++) {
        auto data = InPipe::read();
        // computation could be placed here
        OutPipe::write(data);
      }
    });
  });

  return e;
}

//
// This function creates the pipeline design for the multi-kernel version.
// It instantiates 3 SubmitSinglePipeWorker functions (above) and connects
// them with internal pipes Pipe0 and Pipe1. This function represents a generic
// pipeline made of 3 kernels (K0, K1, and K2) connected by pipes (a common
// FPGA design pattern). The function that calls this can connect the design
// to the InPipe and OutPipe in whatever fashion it likes.
//
// The following is a block diagram of this kernel this function creates:
//
//      InPipe  |----| Pipe0 |----| Pipe1 |----| OutPipe
// ... ========>| K0 |======>| K1 |======>| K2 |========> ...
//              |----|       |----|       |----|
//
template<typename T, typename InPipe, typename OutPipe>
std::vector<event> SubmitMultiKernelWorkers(queue &q, size_t size) {
  // internal pipes between kernels
  using Pipe0 = sycl::pipe<class Pipe0Class, T>;
  using Pipe1 = sycl::pipe<class Pipe1Class, T>;

  // submit the kernels
  event e0 = SubmitSinglePipeWorker<K0, T, InPipe, Pipe0>(q, size);
  event e1 = SubmitSinglePipeWorker<K1, T, Pipe0, Pipe1>(q, size);
  event e2 = SubmitSinglePipeWorker<K2, T, Pipe1, OutPipe>(q, size);

  // return the events
  return {e0, e1, e2};
}

#endif /* __MULTI_KERNEL_HPP__ */
