#ifndef __LOOPBACKKERNEL_HPP__
#define __LOOPBACKKERNEL_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

// Forward declare the kernel names to reduce name mangling
class Kernel1;
class Kernel2;
class Kernel3;

//
// A simple loopback kernel. Returns the launched kernel event.
// Kernel structure:
//
//  InputPipe |-------| OutputPipe
// ==========>|Kernel1|===========>
//            |-------|
//
//
template <typename KernelClass, typename T, typename InputPipe,
          typename OutputPipe>
event SubmitLoopbackKernel(queue& q, size_t max_size) {
  auto e = q.submit([&](handler& h) {
    // NO-FORMAT comments are for clang-format
    h.single_task<KernelClass>([=
    ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
      for (size_t i = 0; i < max_size; i++) {
        T data = InputPipe::read();
        OutputPipe::write(data);
      }
    });
  });

  return e;
}

//
// Simple loopback kernel, but with three internal kernels connected with
// two internal pipes. Returns a list of events; one for each kernel.
// Kernel structure:
//
//  InputPipe |-------| Pipe1 |-------| Pipe2  |-------| OutputPipe
// ==========>|Kernel1|======>|Kernel2|=======>|Kernel3|===========>
//            |-------|       |-------|        |-------|
//
//
template <typename T, typename InputPipe, typename OutputPipe>
std::vector<event> SubmitLongLoopbackKernel(queue& q, size_t max_size) {
  // the internal pipes
  using Pipe1 = sycl::INTEL::pipe<class Pipe1Class, T>;
  using Pipe2 = sycl::INTEL::pipe<class Pipe2Class, T>;

  // kernel 1
  auto e1 =
      SubmitLoopbackKernel<class Kernel1, T, InputPipe, Pipe1>(q, max_size);

  // kernel 2
  auto e2 = SubmitLoopbackKernel<class Kernel2, T, Pipe1, Pipe2>(q, max_size);

  // kernel 2
  auto e3 =
      SubmitLoopbackKernel<class Kernel3, T, Pipe2, OutputPipe>(q, max_size);

  return {e1, e2, e3};
}

#endif /* __LOOPBACKKERNEL_HPP__ */
