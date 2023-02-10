#ifndef __DMA_KERNELS_HPP__
#define __DMA_KERNELS_HPP__

//
// This file contains the kernels for reading from device memory and
// streaming into the ANR input pipe, as well as the kernels for reading from
// the ANR output pipe and writing to device memory.
//

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "data_bundle.hpp"

using namespace sycl;
using namespace fpga_tools;

//
// Kernel to read data from device memory and write it into the ANR input pipe.
//
template <typename KernelId, typename T, typename Pipe, int pixels_per_cycle>
event SubmitInputDMA(queue &q, T *in_ptr, int rows, int cols, int frames) {
  using PipeType = DataBundle<T, pixels_per_cycle>;

#if defined (IS_BSP)
  // LSU attribute to  turn off caching
  using NonCachingLSU =
      ext::intel::lsu<ext::intel::burst_coalesce<true>, ext::intel::cache<0>,
                      ext::intel::statically_coalesce<true>,
                      ext::intel::prefetch<false>>;
#endif 

  // validate the number of columns
  if ((cols % pixels_per_cycle) != 0) {
    std::cerr << "ERROR: the number of columns is not a multiple of the pixels "
              << "per cycle\n";
    std::terminate();
  }

  // the number of iterations is the number of total pixels (rows*cols)
  // divided by the number of pixels per cycle
  const int iterations = cols * rows / pixels_per_cycle;

  // Using device memory
  return q.single_task<KernelId>([=]() [[intel::kernel_args_restrict]] {

#if defined (IS_BSP)
    device_ptr<T> in(in_ptr);
#else 
    T* in(in_ptr);
#endif  

    // coalesce the following two loops into a single for-loop using the
    // loop_coalesce attribute
    [[intel::loop_coalesce(2)]]
    for (int f = 0; f < frames; f++) {
      for (int i = 0; i < iterations; i++) {
        PipeType pipe_data;
        #pragma unroll
        for (int k = 0; k < pixels_per_cycle; k++) {
#if defined (IS_BSP)
          pipe_data[k] = NonCachingLSU::load(in + i * pixels_per_cycle + k);
#else 
          pipe_data[k] = in[i * pixels_per_cycle + k];
#endif   
        }
        Pipe::write(pipe_data);
      }
    }
  });
}

//
// Kernel to pull data out of the ANR output pipe and writes to device memory.
//
template <typename KernelId, typename T, typename Pipe, int pixels_per_cycle>
event SubmitOutputDMA(queue &q, T *out_ptr, int rows, int cols, int frames) {
  // validate the number of columns
  if ((cols % pixels_per_cycle) != 0) {
    std::cerr << "ERROR: the number of columns is not a multiple of the pixels "
              << "per cycle\n";
    std::terminate();
  }

  // the number of iterations is the number of total pixels (rows*cols)
  // divided by the number of pixels per cycle
  const int iterations = cols * rows / pixels_per_cycle;

  // Using device memory
  return q.single_task<KernelId>([=]() [[intel::kernel_args_restrict]] {

#if defined (IS_BSP)
    device_ptr<T> out(out_ptr);
#else 
    T* out(out_ptr);
#endif      

    // coalesce the following two loops into a single for-loop using the
    // loop_coalesce attribute
    [[intel::loop_coalesce(2)]]
    for (int f = 0; f < frames; f++) {
      for (int i = 0; i < iterations; i++) {
        auto pipe_data = Pipe::read();
        #pragma unroll
        for (int k = 0; k < pixels_per_cycle; k++) {
          out[i * pixels_per_cycle + k] = pipe_data[k];
        }
      }
    }
});
}

#endif /* __DMA_KERNELS_HPP__ */