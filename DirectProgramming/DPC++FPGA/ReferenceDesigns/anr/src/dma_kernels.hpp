#ifndef __DMA_KERNELS_HPP__
#define __DMA_KERNELS_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "data_bundle.hpp"

using namespace sycl;
using namespace hldutils;

//
// This kernel provides data into the ANR input pipe from either device memory
// (disable_global_mem == false) or dummy data to avoid connecting to device
// memory (disable_global_mem == true). The latter allows use to avoid
// having a global memory interconnect in our kernel system for testing the IP.
//
template<typename KernelId, typename T, typename Pipe, int pixels_per_cycle,
         bool disable_global_mem>
event SubmitInputDMA(queue& q, T *in_ptr, int rows, int cols, int frames) {
  using PipeType = DataBundle<T, pixels_per_cycle>;

  // LSU attribute to  turn off caching
  using NonCachingLSU = INTEL::lsu<INTEL::burst_coalesce<true>,
                                   INTEL::cache<0>,
                                   INTEL::statically_coalesce<true>,
                                   INTEL::prefetch<false>>;

  assert(((rows * cols) % pixels_per_cycle) == 0);
  const int iterations = cols * rows / pixels_per_cycle;

  if constexpr (!disable_global_mem) {
    return q.submit([&](handler &h) {
      h.single_task<KernelId>([=]() [[intel::kernel_args_restrict]] {
        device_ptr<T> in(in_ptr);

        [[intel::loop_coalesce(2)]]
        for (int f = 0; f < frames; f++) {
          for (int i = 0; i < iterations; i++) {
            PipeType pipe_data;
            #pragma unroll
            for (int k = 0; k < pixels_per_cycle; k++) {
              pipe_data[k] = NonCachingLSU::load(in + i * pixels_per_cycle + k);
            }
            Pipe::write(pipe_data);
          }
        }
      });
    });
  } else {
    return q.submit([&](handler &h) {
      h.single_task<KernelId>([=] {
        [[intel::loop_coalesce(2)]]
        for (int f = 0; f < frames; f++) {
          for (int i = 0; i < iterations; i++) {
            PipeType pipe_data;
            #pragma unroll
            for (int k = 0; k < pixels_per_cycle; k++) {
              pipe_data[k] = (i * pixels_per_cycle + k);
            }
            Pipe::write(pipe_data);
          }
        }
      });
    });
  }
}

//
// This kernel pulls data out of the ANR output pipe and either writes it to
// device memory (disable_global_mem == false) or drops it to avoid connecting
// to device memory (disable_global_mem == true). The latter allows use to avoid
// having a global memory interconnect in our kernel system for testing the IP.
//
template<typename KernelId, typename T, typename Pipe, int pixels_per_cycle,
         bool disable_global_mem>
event SubmitOutputDMA(queue& q, T *out_ptr, int rows, int cols, int frames) {
  assert(((rows * cols) % pixels_per_cycle) == 0);
  const int iterations = cols * rows / pixels_per_cycle;

  if constexpr(!disable_global_mem) {
    return q.submit([&](handler &h) {
      h.single_task<KernelId>([=]() [[intel::kernel_args_restrict]] {
        device_ptr<T> out(out_ptr);
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
    });
  } else {
    return q.submit([&](handler &h) {
      h.single_task<KernelId>([=] {      
        [[intel::loop_coalesce(2)]]
        for (int f = 0; f < frames; f++) {
          for (int i = 0; i < iterations; i++) {
            (void)Pipe::read();
          }
        }
      });
    });
  }
}

#endif /* __DMA_KERNELS_HPP__ */