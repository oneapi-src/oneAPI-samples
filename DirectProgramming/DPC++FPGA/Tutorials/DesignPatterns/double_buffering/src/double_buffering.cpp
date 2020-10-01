//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <random>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// kTimes = # times to execute the kernel. kTimes must be >= 2
// kSize = # of floats to process on each kernel execution.
// run less in emulation to avoid high run time
#if defined(FPGA_EMULATOR)
constexpr int kTimes = 20;
constexpr int kSize = 4096;
#else
constexpr int kTimes = 100;
constexpr int kSize = 2621440;
#endif

// Kernel executes a power function (base^kPow). Must be
// >= 2. Can increase this to increase kernel execution
// time, but ProcessOutput() time will also increase.
constexpr int kPow = 20;

// Number of iterations through the main loop
constexpr int kNumRuns = 2;

bool pass = true;

class SimpleVpow;

/*  Kernel function.
    Performs buffer_b[i] = buffer_a[i] ** pow
    Only supports pow >= 2.
    This kernel is not meant to be an optimal implementation of the power
   operation -- it's just a sample kernel for this tutorial whose execution time
   is easily controlled via the pow parameter. SYCL buffers are created
   externally and passed in by reference to control (external to this function)
   when the buffers are destructed. The destructor causes a blocking buffer
   transfer from device to host and double buffering requires us to not block
   here (because we need to launch another kernel). So we only want this
   transfer to occur at the end of overall execution, not at the end of each
   individual kernel execution.
*/
void SimplePow(std::unique_ptr<queue> &q, buffer<float, 1> &buffer_a,
               buffer<float, 1> &buffer_b, event &e) {
  // Submit to the queue and execute the kernel
  e = q->submit([&](handler &h) {
    // Get kernel access to the buffers
    auto accessor_a = buffer_a.get_access<access::mode::read>(h);
    auto accessor_b = buffer_b.get_access<access::mode::discard_read_write>(h);

    const int num = kSize;
    assert(kPow >= 2);
    const int p = kPow - 1;  // Assumes pow >= 2;

    h.single_task<SimpleVpow>([=]() [[intel::kernel_args_restrict]] {
      for (int j = 0; j < p; j++) {
        if (j == 0) {
          for (int i = 0; i < num; i++) {
            accessor_b[i] = accessor_a[i] * accessor_a[i];
          }
        } else {
          for (int i = 0; i < num; i++) {
            accessor_b[i] = accessor_b[i] * accessor_a[i];
          }
        }
      }
    });
  });

  event update_host_event;
  update_host_event = q->submit([&](handler &h) {
    auto accessor_b = buffer_b.get_access<access::mode::read>(h);

    /*
      Explicitly instruct the SYCL runtime to copy the kernel's output buffer
      back to the host upon kernel completion. This is not required for
      functionality since the buffer access in ProcessOutput() also implicitly
      instructs the runtime to copy the data back. But it should be noted that
      this buffer access blocks ProcessOutput() until the kernel is complete
      and the data is copied. In contrast, update_host() instructs the runtime
      to perform the copy earlier. This allows ProcessOutput() to optionally
      perform more useful work *before* making the blocking buffer access. Said
      another way, this allows ProcessOutput() to potentially perform more work
      in parallel with the runtime's copy operation.
    */
    h.update_host(accessor_b);
  });
}

// Returns kernel execution time for a given SYCL event from a queue.
ulong SyclGetExecTimeNs(event e) {
  ulong start_time =
      e.get_profiling_info<info::event_profiling::command_start>();
  ulong end_time =
      e.get_profiling_info<info::event_profiling::command_end>();
  return (end_time - start_time);
}

// Local pow function for verifying results
float MyPow(float input, int pow) {
  return (pow == 0) ? 1 : input * MyPow(input, pow - 1);
}

/*  Compares kernel output against expected output. Only compares part of the
   output so that this method completes quickly. This is done
   intentionally/artificially keep host-processing time shorter than kernel
   execution time. Grabs kernel output data from its SYCL buffer. Reading from
   this buffer is a blocking operation that will block on the kernel completing.
    Queries and records execution time of the kernel that just completed. This
   is a natural place to do this because ProcessOutput() is blocked on kernel
   completion.
*/
void ProcessOutput(buffer<float, 1> &input_buf,
                   buffer<float, 1> &output_buf, int exec_number, event e,
                   ulong &total_kernel_time_per_slot) {
  auto input_buf_acc = input_buf.get_access<access::mode::read>();
  auto output_buf_acc = output_buf.get_access<access::mode::read>();
  int num_errors = 0;
  int num_errors_to_print = 10;
  /*  The use of update_host() in the kernel function allows for additional
     host-side operations to be performed here, in parallel with the buffer copy
     operation from device to host, before the blocking access to the output
     buffer is made via output_buf_acc[]. To be clear, no real operations are
     done here and this is just a note that this is the place
      where you *could* do it. */
  for (int i = 0; i < kSize / 8; i++) {
    const bool out_valid = (MyPow(input_buf_acc[i], kPow) != output_buf_acc[i]);
    if ((num_errors < num_errors_to_print) && out_valid) {
      if (num_errors == 0) {
        pass = false;
        std::cout << "Verification failed on kernel execution # " << exec_number
                  << ". Showing up to " << num_errors_to_print
                  << " mismatches.\n";
      }
      std::cout << "Verification failed on kernel execution # " << exec_number
                << ", at element " << i << ". Expected " << std::fixed
                << std::setprecision(16) << MyPow(input_buf_acc[i], kPow)
                << " but got " << output_buf_acc[i] << "\n";
      num_errors++;
    }
  }

  // At this point we know the kernel has completed,
  // so can query the profiling data.
  total_kernel_time_per_slot += SyclGetExecTimeNs(e);
}

/*
    Generates input data for the next kernel execution. Only fills part of the
   buffer so that this method completes quickly. This is done
   intentionally/artificially keep host-processing time shorter than kernel
   execution time. Writes the data into the associated SYCL buffer. The write
   will block until the previous kernel execution, that is using this buffer,
   completes.
*/
void ProcessInput(buffer<float, 1> &buf) {
  // We are generating completely new input data, so can use discard_write()
  // here to indicate we don't care about the SYCL buffer's current contents.
  auto buf_acc = buf.get_access<access::mode::discard_write>();

  // RNG seed
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();

  // RNG engine
  std::default_random_engine dre(seed);

  // generate random numbers between 1 and 2
  std::uniform_real_distribution<float> di(1.0f, 2.0f);

  // Randomly generate a start value and increment from there.
  // Compared to randomly generating every value, this is done to
  // speed up this function a bit.
  float start_val = di(dre);

  for (int i = 0; i < kSize / 8; i++) {
    buf_acc[i] = start_val;
    start_val++;
  }
}

int main() {
// Create queue, get platform and device
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector device_selector;
    std::cout << "\nEmulator output does not demonstrate true hardware "
                 "performance. The design may need to run on actual hardware "
                 "to observe the performance benefit of the optimization "
                 "exemplified in this tutorial.\n\n";
#else
    INTEL::fpga_selector device_selector;
#endif

    try {
      auto prop_list =
          property_list{property::queue::enable_profiling()};

      std::unique_ptr<queue> q;
      q.reset(new queue(device_selector, dpc_common::exception_handler, prop_list));

      platform platform = q->get_context().get_platform();
      device device = q->get_device();
      std::cout << "Platform name: "
                << platform.get_info<info::platform::name>().c_str() << "\n";
      std::cout << "Device name: "
                << device.get_info<info::device::name>().c_str() << "\n\n\n";

      std::cout << "Executing kernel " << kTimes << " times in each round.\n\n";

      // Create a vector to store the input/output SYCL buffers
      std::vector<buffer<float, 1>> input_buf;
      std::vector<buffer<float, 1>> output_buf;

      // SYCL events for each kernel launch.
      event sycl_events[2];

      // In nanoseconds. Total execution time of kernels in a given slot.
      ulong total_kernel_time_per_slot[2];

      // Total execution time of all kernels.
      ulong total_kernel_time = 0;

      // Allocate vectors to store the host-side copies of the input data
      // Create and allocate the SYCL buffers
      for (int i = 0; i < 2; i++) {
        input_buf.push_back(buffer<float, 1>(range<1>(kSize)));
        output_buf.push_back(buffer<float, 1>(range<1>(kSize)));
      }

      /*
        Main loop. This loop runs twice to show the performance difference without
        and with double buffering.
      */
      for (int i = 0; i < kNumRuns; i++) {
        for (int i = 0; i < 2; i++) {
          total_kernel_time_per_slot[i] = 0;  // Initialize timers to zero.
        }

        switch (i) {
          case 0: {
            std::cout << "*** Beginning execution, without double buffering\n";
            break;
          }
          case 1: {
            std::cout << "*** Beginning execution, with double buffering.\n";
            break;
          }
          default: {
            std::cout << "*** Beginning execution.\n";
          }
        }

        // Start the timer. This will include the time to process the input data
        // for the first 2 kernel executions.
        dpc_common::TimeInterval exec_time;

        if (i == 0) {  // Single buffering
          for (int i = 0; i < kTimes; i++) {
            // Only print every few iterations, just to limit the prints.
            if (i % 10 == 0) {
              std::cout << "Launching kernel #" << i << "\n";
            }

            ProcessInput(input_buf[0]);
            SimplePow(q, input_buf[0], output_buf[0], sycl_events[0]);
            ProcessOutput(input_buf[0], output_buf[0], i, sycl_events[0],
                          total_kernel_time_per_slot[0]);
          }
        } else {  // Double buffering
          // Process input for first 2 kernel launches and queue them. Then block
          // on processing the output of the first kernel.
          ProcessInput(input_buf[0]);
          ProcessInput(input_buf[1]);

          std::cout << "Launching kernel #0\n";

          SimplePow(q, input_buf[0], output_buf[0], sycl_events[0]);
          for (int i = 1; i < kTimes; i++) {
            if (i % 10 == 0) {
              std::cout << "Launching kernel #" << i << "\n";
            }  // Only print every few iterations, just to limit the prints.

            // Launch the next kernel
            SimplePow(q, input_buf[i % 2], output_buf[i % 2], sycl_events[i % 2]);

            // Process output from previous kernel. This will block on kernel
            // completion.
            ProcessOutput(input_buf[(i - 1) % 2], output_buf[(i - 1) % 2], i,
                          sycl_events[(i - 1) % 2],
                          total_kernel_time_per_slot[(i - 1) % 2]);

            // Generate input for the next kernel.
            ProcessInput(input_buf[(i - 1) % 2]);
          }

          // Process output of the final kernel
          ProcessOutput(input_buf[(kTimes - 1) % 2], output_buf[(kTimes - 1) % 2],
                        i, sycl_events[(kTimes - 1) % 2],
                        total_kernel_time_per_slot[(kTimes - 1) % 2]);
        }

        // Add up the overall kernel execution time.
        total_kernel_time = 0;
        for (int i = 0; i < 2; i++) {
          total_kernel_time += total_kernel_time_per_slot[i];
        }

        // Stop the timer.
        double time_span = exec_time.Elapsed();

        std::cout << "\nOverall execution time "
                  << ((i == 0) ? "without" : "with") << " double buffering = "
                  << (unsigned)(time_span * 1000) << " ms\n";
        std::cout << "Total kernel-only execution time "
                  << ((i == 0) ? "without" : "with") << " double buffering = "
                  << (unsigned)(total_kernel_time / 1000000) << " ms\n";
        std::cout << "Throughput = " << std::setprecision(8)
                  << (float)kSize * (float)kTimes * (float)sizeof(float) /
                         (float)time_span / 1000000
                  << " MB/s\n\n\n";
      }
      if (pass) {
        std::cout << "Verification PASSED\n";
      } else {
        std::cout << "Verification FAILED\n";
        return 1;
      }
    } catch (sycl::exception const& e) {
      // Catches exceptions in the host code
      std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

      // Most likely the runtime couldn't find FPGA hardware!
      if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
        std::cout << "If you are targeting an FPGA, please ensure that your "
                     "system has a correctly configured FPGA board.\n";
        std::cout << "If you are targeting the FPGA emulator, compile with "
                     "-DFPGA_EMULATOR.\n";
      }
      std::terminate();
    }
  return 0;
}
