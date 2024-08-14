#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "restartable_counter_kernel.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class CounterID;

constexpr int kIterations = 256;

// Read integers from `PipeType` and check that they start at some expected
// value and increment by one each time. Optionally, flush data from `PipeType`
// until a `start_of_packet` is seen.
template <typename PipeType>
bool CheckIncrements(sycl::queue q, int count_start, int iterations,
                     bool should_flush = true) {
  bool passed = true;
  int expected_count = count_start;

  if (should_flush) {
    std::cout << "Flush pipe until 'start of packet' is seen." << std::endl;
  } 
    else {
      std::cout << "Start counting from " << expected_count << std::endl;
  }

  int flushed_count = 0;
  for (int itr = 0; itr < iterations; itr++) {
    restartable_counter::OutputBeat beat = PipeType::read(q);

    // Flush the pipe in case we are starting fresh.
    if (should_flush && itr == 0) {
      while (beat.sop != true) {
        beat = PipeType::read(q);
        flushed_count++;
      }
      std::cout << "\tFlushed " << flushed_count << " beats." << std::endl;
      std::cout << "Start counting from " << expected_count << std::endl;
    }

    int calculated_count = beat.data;
    if (calculated_count != expected_count) {
      std::cout << "\nitr=" << itr << ": result " << calculated_count
                << ", expected (" << expected_count << ")";
      passed = false;
    }
    expected_count++;
  }

  return passed;
}

int main() {
  bool passed = false;

  try {
    // Use compile-time macros to select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    //  - the simulator device
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;
    {
      int count_start = 7;
      std::cout << "\nStart kernel RestartableCounter at " << count_start << ". "
                << std::endl;

      // Capture the event so that we can stop the kernel later on
      sycl::event e = q.single_task<CounterID>(
          restartable_counter::RestartableCounter{count_start});

      passed = CheckIncrements<restartable_counter::OutputPipe>(q, count_start,
                                                              kIterations);

      int new_start = count_start + kIterations;
      // continue reading more results
      passed &= CheckIncrements<restartable_counter::OutputPipe>(
          q, new_start, kIterations, false);

      std::cout << "Stop kernel RestartableCounter" << std::endl;
      // Write a `true` into `StopPipe` to instruct the kernel to break out of
      // its main loop, then wait for the kernel to complete.
      restartable_counter::StopPipe::write(q, true);
      e.wait();
    }
    {
      int count_start = 77;
      std::cout << "\nStart RestartableCounter at " << count_start << "."
                << std::endl;
      sycl::event e = q.single_task<CounterID>(
          restartable_counter::RestartableCounter{count_start});
      passed &= CheckIncrements<restartable_counter::OutputPipe>(q, count_start,
                                                               kIterations);

      std::cout << "Stop kernel RestartableCounter" << std::endl;
      restartable_counter::StopPipe::write(q, true);
      e.wait();
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code.
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}