#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>

#include "exception_handler.hpp"

// Test related header files
#include "board_test.hpp"

int main(int argc, char* argv[]) {
  // Always print small help at the beginning of board test
  PrintHelp(0);

  // Default is to run all tests
  int test_to_run = 0;

  // test_to_run value changed according to user selection if user has provided
  // an input using "-test=<test_number>" option (see PrintHelp function for
  // test details)
  if (argc == 2) {
    std::string tmp_test(argv[1]);
    if ((tmp_test.compare(0, 6, "-test=")) == 0) {
      test_to_run = std::stoi(tmp_test.substr(6));
      if (test_to_run < 0 || test_to_run > 7) {
        std::cerr
            << "Not a valid test number, please select the correct test from "
            << "list above and re-run binary with updated value.\n\n";
        return 1;
      }
    } else if ((tmp_test.compare(0, 5, "-help")) == 0) {
      PrintHelp(1);
      return 0;
    } else {
      std::cerr << "Incorrect argument passed to ./board_test.fpga! Cannot run "
                << "test\n, please refer to usage information above for "
                << "correct command.\nTerminating test!\n";
      return 1;
    }
  }

  if (test_to_run > 0)
    std::cout << "User has selected to run only test number " << test_to_run
              << " from test list above\n";
  else
    std::cout << "Running all tests \n";

// Device Selection
// Select either:
//  - the FPGA emulator device (CPU emulation of the FPGA) using FPGA_EMULATOR
//  macro
//  - the FPGA device (a real FPGA)
#if FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  // Variable ORed with result of each test
  // Value of 0 at the end indicates all tests completed successfully
  int ret = 0;

  // Queue creation
  try {
    // queue properties to enable profiling
    sycl::property_list q_prop_list{sycl::property::queue::enable_profiling()};

    // Create a queue bound to the chosen device
    // If the device is unavailable, a SYCL runtime exception is thrown
    sycl::queue q(selector, fpga_tools::exception_handler, q_prop_list);

    auto device = q.get_device();

    // Print out the device information.
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>() 
              << std::endl;

    // Create a oneAPI Shim object
    ShimMetrics hldshim(q);

    // Test 1 - Host speed and host read write test
    if (test_to_run == 0 || test_to_run == 1) {
      std::cout << "\n*****************************************************************\n"
                << "*********************** Host Speed Test *************************\n"
                << "*****************************************************************\n\n";

      ret |= hldshim.HostSpeed(q);

      std::cout << "\n*****************************************************************\n"
                << "********************* Host Read Write Test **********************\n"
                << "*****************************************************************\n\n";

      // Tasks run in Host read write with dev_offset=0 (aligned dev addr):
      // 1. write from unaligned host memory to aligned device memory address
      // 2. read from aligned device memory to aligned host memory
      size_t dev_offset = 0;
      int ret_hostrw = hldshim.HostRWTest(q, dev_offset);
      if (ret_hostrw != 0) {
        std::cerr << "Error: Host read write test failed with aligned dev "
                  << "address (dev_offset = " << dev_offset << ")\n"
                  << "\nBOARD TEST FAILED\n";
        return 1;
      }
// Do not run on SoC boards. ARM processor does not support unaligned memory
// access
#ifndef __arm__
      // unaligned device addr
      dev_offset = 3;
      ret_hostrw = hldshim.HostRWTest(q, dev_offset);
      if (ret_hostrw != 0) {
        std::cerr << "Error: Host read write test failed with unaligned dev "
                  << "address (dev_offset = " << dev_offset << ")\n"
                  << "\nBOARD TEST FAILED\n";
        return 1;
      }
#endif

      if (!ret_hostrw) std::cout << "\nHOST READ-WRITE TEST PASSED!\n";

      ret |= ret_hostrw;
    }

    // Test 2 - Kernel clock frequency (this is measured before all tests except
    // 1)
    if (test_to_run == 0 || test_to_run == 5 || test_to_run == 2 ||
        test_to_run == 3 || test_to_run == 4 || test_to_run == 6 ||
        test_to_run == 7) {
      std::cout << "\n*****************************************************************\n"
                << "*******************  Kernel Clock Frequency Test  ***************\n"
                << "*****************************************************************\n\n";

#if defined(FPGA_EMULATOR)
      std::cout << "Kernel clock frequency test is not run in emulation, "
                << "skipping this test \n";
#else
      // This test offers the option to skip reading compiled kernel frequency
      // from reports using report_chk variable (either user does not have
      // reports or purposely wants to skip when there is a bigger mismatch in
      // measured vs compiled value) Ideally should be true as skipping this
      // check can lead to erroneous behavior due to kernel clock frequency
      // issues on hardware
      bool report_chk = true;
      int ret_kernel_clk = hldshim.KernelClkFreq(q, report_chk);
      // If test failed and user has selected to check the Quartus compiled
      // frequency, then terminate test
      if (ret_kernel_clk && report_chk) {
        std::cerr << "\nBOARD TEST FAILED\n";
        return 1;
      }

      ret |= ret_kernel_clk;
#endif
    }

    // Test 3 - Kernel Launch Test
    if (test_to_run == 0 || test_to_run == 3) {
      std::cout << "\n*****************************************************************\n"
                << "********************* Kernel Launch Test ************************\n"
                << "*****************************************************************\n\n";

      int ret_kernel_launch_test = hldshim.KernelLaunchTest(q);
      if (ret_kernel_launch_test == 0) {
        std::cout << "\nKERNEL_LAUNCH_TEST PASSED\n";
      } else {
        std::cerr << "Error: Kernel Launch Test Failed\n"
                  << "\nBOARD TEST FAILED\n";
        return 1;
      }

      ret |= ret_kernel_launch_test;
    }

    // Test 4 - Kernel Latency Measurement
    if (test_to_run == 0 || test_to_run == 4) {
      std::cout << "\n*****************************************************************\n"
                << "********************  Kernel Latency  **************************\n"
                << "*****************************************************************\n\n";

      ret |= hldshim.KernelLatency(q);
    }

    // Test 5 - Kernel-to-Memory Read Write Test
    if (test_to_run == 0 || test_to_run == 5) {
      std::cout << "\n*****************************************************************\n"
                << "*************  Kernel-to-Memory Read Write Test  ***************\n"
                << "*****************************************************************\n\n";

      int ret_kernel_mem_rw = hldshim.KernelMemRW(q);
      if (ret_kernel_mem_rw != 0) {
        std::cerr << "Error: kernel-memory read write test failed. \n"
                  << "\nBOARD TEST FAILED\n";
        return 1;
      }

      ret |= ret_kernel_mem_rw;
    }

    // Test 6 - Kernel-to-Memory Bandwidth Measurement
    if (test_to_run == 0 || test_to_run == 6) {
      std::cout << "\n*****************************************************************\n"
                << "*****************  Kernel-to-Memory Bandwidth  *****************\n"
                << "*****************************************************************\n\n";

      ret |= hldshim.KernelMemBW(q);
    }
  }  // End of try block

  catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                << "system has a correctly configured FPGA board.\n"
                << "Run sys_check in the oneAPI root directory to verify.\n"
                << "If you are targeting the FPGA emulator, compile with "
                << "-DFPGA_EMULATOR.\n";
    }
    // Caught exception, terminate program
    std::terminate();
  }  // End of catch block

  // If value of returns from all tests is 0 (i.e. without
  // errors) - Board test passed
  if (ret == 0)
    std::cout << "\nBOARD TEST PASSED\n";
  else
    std::cerr << "\nBOARD TEST FAILED\n";
  return ret;

}  // End of main
