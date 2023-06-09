// Header file to accompany board_test
#include <sycl/sycl.hpp>

constexpr size_t kKB = 1024;
constexpr size_t kMB = 1024 * 1024;
constexpr size_t kGB = 1024 * 1024 * 1024;
constexpr size_t kRandomSeed = 1009;

#if defined(_WIN32) || defined(_WIN64)
  std::string kBinaryName = "board_test.fpga.exe";
#elif __linux__
  std::string kBinaryName = "board_test.fpga";
  #define _popen popen
  #define _pclose pclose
#endif

//////////////////////////////////
// **** PrintHelp function **** //
//////////////////////////////////

// Input:
// int details - Selection between long help or short help
// Returns:
// None

// The function does the following task:
// Prints short help with usage infomation and a longer help with details about
// each test

void PrintHelp(int details) {
  if (!details) {
    std::cout << "\n*** Board_test usage information ***\n"
              << "Command to run board_test using generated binary:\n"
              << "  > To run all tests (default): run " << kBinaryName << "\n"
              << "  > To run a specific test (see list below); pass the test "
              << "number as argument to \"-test\" option: \n" 
              << "  Linux: ./board_test.fpga -test=<test_number>\n"
              << "  Windows: board_test.exe -test=<test_number>\n"
              << "  > To see more details on what each test does use"
              << " -help option\n"
              << "The tests are:\n"
              << "  1. Host Speed and Host Read Write Test\n"
              << "  2. Kernel Clock Frequency Test\n"
              << "  3. Kernel Launch Test\n"
              << "  4. Kernel Latency Measurement\n"
              << "  5. Kernel-to-Memory Read Write Test\n"
              << "  6. Kernel-to-Memory Bandwidth Test\n"
              << "Note: Kernel Clock Frequency is run along with all tests "
              << "except 1 (Host Speed and Host Read Write test)\n\n";
  } else {
    std::cout
        << "*** Board_test test details ***\n"
        << "The tests are:\n\n"
        << "  * 1. Host Speed and Host Read Write Test *\n"
        << "    Host Speed and Host Read Write test check the host to device "
        << "interface\n"
        << "    Host Speed test measures the host to device global memory "
        << "read, write as well as read-write bandwidth and reports it\n"
        << "    Host Read Write Test writes does unaligned memory to unaligned "
        << "device memory writes as well as reads from device to unaligned "
        << "host memory\n\n"
        << "  * 2. Kernel Clock Frequency Test *\n"
        << "    Kernel Clock Frequency Test measures the kernel clock "
        << "frequency of the bitstream running on the FPGA and compares this "
        << "to the Quartus compiled frequency for the kernel.\n\n"
        << "  * 3. Kernel Launch Test *\n"
        << "    Kernel Launch test checks if the kernel launched and executed "
        << "successfully. This is done by launching a sender kernel that "
        << "writes a value to a pipe, this pipe is read by the receiver "
        << "kernel, which completes if the correct value is read.\n"
        << "    This test will hang if the receiver kernel does not receive "
        << "the correct value\n\n"
        << "  * 4. Kernel Latency Measurement *\n"
        << "    This test measures the round trip kernel latency by launching "
        << "a no-operation kernel\n\n"
        << "  * 5. Kernel-to-Memory Read Write Test *\n"
        << "    Kernel-to-Memory Read Write test checks kernel to device "
        << "global memory interface. The test writes data to the entire device "
        << "global memory from host; the kernel then reads -> modifies and "
        << "writes the data back to the device global memory."
        << "    The host reads the modified data back and verifies the read "
        << "back values match expected value\n"
        << "  * 6. Kernel-to-Memory Bandwidth Test *\n"
        << "    Kernel-to-Memory Bandwidth test measures the kernel to device "
        << "global memory bandwidth and compares this with the theoretical "
        << "bandwidth defined in board_spec.xml file in the oneAPI shim/BSP.\n\n"
        << "    Note: This test assumes that design was compiled with "
        << "-Xsno-interleaving option\n\n"
        << "Please use the commands shown at the beginning of this help to run "
        << "all or one of the above tests\n\n";
  }
}  // End of PrintHelp

/////////////////////////////////////////////
// **** SyclGetQStExecTimeNs function **** //
/////////////////////////////////////////////

// Input:
// event e - Sycl event with profiling information
// Returns:
// Difference in time from command start to command end (in nanoseconds)

// The function does the following task:
// Gets profiling information from a Sycl event and
// returns execution time for a given SYCL event from a queue

unsigned long SyclGetQStExecTimeNs(sycl::event e) {
  unsigned long start_time =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  unsigned long end_time =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return (end_time - start_time);
}  // End of SyclGetQStExecTimeNs

///////////////////////////////////////////
// **** SyclGetTotalTimeNs function **** //
///////////////////////////////////////////

// Input:
// event first_evt - Sycl event with profiling information
// event last_evt - another Sycl event with profiling information
// Returns:
// Difference in time from command submission of first event to command end of
// last event (in nanoseconds)

// The function does the following task:
// Gets profiling information from two different Sycl events and
// returns the total execution time for all events between first and last

unsigned long SyclGetTotalTimeNs(sycl::event first_evt, sycl::event last_evt) {
  unsigned long first_evt_start =
      first_evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  unsigned long last_evt_end =
      last_evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  return (last_evt_end - first_evt_start);
}  // End of SyclGetTotalTimeNs

/////////////////////////////////////////
// **** InitializeVector function **** //
/////////////////////////////////////////

// Inputs:
// 1. unsigned *vector - pointer to host memory that has to be initialized
// (allocated in calling function)
// 2. size_t size - number of elements to initialize
// 3. size_t offset - value to use for initialization
// Returns:
// None

// The function does the following task:
// Initializes "size" number of elements in memory pointed
// to with "offset + i", where i is incremented by loop controlled by "size"

void InitializeVector(unsigned *vector, size_t size, size_t offset) {
  for (size_t i = 0; i < size; ++i) {
    vector[i] = offset + i;
  }
}

/////////////////////////////////////////
// **** InitializeVector function **** //
/////////////////////////////////////////

// Inputs:
// 1. unsigned *vector - pointer to host memory that has to be initialized
// (allocated in calling function)
// 2. size_t size - number of elements to initialize
// Returns:
// None

// The function does the following task:
// Initializes "size" number of elements in memory pointed
// to with random values (output of rand() function)

void InitializeVector(unsigned *vector, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    vector[i] = rand();
  }
}