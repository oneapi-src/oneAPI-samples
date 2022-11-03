#include <iostream>

// oneAPI headers
#include <sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/host_pipes.hpp>

using namespace sycl;

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class Add;

// Forward declare pipe names to reduce name mangling (maybe get rid of; pipes already have bad names)
class ID_PipeOut;

// use host pipes to write into addresses in the CSR
using OutputPipe = sycl::ext::intel::prototype::pipe<ID_PipeOut, int, 1,
                                                         // these 3 shouldn't matter
                                                         0, 1, true, false,
                                                         // store the most recently processed index
                                                         sycl::ext::intel::prototype::internal::protocol_name::AVALON_MM>;

class Add_Kernel
{
public:
    int a;
    int b;

    void operator()() const
    {
        int sum = a + b;

        OutputPipe::write(sum);
    }
};

// Forward declare pipe names to reduce name mangling (maybe get rid of; pipes already have bad names)
class ID_A;
class ID_B;
class ID_C;

// use host pipes to read from addresses in the CSR
using InputPipeA = sycl::ext::intel::prototype::pipe<ID_A, int, 1,
                                                         // choose defaults for these so we can set the 
                                                         0, 1, true, false,
                                                         // store the most recently processed index
                                                         sycl::ext::intel::prototype::internal::protocol_name::AVALON_MM>;
using InputPipeB = sycl::ext::intel::prototype::pipe<ID_B, int, 1,
                                                         // these 3 shouldn't matter
                                                         0, 1, true, false,
                                                         // store the most recently processed index
                                                         sycl::ext::intel::prototype::internal::protocol_name::AVALON_MM>;

// use host pipes to write into addresses in the CSR
using OutputPipeC = sycl::ext::intel::prototype::pipe<ID_C, int, 1,
                                                          // these 3 shouldn't matter
                                                          0, 1, true, false,
                                                          // store the most recently processed index
                                                          sycl::ext::intel::prototype::internal::protocol_name::AVALON_MM>;

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class AddCSRPipes;

class AddCSRPipes_Kernel
{
public:
    void operator()() const
    {
        int a = InputPipeA::read();
        int b = InputPipeB::read();

        int sum = a + b;

        OutputPipeC::write(sum);
    }
};

int main()
{
    bool passed = false;

    try{

    // choose a selector that was selected by the default FPGA build system.
    #if FPGA_SIMULATOR
    std::cout << "using FPGA Simulator." << std::endl;
    sycl::queue q(sycl::ext::intel::fpga_simulator_selector{});
#elif FPGA_HARDWARE
    std::cout << "using FPGA Hardware." << std::endl;
    sycl::queue q(sycl::ext::intel::fpga_selector{});
#else  // #if FPGA_EMULATOR
    std::cout << "using FPGA Emulator." << std::endl;
    sycl::queue q(sycl::ext::intel::fpga_emulator_selector{});
#endif

    int a = 3;
    int b = 76;

    int expectedSum = a + b;

    std::cout << "add two integers using CSR for input." << std::endl;

    q.single_task<Add>(Add_Kernel{a, b}).wait();

    // verify that outputs are correct
    bool passed = true;

    std::cout << "collect results." << std::endl;
    int calc_add = OutputPipe::read(q);

    std::cout << "Add sum: " << calc_add << ", expected (" << expectedSum << ")" << std::endl;
    if (calc_add != expectedSum)
    {
        passed = false;
    }

    std::cout << "add two integers using CSR->pipes for inputs." << std::endl;

    // push data into pipes
    InputPipeA::write(q, a);
    InputPipeB::write(q, b);

    q.single_task<AddCSRPipes>(AddCSRPipes_Kernel{}).wait();

    std::cout << "collect results." << std::endl;
    int calc_addCSRPipes = OutputPipeC::read(q);

    std::cout << "AddCSR sum =" << calc_addCSRPipes << ", expected (" << expectedSum << ")" << std::endl;
    if (calc_addCSRPipes != expectedSum)
    {
        passed = false;
    }
    }  catch (sycl::exception const &e) {
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

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
