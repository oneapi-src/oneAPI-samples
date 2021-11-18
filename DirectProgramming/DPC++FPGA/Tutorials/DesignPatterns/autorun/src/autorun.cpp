#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "autorun.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace sycl;

// choose the device selector based on emulation or actual hardware
// we make this a global variable so it can be used by the autorun kernels
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector ds;
#else
  ext::intel::fpga_selector ds;
#endif

// declare the kernel names globally to reduce name mangling
class ARProducerID;
class ARKernelID;
class ARConsumerID;
class ARForeverProducerID;
class ARForeverKernelID;
class ARForeverConsumerID;

// declare the pipe names globally to reduce name mangling
class ARProducePipeID;
class ARConsumePipeID;
class ARForeverProducePipeID;
class ARForeverConsumePipeID;

// pipes
using ARProducePipe = ext::intel::pipe<ARProducePipeID, int>;
using ARConsumePipe = ext::intel::pipe<ARConsumePipeID, int>;
using ARForeverProducePipe = ext::intel::pipe<ARForeverProducePipeID, int>;   
using ARForeverConsumePipe = ext::intel::pipe<ARForeverConsumePipeID, int>;

////////////////////////////////////////////////////////////////////////////////
// Autorun user kernel and global variable
struct MyAutorun {
  void operator()() const {
    // notice that in this version, we explicitly add the while(1)-loop
    while (1) {
      auto d = ARProducePipe::read();
      ARConsumePipe::write(d);
    }
  }
};

// declaring a global instance of this class causes the constructor to be called
// before main() starts, and the constructor launches the kernel.
fpga_tools::Autorun<ARKernelID> ar_kernel{ds, MyAutorun{}};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// AutorunForever user kernel and global variable
// The AutorunForever kernel implicitly wraps the code below in a while(1) loop
struct MyAutorunForever {
  void operator()() const {
    // this code is implicitly placed in a while(1)-loop by the 
    // fpga_tools::AutorunForever class
    auto d = ARForeverProducePipe::read();
    ARForeverConsumePipe::write(d);
  }
};

// declaring a global instance of this class causes the constructor to be called
// before main() starts, and the constructor launches the kernel.
fpga_tools::AutorunForever<ARForeverKernelID> ar_forever_kernel{ds,
    MyAutorunForever{}};
////////////////////////////////////////////////////////////////////////////////

//
// Submit a kernel to read data from global memory and write to a pipe
//
template<typename KernelID, typename Pipe>
event SubmitProducerKernel(queue& q, buffer<int, 1>& in_buf) {
  return q.submit([&](handler &h) {
    accessor in(in_buf, h, read_only);
    int size = in_buf.size();
    h.single_task<KernelID>([=] {
      for (int i = 0; i < size; i++) {
        Pipe::write(in[i]);
      }
    });
  });
}

//
// Submit a kernel to read data from a pipe and write to global memory
//
template<typename KernelID, typename Pipe>
event SubmitConsumerKernel(queue& q, buffer<int, 1>& out_buf) {
  return q.submit([&](handler &h) {
    accessor out(out_buf, h, write_only, no_init);
    int size = out_buf.size();
    h.single_task<KernelID>([=] {
      for (int i = 0; i < size; i++) {
        out[i] = Pipe::read();
      }
    });
  });
}

int main() {
  int count = 5000;
  bool passed = true;

  std::vector<int> in_data(count), out_data(count);

  // populate random input data, clear output data
  std::generate(in_data.begin(), in_data.end(), [] { return rand() % 100; } );
  std::fill(out_data.begin(), out_data.end(), -1);

  try {
    // create the queue
    queue q(ds, dpc_common::exception_handler);

    // stream data through the Autorun kernel
    std::cout << "Running the Autorun kernel test\n";
    {
      // Create input and output buffers
      buffer in_buf(in_data);
      buffer out_buf(out_data);
      SubmitProducerKernel<ARProducerID, ARProducePipe>(q, in_buf);
      SubmitConsumerKernel<ARConsumerID, ARConsumePipe>(q, out_buf);
    }

    // validate the results
    // operator== for a vector checks sizes, then checks per-element
    passed &= (out_data == in_data);

    // stream data through the AutorunForever kernel
    std::cout << "Running the AutorunForever kernel test\n";
    {
      // Create input and output buffers
      buffer in_buf(in_data);
      buffer out_buf(out_data);
      SubmitProducerKernel<ARForeverProducerID, ARForeverProducePipe>(q,
          in_buf);
      SubmitConsumerKernel<ARForeverConsumerID, ARForeverConsumePipe>(q,
          out_buf);
    }

    // validate the results
    // operator== for a vector checks sizes, then checks per-element
    passed &= (out_data == in_data);
  } catch (sycl::exception const& e) {
    // Catches exceptions in the host code
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

  if (passed) {
    std::cout << "PASS\n";
    return 0;
  } else {
    std::cout << "FAIL\n";
    return 1;
  }
}
