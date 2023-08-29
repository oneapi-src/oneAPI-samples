#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

static constexpr int kBL0 = 0;
static constexpr int kBL1 = 1;

struct SimpleDMA {
  // declaring the pointers first since they are 64-bit, I want the wide types
  // first so that the CSRs will be aligned to them
  conduit_mmhost(
      kBL0,  // buffer space, this needs to be unique per master (don't forget
             // to encode this value into pointer bits 46:41 otherwise the
             // decoding might not work as expected)
      32,    // address width
      32,  // data width, typically this would be wider but the memory connected
           // is only 32-bit
      0,   // latency (choose 0-latency so that waitrequest will work)
      1,   // 1 = read only
      15,  // max burst size
      0,   // aligned transactions only
      1    // include waitrequest
      ) unsigned int* const source;

  using params1 =
      decltype(sycl::ext::intel::experimental::properties{
               sycl::ext::intel::experimental::buffer_location<kBL1>,
               sycl::ext::intel::experimental::dwidth<32>,
               sycl::ext::intel::experimental::awidth<32>,
               sycl::ext::intel::experimental::latency<0>,
               sycl::ext::intel::experimental::});

  sycl::ext::oneapi::experimental::annotated_arg<unsigned int* const, params1>
      source;

  conduit_mmhost(
      kBL1,  // buffer space, this needs to be unique per master (don't forget
             // to encode this value into pointer bits 46:41 otherwise the
             // decoding might not work as expected)
      32,  // address width
      32,  // data width, typically this would be wider but the memory connected
           // is only 32-bit
      0,   // latency (choose 0-latency so that waitrequest will work)
      2,   // 2 = write only
      15,  // max burst size
      0,   // aligned transactions only
      1    // include waitrequest
      ) unsigned int* const destination;

  unsigned int length;  // measured in bytes, must be a multiple of 4

  // Implementation of the DMA kernel.  This accelerator will be controlled via
  // its Avalon-MM agent interface
  streaming_pipelined_interface void operator()() const {
    // This loop does not handle partial accesses (less than 4 bytes) at the
    // start and end of the source/destination so ensure they are at least
    // 4-byte aligned
    for (unsigned int i = 0; i < (length / 4);
         i++)  // will be testing with Nios V where an int is 4 bytes
    {
      destination[i] = source[i];
    }
  }
};

constexpr int LEN = 128;  // bytes

int main() {
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

  // create the device queue
  sycl::queue q(selector);

  // make sure the device supports USM host allocations
  auto device = q.get_device();

  std::cout << "Running on device: "
            << device.get_info<sycl::info::device::name>().c_str() << std::endl;

  // define with old property_list syntax until mm_host gets support for the new
  // properties syntax
  unsigned int* src = sycl::malloc_shared<unsigned int>(
      LEN, q, property_list{buffer_location(kBL0)});
  unsigned int* dest = sycl::malloc_shared<unsigned int>(
      LEN, q, property_list{buffer_location(kBL1)});
  unsigned int len = LEN;

  // pre-load
  for (int i = 0; i < len; i++) {
    src[i] = len - i;
    dest[i] = 0;
  }

  // line below is what associates the name "SimpleDMA" to the kernel
  q.single_task(SimpleDMA{src, dest, len}).wait();

  // check results
  bool passed = true;
  for (int i = 0; i < (len / 4); i++) {
    bool ok = (src[i] == dest[i]);
    passed &= ok;

    if (!passed) {
      std::cerr << "ERROR: [" << i << "] expected " << src[i] << " saw "
                << dest[i] << ". " << std::endl;
    }
  }

  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
  sycl::free(src, q);
  sycl::free(dest, q);
}
