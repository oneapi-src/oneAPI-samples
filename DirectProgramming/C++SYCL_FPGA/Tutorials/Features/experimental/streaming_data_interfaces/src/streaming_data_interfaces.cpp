#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

#define THRESHOLD 200

// Forward declare the kernel and pipe names
// (this prevents unwanted name mangling in the optimization report)
class InPixel;
class OutPixel;
class Threshold;

// StreamingBeat struct to support Avalon packet sideband signals
using StreamingBeatT = sycl::ext::intel::experimental::StreamingBeat<
    unsigned short, true, false>;

// A kernel that thresholds pixel values in an image over a stream. Uses start
// of packet and end of packet signals on the streams to determine the beginning
// and end of the image.
template <typename InPixelPipe, typename OutPixelPipe>
struct ThresholdKernel {

  void operator()() const {
    bool start_of_packet = false;
    bool end_of_packet = false;

    while (!end_of_packet) {

      // Read in next pixel
      StreamingBeatT in_beat = InPixelPipe::read();
      auto pixel = in_beat.data;
      start_of_packet = in_beat.sop;
      end_of_packet = in_beat.eop;

      // Threshold
      if (pixel > THRESHOLD) pixel = THRESHOLD;

      // Write out result
      StreamingBeatT out_beat(pixel, start_of_packet, end_of_packet);
      OutPixelPipe::write(out_beat);

    }
  }
};

int main() {

  try {

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
  sycl::queue q(selector, fpga_tools::exception_handler);

  auto device = q.get_device();
  std::cout << "Running on device: "
            << device.get_info<sycl::info::device::name>().c_str()
            << std::endl;

  // Test image dimensions1
  unsigned int width  = 16;
  unsigned int height = 16;

  // Pipe properties
  using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<16>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready));

  // Image streams
  using InPixelPipe = sycl::ext::intel::experimental::pipe<
      InPixel,        // An identifier for the pipe
      StreamingBeatT, // The type of data in the pipe
      0,              // The capacity of the pipe
      PipePropertiesT // Customizable pipe properties
      >;
  using OutPixelPipe = sycl::ext::intel::experimental::pipe<
      OutPixel,       // An identifier for the pipe
      StreamingBeatT, // The type of data in the pipe
      0,              // The capacity of the pipe
      PipePropertiesT // Customizable pipe properties
      >;

  // Generate pixel data
  for (int i = 0; i < (width * height); ++i) {
    bool start_of_packet = (i == 0);
    bool end_of_packet = (i == ((width * height) - 1));
    StreamingBeatT in_beat(i, start_of_packet, end_of_packet);
    InPixelPipe::write(q, in_beat);
  }

  // Call the kernel
  q.single_task<Threshold>(ThresholdKernel<InPixelPipe, OutPixelPipe>{});

  // Check that output pixels are below the threshold
  bool passed = true;
  for (int i = 0; i < (width * height); ++i) {
    StreamingBeatT out_beat = OutPixelPipe::read(q);
    passed &= (out_beat.data <= THRESHOLD);
  }

  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }
}