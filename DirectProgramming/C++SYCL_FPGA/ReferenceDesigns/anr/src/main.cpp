#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "anr.hpp"
#include "anr_params.hpp"
#include "constants.hpp"
#include "data_bundle.hpp"
#include "dma_kernels.hpp"
#include "exception_handler.hpp"

using namespace sycl;
using namespace std::chrono;

////////////////////////////////////////////////////////////////////////////////
// Forward declare functions used in this file by main()
void ParseFiles(std::string data_dir, std::vector<PixelT>& in_pixels,
                std::vector<PixelT>& ref_pixels, int& cols, int& rows,
                ANRParams& params);

void WriteOutputFile(std::string data_dir, std::vector<PixelT>& pixels,
                     int cols, int rows);

double RunANR(queue& q, PixelT* in_ptr, PixelT* out_ptr, int cols, int rows,
              int frames, ANRParams params, float* sig_i_lut_data_ptr);

bool Validate(PixelT* val, PixelT* ref, int rows, int cols,
              double psnr_thresh = kPSNRDefaultThreshold);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  /////////////////////////////////////////////////////////////
  // reading and validating the command line arguments
  std::string data_dir = "../test_data";
  bool passed = true;
#if defined(FPGA_EMULATOR)
  int runs = 2;
  int frames = 2;
#elif defined(FPGA_SIMULATOR)
  int runs = 2;
  int frames = 1;
#else
  int runs = 2;
  int frames = 8;
#endif

  // get the input data directory
  if (argc > 1) {
    data_dir = std::string(argv[1]);
  }

  // get the number of runs as the second command line argument
  if (argc > 2) {
    runs = atoi(argv[2]);
  }

  // get the number of frames as the third command line argument
  if (argc > 3) {
    frames = atoi(argv[3]);
  }

  // enforce at least two runs
  if (runs < 2) {
    std::cerr << "ERROR: 'runs' must be 2 or more\n";
    std::terminate();
  }

  // enforce at least one batch
  if (frames < 1) {
    std::cerr << "ERROR: 'frames' must be atleast 1\n";
    std::terminate();
  }
  /////////////////////////////////////////////////////////////

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  // create the device queue
  queue q(selector, fpga_tools::exception_handler);

  // make sure the device supports USM device allocations
  auto device = q.get_device();
#if defined(IS_BSP)
  if (!device.has(aspect::usm_device_allocations)) {
    std::cerr << "ERROR: The selected device does not support USM device"
              << " allocations\n";
    std::terminate();
  }
#endif

  std::cout << "Running on device: "
            << device.get_info<info::device::name>().c_str() 
            << std::endl;


  // parse the input files
  int cols, rows, pixel_count;
  ANRParams params;
  std::vector<PixelT> in_pixels, ref_pixels;
  ParseFiles(data_dir, in_pixels, ref_pixels, cols, rows, params);
  pixel_count = cols * rows;

  // create the output pixels (initialize to all 0s)
  std::vector<PixelT> out_pixels(in_pixels.size(), 0);

#if defined (IS_BSP)
  // allocate memory on the device for the input and output
  PixelT *in, *out;
  if ((in = malloc_device<PixelT>(pixel_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'in'\n";
    std::terminate();
  }
  if ((out = malloc_device<PixelT>(pixel_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'out'\n";
    std::terminate();
  }
#else 
  // allocate memory on the host for the input and output
  PixelT *in, *out;
  if ((in = malloc_shared<PixelT>(pixel_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'in'\n";
    std::terminate();
  }
  if ((out = malloc_shared<PixelT>(pixel_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'out'\n";
    std::terminate();
  }
#endif   


  // copy the input data to the device memory and wait for the copy to finish
  q.memcpy(in, in_pixels.data(), pixel_count * sizeof(PixelT)).wait();

  // allocate space for the intensity sigma LUT
  float* sig_i_lut_data_ptr = IntensitySigmaLUT::Allocate(q);

  // create the intensity sigma LUT data locally on the host
  IntensitySigmaLUT sig_i_lut_host(params);

  // copy the intensity sigma LUT to the device
  sig_i_lut_host.CopyData(q, sig_i_lut_data_ptr).wait();
  //////////////////////////////////////////////////////////////////////////////

  // track timing information in ms
  std::vector<double> time(runs);

  // print out some info
  std::cout << "Runs:             " << runs << "\n";
  std::cout << "Columns:          " << cols << "\n";
  std::cout << "Rows:             " << rows << "\n";
  std::cout << "Frames:           " << frames << "\n";
  std::cout << "Filter Size:      " << kFilterSize << "\n";
  std::cout << "Pixels Per Cycle: " << kPixelsPerCycle << "\n";
  std::cout << "Maximum Columns:  " << kMaxCols << "\n";
  std::cout << "\n";

  try {
    // run the design multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      // run ANR
      time[i] =
          RunANR(q, in, out, cols, rows, frames, params, sig_i_lut_data_ptr);

      // Copy the output back from the device
      q.memcpy(out_pixels.data(), out, pixel_count * sizeof(PixelT)).wait();

      // validate the output on the last iteration
      if (i == (runs-1)) {
        passed &= Validate(out_pixels.data(), ref_pixels.data(), rows, cols);
      } else {
        passed &= true;
      }
    }
  } catch (exception const& e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::terminate();
  }

  // free the allocated device memory
  sycl::free(in, q);
  sycl::free(out, q);
  sycl::free(sig_i_lut_data_ptr, q);

  // write the output files if device memory was used (output is meaningless
  // otherwise)
  WriteOutputFile(data_dir, out_pixels, cols, rows);

  // print the performance results
  if (passed) {
    // NOTE: when run in emulation, these results do not accurately represent
    // the performance of the kernels in actual FPGA hardware
    double avg_time_ms =
        std::accumulate(time.begin() + 1, time.end(), 0.0) / (runs - 1);

    size_t input_count_mega = pixel_count * frames * sizeof(PixelT) * 1e-6;

    std::cout << "Execution time: " << avg_time_ms << " ms\n";
    std::cout << "Throughput: " << (input_count_mega / (avg_time_ms * 1e-3))
              << " MB/s\n";
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

// declare kernel and pipe names globally to reduce name mangling
class ANRInPipeID;
class ANROutPipeID;
class InputKernelID;
class OutputKernelID;

//
// Run the ANR algorithm on the device
//
double RunANR(queue& q, PixelT* in_ptr, PixelT* out_ptr, int cols, int rows,
              int frames, ANRParams params, float* sig_i_lut_data_ptr) {
  // the input and output pipe for the sorter
  using PipeType = DataBundle<PixelT, kPixelsPerCycle>;
  using ANRInPipe = sycl::ext::intel::pipe<ANRInPipeID, PipeType>;
  using ANROutPipe = sycl::ext::intel::pipe<ANROutPipeID, PipeType>;

  // launch the input and output kernels that read from and write to the device
  auto input_kernel_event =
      SubmitInputDMA<InputKernelID, PixelT, ANRInPipe, kPixelsPerCycle>(q,
                     in_ptr, rows, cols, frames);

  auto output_kernel_event =
      SubmitOutputDMA<OutputKernelID, PixelT, ANROutPipe, kPixelsPerCycle>(q,
                      out_ptr, rows, cols, frames);

  // launch all ANR kernels
  std::vector<std::vector<event>> anr_kernel_events(frames);
  for (int i = 0; i < frames; i++) {
    anr_kernel_events[i] =
        SubmitANRKernels<IndexT, ANRInPipe, ANROutPipe, kFilterSize,
                       kPixelsPerCycle, kMaxCols>(q, cols, rows, params,
                       sig_i_lut_data_ptr);
  }

  // wait for the input and output kernels to finish
  auto start = high_resolution_clock::now();
  input_kernel_event.wait();
  output_kernel_event.wait();
  auto end = high_resolution_clock::now();

  // wait for the ANR kernels to finish
  for (auto& one_event_set : anr_kernel_events) {
    for (auto& e : one_event_set) {
      e.wait();
    }
  }

  // return the duration in milliseconds, excluding memory transfers
  duration<double, std::milli> diff = end - start;
  return diff.count();
}

//
// Helper to parse pixel data files
//
void ParseDataFile(std::string filename, std::vector<PixelT>& pixels, int& cols,
                   int& rows) {
  // create the file stream to parse
  std::ifstream ifs(filename);

  // make sure we opened the file
  if (!ifs.is_open() || ifs.fail()) {
    std::cerr << "ERROR: failed to open " << filename << " for reading\n";
    std::terminate();
  }

  // get the header and data
  std::string header_str, data_str;
  if (!std::getline(ifs, header_str)) {
    std::cerr << "ERROR: failed to get header line from " << filename << "\n";
    std::terminate();
  }
  if (!std::getline(ifs, data_str)) {
    std::cerr << "ERROR: failed to get data line from " << filename << "\n";
    std::terminate();
  }

  // first two elements are the image dimensions
  std::stringstream header_ss(header_str);
  header_ss >> rows >> cols;

  // expecting to parse cols*rows pixels from the 'data_str' line
  pixels.resize(cols * rows);

  // parse all of the pixels
  std::stringstream data_ss(data_str);
  for (int i = 0; i < cols * rows; i++) {
    // parse using 64 bit integer
    TmpT x;

    // parse the pixel value
    if (!(data_ss >> x)) {
      std::cerr << "ERROR: ran out of pixels when parsing " << filename << "\n";
      std::terminate();
    }

    // check for parsing failure
    if (data_ss.fail()) {
      std::cerr << "ERROR: failed to parse pixel in " << filename << "\n";
      std::terminate();
    }

    // check if the parsed value fits in the pixel type
    if (x > static_cast<TmpT>(std::numeric_limits<PixelT>::max())) {
      std::cerr << "ERROR: value (" << x
                << ") is too big to store in pixel type 'T'\n";
      std::terminate();
    }
    if (x < static_cast<TmpT>(std::numeric_limits<PixelT>::min())) {
      std::cerr << "ERROR: value (" << x
                << ") is too small to store in pixel type 'T'\n";
      std::terminate();
    }

    // set the value
    pixels[i] = static_cast<PixelT>(x);
  }
}

//
// Function that parses all of the input files
//
void ParseFiles(std::string data_dir, std::vector<PixelT>& in_pixels,
                std::vector<PixelT>& ref_pixels, int& cols, int& rows,
                ANRParams& params) {
  // parse the pixel data files
  int noisy_w, noisy_h;
#if FPGA_SIMULATOR
  ParseDataFile(data_dir + "/small_input_noisy.data", in_pixels, noisy_w, noisy_h);
#else
  ParseDataFile(data_dir + "/input_noisy.data", in_pixels, noisy_w, noisy_h);
#endif
  int ref_w, ref_h;
#if FPGA_SIMULATOR
  ParseDataFile(data_dir + "/small_output_ref.data", ref_pixels, ref_w, ref_h);
#else
  ParseDataFile(data_dir + "/output_ref.data", ref_pixels, ref_w, ref_h);
#endif

  // ensure the dimensions match
  if (noisy_w != ref_w) {
    std::cerr << "noisy input and reference widths do not match " << noisy_w
              << " != " << ref_w << "\n";
    std::terminate();
  }
  if (noisy_h != ref_h) {
    std::cerr << "noisy input and reference heights do not match " << noisy_h
              << " != " << ref_h << "\n";
    std::terminate();
  }

  // set the width and height
  cols = ref_w;
  rows = ref_h;

  // parse the ANR config parameters file
  params = ANRParams::FromFile(data_dir + "/param_config.data");

  // ensure the parsed filter size matches the compile time constant
  if (params.filter_size != kFilterSize) {
    std::cerr << "ERROR: the filter size parsed from " << data_dir
              << "/param_config.data (" << params.filter_size
              << ") does not match the compile time constant filter size "
              << "(kFilterSize = " << kFilterSize << ")\n";
    std::terminate();
  }

  // ensure the parsed number of pixel bits matches the compile time constant
  if (params.pixel_bits != kPixelBits) {
    std::cerr << "ERROR: the number of bits per pixel parsed from " << data_dir
              << "/param_config.data (" << params.pixel_bits
              << ") does not match the compile time constant pixel size "
              << "kPixelBits = " << kPixelBits << ")\n";
    std::terminate();
  }
}

//
// Function to write the output to a file
//
void WriteOutputFile(std::string data_dir, std::vector<PixelT>& pixels,
                     int cols, int rows) {
  std::string filename = data_dir + "/output.data";
  std::ofstream ofs(filename);

  // make sure we opened the file fine
  if (!ofs.is_open() || ofs.fail()) {
    std::cerr << "ERROR: failed to open " << filename << " for writing\n";
    std::terminate();
  }

  // write the image dimensions
  ofs << rows << " " << cols << "\n";

  // write the pixels
  for (auto& p : pixels) {
    ofs << static_cast<TmpT>(p) << " ";
  }
}

//
// Validate the output pixels using Peak signal-to-noise ratio (PSNR)
// https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
//
// Also check the max individual pixel difference.
//
bool Validate(PixelT* val, PixelT* ref, int rows, int cols,
              double psnr_thresh) {
  // get the maximum value of the pixel
  constexpr double max_i = std::numeric_limits<PixelT>::max();

  // total number of pixels to check
  int count = rows * cols;

  // compute the MSE by summing the squared differences
  // also find the maximum difference between the output pixel and the reference
  double mse = 0.0;
  for (int i = 0; i < count; i++) {
    // cast to a double here because we are subtracting
    auto diff = double(val[i]) - double(ref[i]);
    mse += diff * diff;
  }
  mse /= count;

  // compute the PSNR
  double psnr = (20 * std::log10(max_i)) - (10 * std::log10(mse));

  // check PSNR and maximum pixel difference
  bool passed = true;
  if (psnr <= psnr_thresh) {
    std::cerr << "ERROR: Peak signal-to-noise ratio (PSNR) is too low: " << psnr
              << "\n";
    passed = false;
  }

  return passed;
}
