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

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

#include "anr.hpp"
#include "anr_params.hpp"
#include "data_bundle.hpp"
#include "dma_kernels.hpp"
#include "mp_math.hpp"

using namespace sycl;
using namespace std::chrono;

// the user can disable global memory by defining the 'DISABLE_GLOBAL_MEM'
// macro: -DDISABLE_GLOBAL_MEM
#if defined(DISABLE_GLOBAL_MEM)
constexpr bool kDisableGlobalMem = true;
#else
constexpr bool kDisableGlobalMem = false;
#endif

// The size of the filter can be changed at the command line
#ifndef FILTER_SIZE
#define FILTER_SIZE 9
#endif
constexpr unsigned kFilterSize = FILTER_SIZE;
static_assert(kFilterSize > 1);

// The number of pixels per cycle
#ifndef PIXELS_PER_CYCLE
#define PIXELS_PER_CYCLE 1
#endif
constexpr unsigned kPixelsPerCycle = PIXELS_PER_CYCLE;
static_assert(kPixelsPerCycle > 0);
static_assert(IsPow2(kPixelsPerCycle) > 0);

// The maximum number of columns in the image
#ifndef MAX_COLS
//#define MAX_COLS 1920 // HD
#define MAX_COLS 3840 // 4K
#endif
constexpr unsigned kMaxCols = MAX_COLS;
static_assert(kMaxCols > 0);
static_assert(kMaxCols > kPixelsPerCycle);

// the type to use for the pixel intensity values and a temporary type
// which should have more bits than the pixel type to check for overflow.
// We will use subtraction on the temporary type, so it must be signed.
using PixelT = unsigned char; // 8 bits, unsigned
using TmpT = long long; // 64 bits, signed
static_assert(std::is_unsigned_v<PixelT>);
static_assert(std::is_signed_v<TmpT>);
static_assert(sizeof(TmpT) > sizeof(PixelT));

// the type used for indexing the rows and columns of the image
using IndexT = short;
static_assert(std::is_integral_v<IndexT>);
static_assert(!std::is_unsigned_v<IndexT>);

////////////////////////////////////////////////////////////////////////////////
// Forward declare functions used in this file by main()
void ParseFiles(std::string data_dir, std::vector<PixelT>& in_pixels,
                std::vector<PixelT>& ref_pixels, int& cols, int& rows,
                ANRParams& params);

void WriteOutputFile(std::string data_dir, std::vector<PixelT>& pixels,
                     int cols, int rows);

double RunANR(queue &q, PixelT *in_ptr, PixelT *out_ptr,
              ANRParams params, int cols, int rows, int frames);

bool Validate(PixelT *val, PixelT *ref, unsigned int count, double thresh=0.0001);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  /////////////////////////////////////////////////////////////
  // reading and validating the command line arguments
  // defaults
  std::string data_dir = "../test_data";
  bool passed = true;
#ifdef FPGA_EMULATOR
  int runs = 2;
  int frames = 2;
#else
  int runs = 9;
  int frames = 8;
#endif

  // get the input directory
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

  // the device selector
#ifdef FPGA_EMULATOR
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  // create the device queue
  queue q(selector, dpc_common::exception_handler);

  // make sure the device supports USM device allocations
  auto d = q.get_device();
  if (!d.get_info<info::device::usm_device_allocations>()) {
    std::cerr << "ERROR: The selected device does not support USM device"
              << " allocations\n";
    std::terminate();
  }

  // parse the input files
  int cols, rows, pixel_count;
  ANRParams params;
  std::vector<PixelT> in_pixels, ref_pixels;
  ParseFiles(data_dir, in_pixels, ref_pixels, cols, rows, params);
  pixel_count = cols * rows;

  //////////////////////////////////////////////////////////////////////////////
  // DEBUG
  /*
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%d ", in_pixels[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
  */
  /*
  //std::array<PixelT, kFilterSize> test_in = {127, 0, 199, 0, 231, 0, 185, 0, 221};
  //PixelT ref = 211, res = 0;
  std::array<PixelT, kFilterSize> test_in = {0, 0, 0, 0, 211, 0, 164, 0, 255};
  PixelT ref = 210, res = 0;
  if (!TestBilateralFilter<PixelT, kFilterSize>(test_in, params, ref, res)) {
    std::cerr << "ERROR: TestBilateralFilter failed ("
              << (TmpT)res << " != " << (TmpT)ref << ")\n";
    std::terminate();
  }
  */
  //////////////////////////////////////////////////////////////////////////////

  // create the output (initialize to all 0s)
  std::vector<PixelT> out_pixels(in_pixels.size(), 0);

  // allocate memory on the device for the input and output
  PixelT *in, *out;
  if ((in = malloc_device<PixelT>(pixel_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'in' using "
              << "malloc_device\n";
    std::terminate();
  }
  if ((out = malloc_device<PixelT>(pixel_count, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'out' using "
              << "malloc_device\n";
    std::terminate();
  }

  // copy the input data to the device memory and wait for the copy to finish
  q.memcpy(in, in_pixels.data(), pixel_count * sizeof(PixelT)).wait();

  // track timing information, in ms
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
      time[i] = RunANR(q, in, out, params, cols, rows, frames);

      // Copy the output to 'out_vec'. In the case where we are using USM host
      // allocations this is unnecessary since we could simply deference
      // 'out'. However, it makes the following code cleaner since the output
      // is always in 'out_vec' and this copy is not part of the performance
      // timing.
      q.memcpy(out_pixels.data(), out, pixel_count * sizeof(PixelT)).wait();

      // validate the output
      if constexpr (!kDisableGlobalMem) {
        passed &= Validate(out_pixels.data(), ref_pixels.data(), pixel_count);
        //passed &= Validate(out_pixels.data(), in_pixels.data(), pixel_count);
      } else {
        passed = true;
      }
    }
  } catch (exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::terminate();
  }

  // free the memory allocated with malloc_host or malloc_device
  sycl::free(in, q);
  sycl::free(out, q);

  // write the output files if global memory was used (output is meaningless,
  // otherwise)
  if constexpr (kDisableGlobalMem) {
    WriteOutputFile(data_dir, out_pixels, cols, rows);
  }

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
// Run the ANR algorithm on the FPGA
//
double RunANR(queue &q, PixelT *in_ptr, PixelT *out_ptr,
              ANRParams params, int cols, int rows, int frames) {
  // the input and output pipe for the sorter
  using PipeType = DataBundle<PixelT, kPixelsPerCycle>;
  using ANRInPipe = sycl::INTEL::pipe<ANRInPipeID, PipeType>;
  using ANROutPipe = sycl::INTEL::pipe<ANROutPipeID, PipeType>;

  // launch the input and output kernels that read and write from device memory
  auto input_kernel_event =
    SubmitInputDMA<InputKernelID, PixelT, ANRInPipe,
                   kPixelsPerCycle, kDisableGlobalMem>(q, in_ptr, rows, cols, frames);

  auto output_kernel_event =
    SubmitOutputDMA<OutputKernelID, PixelT, ANROutPipe,
                   kPixelsPerCycle, kDisableGlobalMem>(q, out_ptr, rows, cols, frames);

  // launch ANR kernel
  auto anr_kernel_events =
    SubmitANRKernels<PixelT, IndexT, ANRInPipe, ANROutPipe, kFilterSize, kPixelsPerCycle>(q, params,
                                                                                          cols, rows,
                                                                                          frames);

  // wait for the input and output kernels to finish
  auto start = high_resolution_clock::now();
  input_kernel_event.wait();
  output_kernel_event.wait();
  auto end = high_resolution_clock::now();

  // wait for the ANR kernels to finish
  for (auto &e : anr_kernel_events) {
    e.wait();
  }

  // return the duration in milliseconds, excluding memory transfers
  duration<double, std::milli> diff = end - start;
  return diff.count();
}

//
// Helper to parse data files
//
void ParseDataFile(std::string filename, std::vector<PixelT>& pixels,
                   int& cols, int& rows) {
  // create the file stream to parse
  std::ifstream is(filename);

  // get header and data
  std::string header_str, data_str;
  if (!std::getline(is, header_str)) {
    std::cerr << "ERROR: failed to get header line from " << filename << "\n";
    std::terminate();
  }
  if (!std::getline(is, data_str)) {
    std::cerr << "ERROR: failed to get data line from " << filename << "\n";
    std::terminate();
  }

  // first two elements are the width and height
  std::stringstream header_ss(header_str);
  header_ss >> cols >> rows;

  // expecting to parse cols*rows pixels
  pixels.resize(cols*rows);

  // parse all of the pixels
  std::stringstream data_ss(data_str);
  for (int i = 0; i < cols*rows; i++) {
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

    // check if the parsed value fits in given type 'PixelT'
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
// Function that parses an input file and returns the result
//
void ParseFiles(std::string data_dir, std::vector<PixelT>& in_pixels,
                std::vector<PixelT>& ref_pixels, int& cols, int& rows,
                ANRParams& params) {
  // parse the files
  int noisy_w, noisy_h;
  ParseDataFile(data_dir + "/input_noisy.data", in_pixels, noisy_w, noisy_h);
  int ref_w, ref_h;
  ParseDataFile(data_dir + "/output_ref.data", ref_pixels, ref_w, ref_h);

  // ensure dimensions match
  if (noisy_w != ref_w) {
    std::cerr << "noisy input and reference widths do not match "
              << noisy_w << " != " << ref_w << "\n";
    std::terminate();
  }
  if (noisy_h != ref_h) {
    std::cerr << "noisy input and reference heights do not match "
              << noisy_h << " != " << ref_h << "\n";
    std::terminate();
  }

  // set width and height
  cols = ref_w;
  rows = ref_h;

  // parse config parameters file
  params = ANRParams::FromFile(data_dir + "/param_config.data");

  // ensure the parsed filter size matches the compile time constant
  if (params.filter_size != kFilterSize) {
    std::cerr << "ERROR: the filter size parsed from " << data_dir
              << "/param_config.data (" << params.filter_size
              << ") does not match the compile time constant filter size "
              << "(kFilterSize = " << kFilterSize << ")\n";
    std::terminate();
  }
}

//
// Function to write the output to a file
//
void WriteOutputFile(std::string data_dir, std::vector<PixelT>& pixels,
                     int cols, int rows) {
  std::ofstream of(data_dir + "/output.data");

  // write the size
  of << cols << " " << rows << "\n";

  // write the pixels 
  for (auto& p : pixels) {
    of << static_cast<TmpT>(p) << " ";
  }
}

//
// Validate the output pixels using normalized-root-mean-squared-error (NRMSE)
// https://en.wikipedia.org/wiki/Root-mean-square_deviation
//
// Acronyms
//    MSE     = mean-square-error
//    RMSE    = root-mean-square-error
//    NRMSE   = normalized-root-mean-square-error
//
bool Validate(PixelT *val, PixelT *ref, unsigned int count, double thresh) {
  // compute the MSE by summing the squared differences
  double nrmse = 0.0;
  for (unsigned int i = 0; i < count; i++) {
    // cast to a double here because we are subtracting
    auto diff = double(val[i]) - double(ref[i]);
    if(diff != 0) nrmse += diff * diff;
  }

  // compute the RMSE
  nrmse = std::sqrt(nrmse / count);

  // normalize by the range of the output to get the NRMSE
  nrmse /=
    (std::numeric_limits<PixelT>::max() - std::numeric_limits<PixelT>::min());

  // check MSE
  if (nrmse >= thresh) {
    std::cerr << "ERROR: Normalized root-mean-squared-error is too high: "
              << nrmse << "\n";
    return false;
  } else {
    //std::cout << "NRMSE = " << nrmse << "\n";
    return true;
  }
}
