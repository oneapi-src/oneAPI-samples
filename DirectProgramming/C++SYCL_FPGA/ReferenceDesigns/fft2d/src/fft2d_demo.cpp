#include <math.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "fft2d.hpp"

// Forward declarations
void TestFFT(bool mangle, bool inverse);
template <int n>
int Coordinates(int iteration, int i);
template <int lognr_points>
void FourierTransformGold(ac_complex<double> *data, bool inverse);
template <int lognr_points>
void FourierStage(ac_complex<double> *data);

int main(int argc, char **argv) {
  if (argc == 1) {
    std::cout << "No program argument was passed, running all fft2d variants"
              << std::endl;

    // test FFT transform with ordered memory layout
    TestFFT(false, false);
    // test inverse FFT transform with ordered memory layout
    TestFFT(false, true);
    // test FFT transform with alternative memory layout
    TestFFT(true, false);
    // test inverse FFT transform with alternative memory layout
    TestFFT(true, true);

  } else {
    std::string mode = argv[1];

    bool mangle{};
    bool inverse{};

    if (mode == "normal") {
      mangle = false;
      inverse = false;
    } else if (mode == "inverse") {
      mangle = false;
      inverse = true;
    } else if (mode == "mangle") {
      mangle = true;
      inverse = false;
    } else if (mode == "inverse-mangle") {
      mangle = true;
      inverse = true;
    } else {
      std::cerr << "Usage: fft2d <mode>" << std::endl;
      std::cerr << "Where mode can be normal|inverse|mangle|inverse-mangle|all"
                << std::endl;
      std::terminate();
    }

    TestFFT(mangle, inverse);
  }
  return 0;
}

void TestFFT(bool mangle, bool inverse) {
  try {
    // Device selector selection
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // Enable the queue profiling to time the execution
    sycl::property_list queue_properties{
        sycl::property::queue::enable_profiling()};
    sycl::queue q =
        sycl::queue(selector, fpga_tools::exception_handler, queue_properties);

    sycl::device device = q.get_device();

    // Print out the device information.
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>() << std::endl;

    // Define the log of the FFT size on each dimension and the level of
    // parallelism to implement
#if FPGA_SIMULATOR
    // Force small sizes in simulation mode to reduce simulation time
    constexpr int kLogN = 4;
    constexpr int kParallelism = 4;
#else
    constexpr int kLogN = LOGN;
    constexpr int kParallelism = PARALLELISM;
#endif

    static_assert(kParallelism == 4 || kParallelism == 8,
                  "The FFT kernel implementation only supports 4-parallel and "
                  "8-parallel FFTs.");

    constexpr int kN = 1 << kLogN;
    constexpr int kLogParallelism = kParallelism == 8 ? 3 : 2;

    // Host memory
    ac_complex<float> *host_input_data =
        (ac_complex<float> *)std::malloc(sizeof(ac_complex<float>) * kN * kN);
    ac_complex<float> *host_output_data =
        (ac_complex<float> *)std::malloc(sizeof(ac_complex<float>) * kN * kN);
    ac_complex<double> *host_verify =
        (ac_complex<double> *)std::malloc(sizeof(ac_complex<double>) * kN * kN);
    ac_complex<double> *host_verify_tmp =
        (ac_complex<double> *)std::malloc(sizeof(ac_complex<double>) * kN * kN);

    if ((host_input_data == nullptr) || (host_output_data == nullptr) ||
        (host_verify == nullptr) || (host_verify_tmp == nullptr)) {
      std::cerr << "Failed to allocate host memory with malloc." << std::endl;
      std::terminate();
    }

    // Initialize input and produce verification data
    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        int where = mangle ? MangleBits<kLogN>(Coordinates<kN>(i, j))
                           : Coordinates<kN>(i, j);
        host_verify[Coordinates<kN>(i, j)].r() = host_input_data[where].r() =
            (float)((double)rand() / (double)RAND_MAX);
        host_verify[Coordinates<kN>(i, j)].i() = host_input_data[where].i() =
            (float)((double)rand() / (double)RAND_MAX);
      }
    }

    // Device memory
    ac_complex<float> *input_data;
    ac_complex<float> *output_data;
    ac_complex<float> *temp_data;

    if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
      std::cout << "Using USM device allocations" << std::endl;
      // Allocate FPGA DDR memory.
      input_data = sycl::malloc_device<ac_complex<float>>(kN * kN, q);
      output_data = sycl::malloc_device<ac_complex<float>>(kN * kN, q);
      temp_data = sycl::malloc_device<ac_complex<float>>(kN * kN, q);
    } else if (q.get_device().has(sycl::aspect::usm_host_allocations)) {
      std::cout << "Using USM host allocations" << std::endl;
      // No device allocations means that we are probably in a SYCL HLS
      // flow

#if defined IS_BSP
      auto prop_list = sycl::property_list{};
#else
      // In the SYCL HLS flow, we need to define the memory interface.
      // For, that we need to assign a location to the memory being accessed.
      auto prop_list = sycl::property_list{
          sycl::ext::intel::experimental::property::usm::buffer_location(1)};
#endif

      input_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q);
      output_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q, prop_list);
      temp_data = sycl::malloc_host<ac_complex<float>>(kN * kN, q, prop_list);
    } else {
      std::cerr << "USM device allocations or USM host allocations must be "
                   "supported to run this sample."
                << std::endl;
      std::terminate();
    }

    if (input_data == nullptr || output_data == nullptr ||
        temp_data == nullptr) {
      std::cerr << "Failed to allocate USM memory." << std::endl;
      std::terminate();
    }

    // Copy the input data from host DDR to USM memory
    q.memcpy(input_data, host_input_data, sizeof(ac_complex<float>) * kN * kN)
        .wait();

    std::cout << "Launching a " << kN * kN << " points " << kParallelism
              << "-parallel " << (inverse ? "inverse " : "")
              << "FFT transform (" << (mangle ? "alternative" : "ordered")
              << " data layout)" << std::endl;

    /*
     * A 2D FFT transform requires applying a 1D FFT transform to each matrix
     * row followed by a 1D FFT transform to each column of the intermediate
     * result.
     * A single FFT engine can process rows and columns back-to-back. However,
     * as matrix data is stored in global memory, the efficiency of memory
     * accesses will impact the overall performance. Accessing consecutive
     * memory locations leads to efficient access patterns. However, this is
     * obviously not possible when accessing both rows and columns.
     *
     * The implementation is divided between three concurrent SYCL kernels, as
     * depicted below:
     *
     *  --------------------      --------------      --------------------------
     *  | read matrix rows | ---> | FFT engine | ---> | bit-reverse, transpose |
     *  |                  |      |            |      |    and write matrix    |
     *  --------------------      --------------      --------------------------
     *
     * This sequence of kernels does back-to-back row processing followed by a
     * data transposition and writes the results back to memory. The host code
     * runs these kernels twice to produce the overall 2D FFT transform
     *
     *
     * These kernels transfer data through pipes.
     * This avoids the need to read and write intermediate data using global
     * memory.
     *
     * In many cases the FFT engine is a building block in a large application.
     * In this case, the memory layout of the matrix can be altered to achieve
     * higher memory transfer efficiency. This implementation demonstrates how
     * an alternative memory layout can improve performance. The host switches
     * between the two memory layouts using a kernel argument. See the
     * 'MangleBits' function for additional details.
     */

    double start_time;
    double end_time;

    // This is a limitation of the design
    static_assert(kN / kParallelism >= kParallelism);

    // Kernel to kernel pipes
    using FetchToFFT =
        sycl::ext::intel::pipe<class FetchToFFTPipe,
                               std::array<ac_complex<float>, kParallelism>, 0>;
    using FFTToTranspose =
        sycl::ext::intel::pipe<class FFTToTransposePipe,
                               std::array<ac_complex<float>, kParallelism>, 0>;

    for (int i = 0; i < 2; i++) {
      ac_complex<float> *to_read = i == 0 ? input_data : temp_data;
      ac_complex<float> *to_write = i == 0 ? temp_data : output_data;

      // Start a 1D FFT on the matrix rows/columns
      auto fetch_event = q.single_task<class FetchKernel>(
          Fetch<kLogN, kLogParallelism, FetchToFFT, float>{to_read, mangle});

      auto fft_event = q.single_task<class FFTKernel>(
          FFT<kLogN, kLogParallelism, FetchToFFT, FFTToTranspose, float>{
              inverse});

      auto transpose_event = q.single_task<class TransposeKernel>(
          Transpose<kLogN, kLogParallelism, FFTToTranspose, float>{to_write,
                                                                   mangle});

      fft_event.wait();
      transpose_event.wait();

      if (i == 0) {
        start_time = fetch_event.template get_profiling_info<
            sycl::info::event_profiling::command_start>();
      } else {
        end_time = transpose_event.template get_profiling_info<
            sycl::info::event_profiling::command_end>();
      }
    }

    double kernel_runtime = (end_time - start_time) / 1.0e9;

    // Copy the output data from the USM memory to the host DDR
    q.memcpy(host_output_data, output_data, sizeof(ac_complex<float>) * kN * kN)
        .wait();

    std::cout << "Processing time = " << kernel_runtime << "s" << std::endl;

    double gpoints_per_sec = ((double)kN * kN / kernel_runtime) * 1e-9;
    double gflops = 2 * 5 * kN * kN * (log((float)kN) / log((float)2)) /
                    (kernel_runtime * 1e9);

    std::cout << "Throughput = " << gpoints_per_sec << " Gpoints / sec ("
              << gflops << " Gflops)" << std::endl;

    // Check signal to noise ratio

    // Run reference code
    for (int i = 0; i < kN; i++) {
      FourierTransformGold<kLogN>(host_verify + Coordinates<kN>(i, 0), inverse);
    }

    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        host_verify_tmp[Coordinates<kN>(j, i)] =
            host_verify[Coordinates<kN>(i, j)];
      }
    }

    for (int i = 0; i < kN; i++) {
      FourierTransformGold<kLogN>(host_verify_tmp + Coordinates<kN>(i, 0),
                                  inverse);
    }

    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        host_verify[Coordinates<kN>(j, i)] =
            host_verify_tmp[Coordinates<kN>(i, j)];
      }
    }

    double magnitude_sum = 0;
    double noise_sum = 0;
    for (int i = 0; i < kN; i++) {
      for (int j = 0; j < kN; j++) {
        int where = mangle ? MangleBits<kLogN>(Coordinates<kN>(i, j))
                           : Coordinates<kN>(i, j);
        double magnitude = (double)host_verify[Coordinates<kN>(i, j)].r() *
                               (double)host_verify[Coordinates<kN>(i, j)].r() +
                           (double)host_verify[Coordinates<kN>(i, j)].i() *
                               (double)host_verify[Coordinates<kN>(i, j)].i();
        double noise = (host_verify[Coordinates<kN>(i, j)].r() -
                        (double)host_output_data[where].r()) *
                           (host_verify[Coordinates<kN>(i, j)].r() -
                            (double)host_output_data[where].r()) +
                       (host_verify[Coordinates<kN>(i, j)].i() -
                        (double)host_output_data[where].i()) *
                           (host_verify[Coordinates<kN>(i, j)].i() -
                            (double)host_output_data[where].i());

        magnitude_sum += magnitude;
        noise_sum += noise;
      }
    }
    double db = 10 * log(magnitude_sum / noise_sum) / log(10.0);

    std::cout << "Signal to noise ratio on output sample: " << db << std::endl;
    std::cout << " --> " << (db > 120 ? "PASSED" : "FAILED") << std::endl;

    sycl::free(input_data, q);
    free(output_data, q);
    free(temp_data, q);

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }
}

/////// HELPER FUNCTIONS ///////

// provides a linear offset in the input array
template <int n>
int Coordinates(int iteration, int i) {
  return iteration * n + i;
}

// Reference Fourier transform
template <int lognr_points>
void FourierTransformGold(ac_complex<double> *data, bool inverse) {
  constexpr int kNrPoints = 1 << lognr_points;

  // The inverse requires swapping the real and imaginary component
  if (inverse) {
    for (int i = 0; i < kNrPoints; i++) {
      double tmp = data[i].r();
      data[i].r() = data[i].i();
      data[i].i() = tmp;
    }
  }

  // Do a FT recursively
  FourierStage<lognr_points>(data);

  // The inverse requires swapping the real and imaginary component
  if (inverse) {
    for (int i = 0; i < kNrPoints; i++) {
      double tmp = data[i].r();
      data[i].r() = data[i].i();
      data[i].i() = tmp;
    }
  }
}

template <int lognr_points>
void FourierStage(ac_complex<double> *data) {
  if constexpr (lognr_points > 0) {
    constexpr int kNrPoints = 1 << lognr_points;

    ac_complex<double> *half1 = (ac_complex<double> *)malloc(
        sizeof(ac_complex<double>) * kNrPoints / 2);
    ac_complex<double> *half2 = (ac_complex<double> *)malloc(
        sizeof(ac_complex<double>) * kNrPoints / 2);

    if (half1 == nullptr || half2 == nullptr) {
      std::cerr << "Failed to allocate memory in validation function."
                << std::endl;
      std::terminate();
    }

    for (int i = 0; i < kNrPoints / 2; i++) {
      half1[i] = data[2 * i];
      half2[i] = data[2 * i + 1];
    }

    FourierStage<lognr_points - 1>(half1);
    FourierStage<lognr_points - 1>(half2);

    for (int i = 0; i < kNrPoints / 2; i++) {
      data[i].r() = half1[i].r() +
                    cos(2 * M_PI * i / kNrPoints) * half2[i].r() +
                    sin(2 * M_PI * i / kNrPoints) * half2[i].i();
      data[i].i() = half1[i].i() -
                    sin(2 * M_PI * i / kNrPoints) * half2[i].r() +
                    cos(2 * M_PI * i / kNrPoints) * half2[i].i();
      data[i + kNrPoints / 2].r() =
          half1[i].r() - cos(2 * M_PI * i / kNrPoints) * half2[i].r() -
          sin(2 * M_PI * i / kNrPoints) * half2[i].i();
      data[i + kNrPoints / 2].i() =
          half1[i].i() + sin(2 * M_PI * i / kNrPoints) * half2[i].r() -
          cos(2 * M_PI * i / kNrPoints) * half2[i].i();
    }

    free(half1);
    free(half2);
  }
}
