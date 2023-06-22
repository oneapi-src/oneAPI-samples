#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "fft2d.hpp"

void ReferenceFFT(short int dir, long m, sycl::double2 *d);

int main(int argc, char *argv[]) {
  constexpr int kRepetitions = 1;
  constexpr int kLogN = 10;
  constexpr int kN = 1 << kLogN;
  constexpr int kRows = kN;
  constexpr int kColumns = kN;

  constexpr int kLogRows = kLogN;
  constexpr int kLogColumns = kLogN;

  using T = sycl::float2;

  constexpr int kPadding = 8192 / sizeof(T);
  constexpr int kLogStride = 3;
  constexpr int kStride = 8;

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

    // Host memory
    T *host_input_data = (T *)std::malloc(
        kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding) *
        sizeof(T));
    T *host_output_data = (T *)std::malloc(
        kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding) *
        sizeof(T));
    sycl::double2 *host_verify_0 = (sycl::double2 *)std::malloc(
        kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding) *
        sizeof(sycl::double2));
    sycl::double2 *host_verify_1 = (sycl::double2 *)std::malloc(
        kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding) *
        sizeof(sycl::double2));

    if ((host_input_data == nullptr) || (host_output_data == nullptr) ||
        (host_verify_0 == nullptr) || (host_verify_1 == nullptr)) {
      std::cerr << "Failed to allocate host memory with malloc." << std::endl;
      std::terminate();
    }

    // Device memory
    T *device_input_data;
    T *device_output_data;
    T *device_temp_data;

    if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
      std::cout << "Using device allocations" << std::endl;
      // Allocate FPGA DDR memory.
      device_input_data = sycl::malloc_device<T>(
          kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding), q);
      device_output_data = sycl::malloc_device<T>(
          kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding), q);
      device_temp_data = sycl::malloc_device<T>(
          kRows * kColumns + (kColumns / kStride) * kPadding, q);
    } else if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
      std::cout << "Using shared allocations" << std::endl;
      // No device allocations means that we are probably in an IP authoring
      // flow
      device_input_data = sycl::malloc_host<T>(
          kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding), q);
      device_output_data = sycl::malloc_host<T>(
          kRepetitions * (kRows * kColumns + (kRows / kStride) * kPadding), q);
      device_temp_data = sycl::malloc_host<T>(
          kRows * kColumns + (kColumns / kStride) * kPadding, q);
    } else {
      std::cerr << "USM device allocations or USM shared allocations must be "
                   "supported to run this sample."
                << std::endl;
      std::terminate();
    }

    // Helper lambdas to compute addresses in one dimensional arrays
    auto coord = [](int &k, int &i, int j) {
      return k * (kRows * kColumns + (kRows / kStride) * kPadding) + i * kRows +
             (i / kStride) * kPadding + j;
    };
    auto coordt = [](int &k, int &i, int j) {
      return k * (kColumns * kRows + (kColumns / kStride) * kPadding) +
             i * kColumns + (i / kStride) * kPadding + j;
    };

    for (int k = 0; k < kRepetitions; k++) {
      // Create random data
      for (int i = 0; i < kRows; i++) {
        for (int j = 0; j < kColumns; j++) {
          host_verify_0[coord(k, i, j)][0] =
              host_input_data[coord(k, i, j)][0] =
                  rand() / (float)RAND_MAX * 32;
          host_verify_0[coord(k, i, j)][1] =
              host_input_data[coord(k, i, j)][1] =
                  rand() / (float)RAND_MAX * 32;
        }
      }

      // Run in-place FFT on CPU using code from web to verify our results
      for (int i = 0; i < kRows; i++) {
        ReferenceFFT(1, kLogColumns, host_verify_0 + coord(k, i, 0));
      }

      for (int i = 0; i < kRows; i++) {
        for (int j = 0; j < kColumns; j++) {
          host_verify_1[coordt(k, j, i)] = host_verify_0[coord(k, i, j)];
        }
      }

      for (int i = 0; i < kColumns; i++) {
        ReferenceFFT(1, kLogRows, host_verify_1 + coordt(k, i, 0));
      }

      for (int i = 0; i < kColumns; i++) {
        for (int j = 0; j < kRows; j++) {
          host_verify_0[coord(k, j, i)] = host_verify_1[coordt(k, i, j)];
        }
      }
    }

    // Copy the input data from host DDR to FPGA DDR
    q.memcpy(device_input_data, host_input_data,
             kRepetitions * sizeof(T) *
                 (kRows * kColumns + (kRows / kStride) * kPadding))
        .wait();

    // Compute the FFT2D on device
    auto fft2d_event =
        q.single_task<class FFT2DKernel>([=]() [[intel::kernel_args_restrict]] {
          FFT2D<kRows, kColumns, kStride, kPadding, kLogColumns, kLogStride, sycl::float2>(
              device_input_data, device_output_data, device_temp_data,
              kRepetitions);
        });

    // Compute the total time the execution lasted
    auto start_time = fft2d_event.template get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto end_time = fft2d_event.template get_profiling_info<
        sycl::info::event_profiling::command_end>();
    double kernel_runtime = (end_time - start_time) / 1.0e9;

    std::cout << "Total duration:   " << kernel_runtime << " s" << std::endl;

    double g_gpoints_per_sec =
        ((double)2 * kRepetitions * kRows * kColumns / kernel_runtime) * 1.0e-9;
    double g_gflops = 5 * kColumns * (log((float)kColumns) / log((float)2)) /
                          (kernel_runtime / (kRepetitions * kRows) * 1E9) +
                      5 * kRows * (log((float)kRows) / log((float)2)) /
                          (kernel_runtime / (kRepetitions * kColumns) * 1E9);
    std::cout << "Timing: " << kernel_runtime
              << " seconds total, throughput = " << g_gpoints_per_sec
              << "Gpoints/s" << std::endl;
    std::cout << "Throughput = " << g_gflops << std::endl;

    q.memcpy(host_output_data, device_output_data,
             kRepetitions * sizeof(T) *
                 (kRows * kColumns + (kRows / kStride) * kPadding))
        .wait();

    // Check
    double fpga_snr = 200;
    for (int k = 0; k < kRepetitions; k++) {
      double mag_sum = 0;
      double noise_sum = 0;
      for (int i = 0; i < kRows; i++) {
        for (int j = 0; j < kColumns; j++) {
          double magnitude = (double)host_output_data[coord(k, i, j)][0] *
                                 (double)host_output_data[coord(k, i, j)][0] +
                             (double)host_output_data[coord(k, i, j)][1] *
                                 (double)host_output_data[coord(k, i, j)][1];

          double noise = (host_verify_0[coord(k, i, j)][0] -
                          (double)host_output_data[coord(k, i, j)][0]) *
                             (host_verify_0[coord(k, i, j)][0] -
                              (double)host_output_data[coord(k, i, j)][0]) +
                         (host_verify_0[coord(k, i, j)][1] -
                          (double)host_output_data[coord(k, i, j)][1]) *
                             (host_verify_0[coord(k, i, j)][1] -
                              (double)host_output_data[coord(k, i, j)][1]);

          mag_sum += magnitude;
          noise_sum += noise;
        }
      }
      double db = 10 * log(mag_sum / noise_sum) / log(10.0);
      if (db < fpga_snr) fpga_snr = db;
    }

    std::cout << "Signal to noise ratio: " << fpga_snr << std::endl;

    if (fpga_snr > 115) printf("PASSED\n");

    free(device_input_data, q);
    free(device_output_data, q);
    free(device_temp_data, q);

    free(host_input_data);
    free(host_output_data);
    free(host_verify_0);
    free(host_verify_1);

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }

  return 0;
}  // end of main

void ReferenceFFT(short int dir, long m, sycl::double2 *d) {
  /* Following from: http://paulbourke.net/miscellaneous/dft/
     This computes an in-place complex-to-complex FFT
     x and y are the real and imaginary arrays of 2^m points.
     dir =  1 gives forward transform, -1 = reverse */
  long n, i, i1, j, k, i2, l, l1, l2;
  double c1, c2, tx, ty, t1, t2, u1, u2, z;

  /* Calculate the number of points */
  n = 1;
  for (i = 0; i < m; i++) n *= 2;

  /* Do the bit reversal */
  i2 = n >> 1;
  j = 0;
  for (i = 0; i < n - 1; i++) {
    if (i < j) {
      tx = d[i][0];
      ty = d[i][1];
      d[i][0] = d[j][0];
      d[i][1] = d[j][1];
      d[j][0] = tx;
      d[j][1] = ty;
    }
    k = i2;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  /* Compute the FFT */
  c1 = -1.0;
  c2 = 0.0;
  l2 = 1;
  for (l = 0; l < m; l++) {
    l1 = l2;
    l2 <<= 1;
    u1 = 1.0;
    u2 = 0.0;
    for (j = 0; j < l1; j++) {
      for (i = j; i < n; i += l2) {
        i1 = i + l1;
        t1 = u1 * d[i1][0] - u2 * d[i1][1];
        t2 = u1 * d[i1][1] + u2 * d[i1][0];
        d[i1][0] = d[i][0] - t1;
        d[i1][1] = d[i][1] - t2;
        d[i][0] += t1;
        d[i][1] += t2;
      }
      z = u1 * c1 - u2 * c2;
      u2 = u1 * c2 + u2 * c1;
      u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == 1) c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }

  /* Do the bit reversal */
  /*  i2 = n >> 1;
    j = 0;
    for (i=0;i<n-1;i++) {
      if (i < j) {
        tx = d[i][0];
        ty = d[i][1];
        d[i][0] = d[j][0];
        d[i][1] = d[j][1];
        d[j][0] = tx;
        d[j][1] = ty;
      }
      k = i2;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }
  */

  /* Scaling for forward transform */
  //  if (dir == 1) {
  //    for (i=0;i<n;i++) {
  //      d[i][0] /= n;
  //      d[i][1] /= n;
  //    }
  //  }
  //  cout << "  ... done post scaling\n";
}
