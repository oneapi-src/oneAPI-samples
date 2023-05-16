#include <math.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#define KDEFLIM 2
#define SHIFT_NOISE 1e-3
#define NO_SHIFT_ITER 10

#define DEBUG 1
#define DEBUG_MATRIX_INDEX 0

#include "exception_handler.hpp"
#include "golden_pca.hpp"
#include "pca.hpp"

typedef double DataTypeCPU;

// Real single precision floating-point PCA
void PCAsycl(std::vector<float> &a_matrix, std::vector<float> &q_matrix,
             std::vector<float> &r_matrix, sycl::queue &q, int matrix_count,
             int repetitions) {}

int main(int argc, char *argv[]) {
  constexpr size_t kFeaturesCount = FEATURES_COUNT;
  constexpr size_t kSamplesCount = SAMPLES_COUNT;

  constexpr size_t kAMatrixSize = kFeaturesCount * kFeaturesCount;
  constexpr size_t kEigenValuesCount = kFeaturesCount;
  constexpr size_t kEigenVectorsMatrixSize = kFeaturesCount * kFeaturesCount;

  constexpr bool kUseRayleighShift =
      false;  // Use Rayleigh shift instead of the Wilkinson shift

  constexpr int k_zero_threshold_1e = -5;

  // Get the number of times we want to repeat the decomposition
  // from the command line.
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;

  if (repetitions < 1) {
    std::cout << "Number of repetitions is lower that 1." << std::endl;
    std::cout << "The computation must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kPCAsToCompute = 1;

  try {
    // SYCL boilerplate
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

    // Create vectors to hold all the input and output matrices
    std::vector<float> a_matrix;
    std::vector<float> eigen_values_vector;
    std::vector<float> eigen_vectors_matrix;

    a_matrix.resize(kAMatrixSize * kPCAsToCompute);
    eigen_values_vector.resize(kEigenValuesCount * kPCAsToCompute);
    eigen_vectors_matrix.resize(kEigenVectorsMatrixSize * kPCAsToCompute);

    std::cout << "Generating " << kPCAsToCompute << " random ";
    std::cout << "matri" << (kPCAsToCompute > 1 ? "ces" : "x") << " of size "
              << kSamplesCount << "x" << kFeaturesCount << " " << std::endl;

    constexpr bool print_debug_information = true;

    PCA<double> pca(kSamplesCount, kFeaturesCount, kPCAsToCompute,
                    print_debug_information);
    pca.populateA();
    pca.standardizeA();
    pca.computeCovarianceMatrix();
    pca.computeEigenValuesAndVectors();

    // Copy all the input matrices to the of the golden implementation to the
    // a_matrix that uses the float datatype, which is going to be used by the
    // hardware implementation
    for (int k = 0; k < (kAMatrixSize * kPCAsToCompute); k++) {
      a_matrix[k] = pca.a_matrix[k];  // implicit double to float cast here
    }

    std::cout << "Running Principal Component analysis of " << kPCAsToCompute
              << " matri" << (kPCAsToCompute > 1 ? "ces " : "x ") << repetitions
              << " times" << std::endl;

    PCAsyclImpl<kSamplesCount, kFeaturesCount, FIXED_ITERATIONS,
                kUseRayleighShift, k_zero_threshold_1e>(
        a_matrix, eigen_values_vector, eigen_vectors_matrix, q, kPCAsToCompute,
        repetitions);

    if (DEBUG) {
      std::cout << "\n Eigen Values: \n";
      for (int i = 0; i < kFeaturesCount; i++) {
        std::cout << eigen_values_vector[i] << " ";
      }
      std::cout << "\n";

      std::cout << "\n Eigen Vectors: \n";
      for (int i = 0; i < kFeaturesCount; i++) {
        for (int j = 0; j < kFeaturesCount; j++) {
          std::cout << eigen_vectors_matrix[i * kFeaturesCount + j] << " ";
        }
        std::cout << "\n";
      }
    }

    /////////////////////////////////////////////////////////////////////
    /////////  Sorting and matching with golden value ///////////////////
    /////////////////////////////////////////////////////////////////////
    std::vector<int> s_index(kFeaturesCount);
    std::vector<int> s_index_SYCL(kFeaturesCount);
    int total_iteration = 0;  // for the run time prediction on FPGA
    int passsed_matrices = 0;
    int KernelFlagCount = 0;
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      int matrix_offset = matrix_index * kAMatrixSize;
      int eigen_vector_offset = matrix_index * kFeaturesCount;

      // Initialize the indexes for sorting the eigen values.

      if (eigen_values_vector[kFeaturesCount + eigen_vector_offset] == 1) {
        KernelFlagCount++;
        continue;
      }

      for (int i = 0; i < kFeaturesCount; i++) {
        s_index[i] = i;
        s_index_SYCL[i] = i;
      }

      // sorting the eigen values
      std::sort(s_index.begin(), s_index.end(), [=](int a, int b) {
        return fabs(pca.a_matrix[matrix_offset + a * kFeaturesCount + a]) >
               fabs(pca.a_matrix[matrix_offset + b * kFeaturesCount + b]);
      });

      // std::sort(s_index_SYCL.begin(), s_index_SYCL.end(), [=](int a, int b) \
    //   { return fabs(rq_matrix[matrix_offset+a*kFeaturesCount+a]) > fabs(rq_matrix[matrix_offset+b*kFeaturesCount+b]);});

      // Relative error is used in error calculation of eigen values
      // This is beacuse eigen values can come in 1000s

      constexpr float k_diff_threshold = 1e-3;
      int rq_ecount_SYCL = 0;
      if (DEBUG) std::cout << "\nEigen values are:\n";
      for (int i = 0; i < kFeaturesCount; i++) {
        int sI = s_index[i];
        if (DEBUG)
          std::cout << pca.a_matrix[matrix_offset + sI * kFeaturesCount + sI]
                    << " ";

        if (fabs(fabs(pca.a_matrix[matrix_offset + sI * kFeaturesCount + sI]) -
                 fabs(eigen_values_vector[eigen_vector_offset + i])) /
                    (fabs(pca.a_matrix[matrix_offset + sI * kFeaturesCount +
                                       sI])) >
                k_diff_threshold ||
            isnan(pca.a_matrix[matrix_offset + sI * kFeaturesCount + sI]) ||
            isnan(eigen_values_vector[i + eigen_vector_offset])) {
          rq_ecount_SYCL++;
          std::cout << "Mis matched CPU and SYCL eigen values are: "
                    << pca.a_matrix[matrix_offset + sI * kFeaturesCount + sI]
                    << ", " << eigen_values_vector[i + eigen_vector_offset]
                    << " at i: " << sI << "\n";
        }
      }

      if (rq_ecount_SYCL == 0) {
      } else {
        std::cout << "\nMatrix: " << matrix_index
                  << " Error is found between kernel and numpy eigen values, "
                     "Mismatch count: "
                  << rq_ecount_SYCL << "\n";
      }

      if (rq_ecount_SYCL > 0) std::cout << "\n\n\n";

      int qq_ecountSYCL = 0;
      if (DEBUG) std::cout << "\n Eigen vector is: \n";
      for (int i = 0; i < kFeaturesCount; i++) {
        for (int j = 0; j < kFeaturesCount; j++) {
          if (DEBUG)
            std::cout << pca.eigen_vectors[matrix_offset + j * kFeaturesCount +
                                           s_index[i]]
                      << " ";

          if (fabs(
                  fabs(pca.eigen_vectors[matrix_offset + j * kFeaturesCount +
                                         s_index[i]]) -
                  fabs(eigen_vectors_matrix[matrix_offset + i * kFeaturesCount +
                                            s_index_SYCL[j]])) >
                  k_diff_threshold ||
              isnan(eigen_vectors_matrix[matrix_offset + i * kFeaturesCount +
                                         s_index_SYCL[j]]) ||
              isnan(pca.eigen_vectors[matrix_offset + j * kFeaturesCount +
                                      s_index[i]])) {
            qq_ecountSYCL++;
            std::cout
                << "Mis matched CPU and SYCL QQ values and corr eigen value "
                   "are: "
                << pca.eigen_vectors[matrix_offset + j * kFeaturesCount +
                                     s_index[i]]
                << ", "
                << eigen_vectors_matrix[matrix_offset + i * kFeaturesCount +
                                        s_index_SYCL[j]]
                << " " << eigen_values_vector[i + eigen_vector_offset]
                << " at i,j:" << i << "," << j << "\n";
          }
        }
        if (DEBUG) std::cout << "\n";
      }

      if (qq_ecountSYCL == 0) {
        passsed_matrices++;
      } else {
        std::cout
            << "Matrix: " << matrix_index
            << "  Error: Mismatch is found between SYCL and numpy QQ, count: "
            << qq_ecountSYCL << "\n";
      }

      if (qq_ecountSYCL > 0) std::cout << "\n\n\n";
    }
    std::cout << "Failed PCA flag count from kernel is: " << KernelFlagCount
              << "\n";
    std::cout << "Mis Matched matrix count is "
              << kPCAsToCompute - passsed_matrices - KernelFlagCount << "\n";
    std::cout << "Passed matrix percenage is "
              << (100.0 * passsed_matrices) / (kPCAsToCompute - KernelFlagCount)
              << "\n";

    // Runtime Prediction
    const bool is_complex = false;
    constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;
    static constexpr int kDummyIterations =
        FIXED_ITERATIONS > kFeaturesCount
            ? (kFeaturesCount - 1) * kFeaturesCount / 2 +
                  (FIXED_ITERATIONS - kFeaturesCount) * kFeaturesCount
            : kFeaturesCount * (kFeaturesCount - 1) / 2;
    // Total number of iterations (including dummy iterations)
    static constexpr int kIterations =
        kFeaturesCount + kFeaturesCount * (kFeaturesCount + 1) / 2 +
        kDummyIterations;

    double preProcLat =
        (kFeaturesCount * kFeaturesCount) *
            ((kSamplesCount + kFeaturesCount - 1) / kFeaturesCount) +
        kFeaturesCount * (kFeaturesCount + kNumElementsPerDDRBurst - 1) /
            kNumElementsPerDDRBurst;

    double QRItrLat =
        total_iteration * (kIterations + kFeaturesCount * kFeaturesCount);

    double SortLat = 2 * kFeaturesCount * kFeaturesCount;

    double WBackLat =
        (kFeaturesCount * (kFeaturesCount + kNumElementsPerDDRBurst - 1) /
             kNumElementsPerDDRBurst +
         (kFeaturesCount + kNumElementsPerDDRBurst) / kNumElementsPerDDRBurst);

    double clocks =
        1.0 * (preProcLat + QRItrLat + (SortLat + WBackLat) * kPCAsToCompute) *
        repetitions;
    double predicted_time = clocks / 2.39e8;
    std::cout << "Predicted runtime is: " << predicted_time << " seconds\n";

    return 0;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;

    std::terminate();
  } catch (std::bad_alloc const &e) {
    std::cerr << "Caught a memory allocation exception on the host: "
              << e.what() << std::endl;
    std::cerr << "   You can reduce the memory requirement by reducing the "
                 "number of matrices generated. Specify a smaller number when "
                 "running the executable."
              << std::endl;
    std::cerr << "   In this run, more than "
              << ((kAMatrixSize * 3) * 2 * kPCAsToCompute * sizeof(float)) /
                     pow(2, 30)
              << " GBs of memory was requested for the decomposition of a "
              << "matrix of size " << kFeaturesCount << " x " << kSamplesCount
              << std::endl;
    std::terminate();
  }
}  // end of main
