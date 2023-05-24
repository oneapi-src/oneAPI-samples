#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <vector>

#define DEBUG 0

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

  constexpr size_t kAMatrixSize = kSamplesCount * kFeaturesCount;
  constexpr size_t kEigenValuesCount = kFeaturesCount;
  constexpr size_t kEigenVectorsMatrixSize = kFeaturesCount * kFeaturesCount;

  constexpr int k_zero_threshold_1e = -7;

  // Get the number of times we want to repeat the decomposition
  // from the command line.
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;

  if (repetitions < 1) {
    std::cout << "Number of repetitions is lower that 1." << std::endl;
    std::cout << "The computation must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kPCAsToCompute = 1000;

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

    // Create vectors to hold all the input and output matrices
    std::vector<float> a_matrix;
    std::vector<float> eigen_values_vector;
    std::vector<float> eigen_vectors_matrix;
    std::vector<ac_int<1, false>> rank_deficient_flag;

    a_matrix.resize(kAMatrixSize * kPCAsToCompute);
    eigen_values_vector.resize(kEigenValuesCount * kPCAsToCompute);
    eigen_vectors_matrix.resize(kEigenVectorsMatrixSize * kPCAsToCompute);
    rank_deficient_flag.resize(kPCAsToCompute);

    std::cout << "Generating " << kPCAsToCompute << " random ";
    std::cout << "matri" << (kPCAsToCompute > 1 ? "ces" : "x") << " of size "
              << kSamplesCount << "x" << kFeaturesCount << " " << std::endl;

    constexpr bool print_debug_information = false;

    GoldenPCA<double> pca(kSamplesCount, kFeaturesCount, kPCAsToCompute,
                          print_debug_information);
    pca.populateA();
    pca.standardizeA();
    pca.computeCovarianceMatrix();
    pca.computeEigenValuesAndVectors();

    // Copy all the input matrices to the of the golden implementation to the
    // a_matrix that uses the float datatype, which is going to be used by the
    // hardware implementation
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      for (int row = 0; row < kSamplesCount; row++) {
        for (int column = 0; column < kFeaturesCount; column++) {
          a_matrix[matrix_index * kFeaturesCount * kSamplesCount +
                   column * kSamplesCount + row] =
              pca.a_matrix[matrix_index * kFeaturesCount * kSamplesCount +
                           row * kFeaturesCount +
                           column];  // implicit double to float cast here
        }
      }
    }

    std::cout << "Running Principal Component analysis of " << kPCAsToCompute
              << " matri" << (kPCAsToCompute > 1 ? "ces " : "x ") << repetitions
              << " times" << std::endl;

    PCAKernel<kSamplesCount, kFeaturesCount, FIXED_ITERATIONS,
              k_zero_threshold_1e>(
        a_matrix, eigen_values_vector, eigen_vectors_matrix,
        rank_deficient_flag, q, kPCAsToCompute, repetitions);

    if (DEBUG) {
      for (int matrix_index = 0; matrix_index < kPCAsToCompute;
           matrix_index++) {
        std::cout << "\n Results : " << matrix_index << std::endl;
        std::cout << "\n Eigen Values: \n";
        for (int i = 0; i < kFeaturesCount; i++) {
          std::cout << eigen_values_vector[matrix_index * kEigenValuesCount + i]
                    << " ";
        }
        std::cout << "\n";

        std::cout << "\n Eigen Vectors: \n";
        for (int i = 0; i < kFeaturesCount; i++) {
          for (int j = 0; j < kFeaturesCount; j++) {
            std::cout
                << eigen_vectors_matrix[matrix_index * kEigenVectorsMatrixSize +
                                        i * kFeaturesCount + j]
                << " ";
          }
          std::cout << "\n";
        }
      }
    }

    /////////////////////////////////////////////////////////////////////
    /////////  Sorting and matching with golden value ///////////////////
    /////////////////////////////////////////////////////////////////////
    std::cout << "Verifying results..." << std::endl;

    std::vector<int> sort_index_golden(kFeaturesCount);
    int total_iteration = 0;  // for the run time prediction on FPGA
    int passed_matrices = 0;
    int kernel_innacurate_result_flag_count = 0;
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {

      if (rank_deficient_flag[matrix_index] != 0){
        // Skip the verification of the current matrix as it was flagged as
        // rank deficient, which is not supported by the kernel
        kernel_innacurate_result_flag_count++;
        continue;
      }

      int eigen_vectors_offset = matrix_index * kEigenVectorsMatrixSize;
      int eigen_values_offset = matrix_index * kEigenValuesCount;

      // Initialize the indexes for sorting the eigen values.
      for (int i = 0; i < kFeaturesCount; i++) {
        sort_index_golden[i] = i;
      }

      // Sort the golden Eigen values by reordering the indexes that we are
      // going to access
      // The Eigen values and vectors from the kernel are already sorted
      std::sort(sort_index_golden.begin(), sort_index_golden.end(),
                [=](int a, int b) {
                  return fabs(pca.eigen_values[eigen_values_offset + a]) >
                         fabs(pca.eigen_values[eigen_values_offset + b]);
                });

      // Absolute threshold at which we consider there is an error
      constexpr float k_diff_threshold = 1e-3;
      int eigen_values_errors = 0;

      // Check the Eigen values
      for (int i = 0; i < kFeaturesCount; i++) {
        int sorted_index_golden = sort_index_golden[i];

        float golden_eigen_value =
            pca.eigen_values[eigen_values_offset + sorted_index_golden];
        float kernel_eigen_value = eigen_values_vector[eigen_values_offset + i];

        if (fabs(fabs(golden_eigen_value) - fabs(kernel_eigen_value)) >
                k_diff_threshold ||
            isnan(golden_eigen_value) ||
            isnan(kernel_eigen_value)) {
          eigen_values_errors++;
          std::cout << "Mismatch between golden and kernel Eigen value for matrix "
                    << matrix_index << std::endl
                    << "golden: " << golden_eigen_value << std::endl
                    << "kernel: " << kernel_eigen_value << std::endl;
        }
      }

      if (eigen_values_errors != 0) {
        std::cout << "Matrix: " << matrix_index << std::endl
                  << "Eigen values mismatch count: " << eigen_values_errors
                  << std::endl;
      }

      // Check the Eigen vectors
      int eigen_vectors_errors = 0;
      for (int row = 0; row < kFeaturesCount; row++) {
        for (int column = 0; column < kFeaturesCount; column++) {
          float golden_vector_element =
              pca.eigen_vectors[eigen_vectors_offset + row * kFeaturesCount +
                                sort_index_golden[column]];
          float kernel_vector_element =
              eigen_vectors_matrix[eigen_vectors_offset + row * kFeaturesCount +
                                   column];

          if (fabs(fabs(golden_vector_element) - fabs(kernel_vector_element)) >
                  k_diff_threshold ||
              isnan(golden_vector_element) || isnan(kernel_vector_element)) {
            eigen_vectors_errors++;

            std::cout << "Mismatch between golden and kernel Eigen vector "
                      << "at index " << row << ", " << column << " in matrix "
                      << matrix_index << std::endl
                      << "golden: " << golden_vector_element << std::endl
                      << "kernel: " << kernel_vector_element << std::endl
                      << std::endl;
          }
        }
      }

      if (eigen_vectors_errors != 0) {
        std::cout << "Matrix: " << matrix_index << std::endl
                  << "Eigen vector elements mismatch count: "
                  << eigen_vectors_errors << std::endl;
      } else {
        passed_matrices++;
      }

    }  // end for:matrix_index

    if (kernel_innacurate_result_flag_count > 0) {
      std::cout << "During the execution, the kernel identified "
                << kernel_innacurate_result_flag_count
                << " rank deficient matrices." << std::endl;
      std::cout << "These matrices were omitted from the data verification." << std::endl;
    }

    if ((passed_matrices + kernel_innacurate_result_flag_count) < kPCAsToCompute){
      std::cerr << "Errors were identified." << std::endl;
      std::cerr << "Pass rate: "
                << (100.0 * (passed_matrices + kernel_innacurate_result_flag_count)) /
                       (kPCAsToCompute)
                << "%" << std::endl;
      std::terminate();
    }


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


    std::cout << "All the tests passed." << std::endl;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }

  return 0;
}  // end of main
