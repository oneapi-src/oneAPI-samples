#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#define DEBUG 0

#include "exception_handler.hpp"
#include "golden_pca.hpp"
#include "pca.hpp"

int main(int argc, char *argv[]) {
#if defined(FPGA_SIMULATOR)
  // Only read a few lines of the input data when running the simulator
  constexpr size_t kPCAsToCompute = 1;
  constexpr size_t kFeaturesCount = 4;
  constexpr size_t kSamplesCount = 8;
  constexpr bool kBenchmarkMode = false;
  constexpr bool kBenchmarkModeForcelyDisabled = true;
  std::cout << "The benchmark mode is disabled when running the simulator"
            << std::endl;
#elif BENCHMARK
  constexpr size_t kPCAsToCompute = 1;
  constexpr size_t kFeaturesCount = 8;
  constexpr size_t kSamplesCount = 4176;
  constexpr bool kBenchmarkMode = true;
  constexpr bool kBenchmarkModeForcelyDisabled = false;
#else
  constexpr size_t kPCAsToCompute = 8;
  constexpr bool kBenchmarkMode = false;
  constexpr bool kBenchmarkModeForcelyDisabled = false;
  constexpr size_t kFeaturesCount = FEATURES_COUNT;
  constexpr size_t kSamplesCount = SAMPLES_COUNT;
#endif

  constexpr size_t kAMatrixSize = kSamplesCount * kFeaturesCount;
  constexpr size_t kEigenValuesCount = kFeaturesCount;
  constexpr size_t kEigenVectorsMatrixSize = kFeaturesCount * kFeaturesCount;

  constexpr int k_zero_threshold_1e = -8;

#if defined(FPGA_EMULATOR) or defined(FPGA_SIMULATOR)
  int repetitions = 1;
#else
  int repetitions = 4096;
#endif
  std::string in_file_name = "";

  if constexpr (kBenchmarkMode || kBenchmarkModeForcelyDisabled) {
    // We expect to read the dataset path from the program arguments
    if ((argc != 2) && (argc != 3)) {
      std::cout << "Usage: " << std::endl
                << "./pca.xxx <path to abalone.csv> n" << std::endl
                << "where n is an optional parameter which specifies how many "
                   "times to "
                   "repeat the computation (for performance evaluation) "
                << std::endl;
      std::terminate();
    }

    // Get the file path
    in_file_name = std::string(argv[1]);

    if (argc == 3) {
      // get the number of repetitions
      repetitions = std::stoi(argv[2]);
    }
  } else {
    // We expect to read the dataset path from the program arguments
    if (argc == 2) {
      // get the number of repetitions
      repetitions = std::stoi(argv[1]);
    }
  }

  // Get the number of times we want to repeat the decomposition
  // from the command line.

  if (repetitions < 1) {
    std::cout << "Number of repetitions is lower that 1." << std::endl;
    std::cout << "The computation must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

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

    if (kBenchmarkMode) {
      std::cout << "Reading the input data from file." << std::endl;
      std::cout << "Features count: " << kFeaturesCount << std::endl;
      std::cout << "Samples count: " << kSamplesCount << std::endl;
    } else {
      std::cout << "Generating " << kPCAsToCompute << " random ";
      std::cout << "matri" << (kPCAsToCompute > 1 ? "ces" : "x") << " of size "
                << kSamplesCount << "x" << kFeaturesCount << " " << std::endl;
    }

    constexpr bool print_debug_information = false;

    GoldenPCA<double> pca(kSamplesCount, kFeaturesCount, kPCAsToCompute,
                          print_debug_information, kBenchmarkMode,
                          in_file_name);
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
              k_zero_threshold_1e>(a_matrix, eigen_values_vector,
                                   eigen_vectors_matrix, rank_deficient_flag, q,
                                   kPCAsToCompute, repetitions);

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
    int passed_matrices = 0;
    int kernel_innacurate_result_flag_count = 0;
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      if (rank_deficient_flag[matrix_index] != 0) {
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
      constexpr float k_diff_threshold = 1e-2;
      int eigen_values_errors = 0;

      // Check the Eigen values
      for (int i = 0; i < kFeaturesCount; i++) {
        int sorted_index_golden = sort_index_golden[i];

        float golden_eigen_value =
            pca.eigen_values[eigen_values_offset + sorted_index_golden];
        float kernel_eigen_value = eigen_values_vector[eigen_values_offset + i];

        if (fabs(fabs(golden_eigen_value) - fabs(kernel_eigen_value)) >
                k_diff_threshold ||
            std::isnan(golden_eigen_value) || std::isnan(kernel_eigen_value)) {
          eigen_values_errors++;
          std::cout
              << "Mismatch between golden and kernel Eigen value for matrix "
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
              eigen_vectors_matrix[eigen_vectors_offset +
                                   column * kFeaturesCount + row];

          if (fabs(fabs(golden_vector_element) - fabs(kernel_vector_element)) >
                  k_diff_threshold ||
              std::isnan(golden_vector_element) ||
              std::isnan(kernel_vector_element)) {
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
      std::cout << "These matrices were omitted from the data verification."
                << std::endl;
    }

    if ((passed_matrices + kernel_innacurate_result_flag_count) <
        kPCAsToCompute) {
      std::cerr << "Errors were identified." << std::endl;
      std::cerr << "Pass rate: "
                << (100.0 *
                    (passed_matrices + kernel_innacurate_result_flag_count)) /
                       (kPCAsToCompute)
                << "%" << std::endl;
      std::terminate();
    }

    std::cout << "All the tests passed." << std::endl;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }

  return 0;
}  // end of main
