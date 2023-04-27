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

#define DEBUG 0
#define DEBUG_MATRIX_INDEX 0

#include "exception_handler.hpp"
#include "pca.hpp"
#include "pca_cpu.hpp"
#include "qr_MGS.hpp"

typedef double DataTypeCPU;

// Real single precision floating-point PCA
void PCAsycl(std::vector<float> &a_matrix, std::vector<float> &q_matrix,
             std::vector<float> &r_matrix, sycl::queue &q, int matrix_count,
             int repetitions) {}

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kFeaturesCount = FEATURES_COUNT;
  constexpr size_t kSamplesCount = SAMPLES_COUNT;
  constexpr size_t ka_matrix_size =
      kFeaturesCount * ((kSamplesCount + kFeaturesCount - 1) / kFeaturesCount) *
      kFeaturesCount;
  constexpr size_t ka_matrix_size_host = kFeaturesCount * kSamplesCount;
  constexpr size_t kAMatrixSize = kFeaturesCount * kFeaturesCount;
  constexpr size_t kQQMatrixSize = kFeaturesCount * kFeaturesCount;

  constexpr bool kUseRayleighShift = false;
  constexpr int k_zero_threshold_1e = -5;
  constexpr float k_zero_threshold = 1e-5;

  std::cout << "ka_matrix_size is: " << ka_matrix_size << "\n";

  // Get the number of times we want to repeat the decomposition
  // from the command line.
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;

  if (repetitions < 1) {
    std::cout << "Number of repetitions is lower that 1." << std::endl;
    std::cout << "The computation must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kPCAsToCompute = 4;

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
    std::cout << "Device name: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Create vectors to hold all the input and output matrices
    std::vector<float> a_matrix;
    std::vector<float> eig_matrix;
    std::vector<float> qq_matrix;

    a_matrix.resize(ka_matrix_size * kPCAsToCompute);
    eig_matrix.resize((kFeaturesCount + 1) * kPCAsToCompute);
    qq_matrix.resize(kQQMatrixSize * kPCAsToCompute);

    std::cout << "Generating " << kPCAsToCompute << " random ";
    std::cout << "matri" << (kPCAsToCompute > 1 ? "ces" : "x") << " of size "
              << kFeaturesCount << "x" << kSamplesCount << " " << std::endl;

    // Generate the random symmetric square matrices
    srand(kRandomSeed);

    PCA<double> pca(kSamplesCount, kFeaturesCount, kPCAsToCompute, 0);
    pca.populate_A();
    pca.normalizeSamples();
    pca.calculate_covariance();

    // TODO: Restriction on kFeaturesCount % kSamplesCount == 0?

    // copying the covariance matrix
    // kFeaturesCount x kFeaturesCount block datalayout on device
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      for (int blk = 0;
           blk < (kSamplesCount + kFeaturesCount - 1) / kFeaturesCount; blk++) {
        for (int i = 0; i < kFeaturesCount; i++) {
          for (int j = 0; j < kFeaturesCount; j++) {
            if ((blk * kFeaturesCount + j) < kSamplesCount) {
              a_matrix[matrix_index * ka_matrix_size +
                       blk * kFeaturesCount * kFeaturesCount +
                       i * kFeaturesCount + j] =
                  pca.matA[matrix_index * ka_matrix_size_host +
                           (blk * kFeaturesCount + j) * kFeaturesCount + i];
            } else {
              a_matrix[matrix_index * ka_matrix_size +
                       blk * kFeaturesCount * kFeaturesCount +
                       i * kFeaturesCount + j] = 0;
            }
          }
        }
      }
    }

    if (DEBUG) {
      for (int matrix_index = 0; matrix_index < kPCAsToCompute;
           matrix_index++) {
        std::cout << "A MATRIX " << matrix_index << std::endl;
        for (size_t i = 0; i < kFeaturesCount; i++) {
          for (size_t j = 0; j < kSamplesCount; j++) {
            std::cout << a_matrix[matrix_index * ka_matrix_size +
                                  i * kSamplesCount + j]
                      << " ";
          }  // end of col
          std::cout << std::endl;
        }  // end of row
      }    // end of matrix_index
    }

    std::cout << "Running Principal Component analysis of " << kPCAsToCompute
              << " matri" << (kPCAsToCompute > 1 ? "ces " : "x ") << repetitions
              << " times" << std::endl;

    PCAsyclImpl<kSamplesCount, kFeaturesCount, FIXED_ITERATIONS, kUseRayleighShift, k_zero_threshold_1e>(
        a_matrix, eig_matrix, qq_matrix, q, kPCAsToCompute, repetitions);

    // eigen value & vector computation on CPU for same data
    std::vector<DataTypeCPU> a_matrix_cpu(kAMatrixSize * kPCAsToCompute);
    std::vector<DataTypeCPU> eigen_vectors_cpu(kAMatrixSize * kPCAsToCompute);
    std::vector<DataTypeCPU> TmpRow(kFeaturesCount);
    std::vector<int> sIndex(kFeaturesCount);
    std::vector<int> sIndexSYCL(kFeaturesCount);

    if (DEBUG) {
      std::cout << "\n Eigen Values: \n";
      for (int i = 0; i < kFeaturesCount; i++) {
        std::cout << eig_matrix[i] << " ";
      }
      std::cout << "\n";

      std::cout << "\n QQ Matrix: \n";
      for (int i = 0; i < kFeaturesCount; i++) {
        for (int j = 0; j < kFeaturesCount; j++) {
          std::cout << qq_matrix[i * kFeaturesCount + j] << " ";
        }
        std::cout << "\n";
      }
    }

    // data strucutre for golden results from numpy
    // std::vector<float> py_w(kFeaturesCount*kPCAsToCompute);
    // std::vector<float> py_V(kFeaturesCount*kFeaturesCount*kPCAsToCompute);

    // copying input matrix and initial eigen vectors for
    // CPU based computation
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      int matrix_offset = matrix_index * kAMatrixSize;

      // copy A matrix to CPU data
      // column major to row major conversion
      for (int i = 0; i < kFeaturesCount; i++) {
        for (int j = 0; j < kFeaturesCount; j++) {
          a_matrix_cpu[matrix_offset + i * kFeaturesCount + j] =
              pca.matC[matrix_offset + i * kFeaturesCount + j];
        }
      }

      // initialize the eigen vectors to identity mtrix
      for (int i = 0; i < kFeaturesCount; i++) {
        for (int j = 0; j < kFeaturesCount; j++) {
          eigen_vectors_cpu[matrix_offset + i * kFeaturesCount + j] =
              (i == j) ? 1 : 0;
        }
      }
    }

    // Writing the input matrix to a file
    // python script will read the file and process
    std::ofstream osA("mat_A.txt");
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      int matrix_offset = matrix_index * kAMatrixSize;
      for (int i = 0; i < kFeaturesCount; i++) {
        for (int j = 0; j < kFeaturesCount; j++) {
          osA << std::setprecision(15)
              << a_matrix[matrix_offset + j * kFeaturesCount + i];
          if (j != kFeaturesCount - 1 || i != kFeaturesCount - 1 ||
              matrix_index != kPCAsToCompute - 1) {
            osA << ",";
          }
        }
      }
    }
    osA.close();

    ////////////////////////////////////////////////////////////////
    ////////  QRD Iteration ////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    std::ofstream rq_matrix_file("Debug_RQ_CPU.txt");
    std::ofstream qq_matrix_file("Debug_QQ_CPU.txt");
    std::ofstream q_matrix_file("Debug_Q_CPU.txt");
    std::ofstream r_matrix_file("Debug_R_CPU.txt");
    std::ofstream a_matrix_file("Debug_A_CPU.txt");

    int total_iteration = 0;  // for the run time prediction on FPGA
    constexpr int kMaxIterationsForConvergence = kFeaturesCount * 100;

    DataTypeCPU *R, *Q;  // pointerfor Q and R matrix after QR decomposition
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      int matrix_offset = matrix_index * kAMatrixSize;
      // QR decomposition on CPU
      QR_Decmp<DataTypeCPU> qrd_cpu(&a_matrix_cpu[matrix_offset], kFeaturesCount,
                                 matrix_index);
      int curent_last_row = kFeaturesCount;
      for (int li = 0; li < kMaxIterationsForConvergence; li++) {
        // convergence test
        bool close_to_zero = true;

        // check zero thereshold for lower part

        // Wilkinson shift computation
        float a_wilk =
            a_matrix_cpu[matrix_offset + (curent_last_row - 2) * kFeaturesCount + curent_last_row - 2];
        float b_wilk =
            a_matrix_cpu[matrix_offset + (curent_last_row - 1) * kFeaturesCount + curent_last_row - 2];
        float c_wilk =
            a_matrix_cpu[matrix_offset + (curent_last_row - 1) * kFeaturesCount + curent_last_row - 1];

        float lamda = (a_wilk - c_wilk) / 2.0;
        float sign_lamda = (lamda > 0) ? 1.0 : -1.0;
        float wilkinson_shift = c_wilk - (sign_lamda * b_wilk * b_wilk) /
                                          (fabs(lamda) + sqrt(lamda * lamda +
                                                              b_wilk * b_wilk));

        float shift = kUseRayleighShift ? c_wilk : wilkinson_shift; 

        shift -= shift * SHIFT_NOISE;
        shift = (li < NO_SHIFT_ITER) ? 0 : shift;

        if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) {
          a_matrix_file << "\n\nA Matrix before shift at iteration: " << li << "\n";
        }
        for (int i = 0; i < curent_last_row; i++) {
          for (int j = 0; j < curent_last_row; j++) {
            if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX)
              a_matrix_file << a_matrix_cpu[matrix_offset + i * kFeaturesCount + j]
                    << " ";
          }
          if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) a_matrix_file << "\n";
        }

        for (int i = 0; i < curent_last_row; i++) {
          a_matrix_cpu[matrix_offset + i * kFeaturesCount + i] -= shift;
        }

        if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) {
          a_matrix_file << "\n\nA Matrix after shift at iteration: " << li << "\n";
        }
        for (int i = 0; i < curent_last_row; i++) {
          for (int j = 0; j < curent_last_row; j++) {
            if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX)
              a_matrix_file << a_matrix_cpu[matrix_offset + i * kFeaturesCount + j]
                    << " ";
          }
          if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) a_matrix_file << "\n";
        }

        qrd_cpu.QR_decompose(curent_last_row);
        R = qrd_cpu.get_R();
        Q = qrd_cpu.get_Q();
        // RQ computation and updating A

        for (int i = 0; i < curent_last_row; i++) {
          for (int j = 0; j < curent_last_row; j++) {
            a_matrix_cpu[matrix_offset + i * kFeaturesCount + j] = 0;
            for (int k = 0; k < curent_last_row; k++) {
              a_matrix_cpu[matrix_offset + i * kFeaturesCount + j] +=
                  R[i * kFeaturesCount + k] * Q[k * kFeaturesCount + j];
            }
          }
        }

        if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) {
          q_matrix_file << "\n\nQ Matrix at iteration: " << li << "\n";
        }
        if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) {
          r_matrix_file << "\n\nR Matrix at iteration: " << li << "\n";
        }
        for (int i = 0; i < curent_last_row; i++) {
          for (int j = 0; j < curent_last_row; j++) {
            if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX)
              q_matrix_file << Q[i * kFeaturesCount + j] << " ";
            if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX)
              r_matrix_file << R[i * kFeaturesCount + j] << " ";
          }
          if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) q_matrix_file << "\n";
          if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) r_matrix_file << "\n";
        }

        // adding back the shift from the matrix
        for (int i = 0; i < curent_last_row; i++) {
          a_matrix_cpu[matrix_offset + i * kFeaturesCount + i] += shift;
        }

        if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) {
          rq_matrix_file << "\n\nRQ Matrix at iteration: " << li << "\n";
        }
        for (int i = 0; i < kFeaturesCount; i++) {
          for (int j = 0; j < kFeaturesCount; j++) {
            if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) {
              rq_matrix_file << a_matrix_cpu[matrix_offset + i * kFeaturesCount + j]
                  << " ";
            }
          }
          if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) {
            rq_matrix_file << "\n";
          }
        }

        // Eigen vector accumulation
        if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX)
          qq_matrix_file << "QQ Matrix at iteration: " << li << "\n";
        for (int i = 0; i < kFeaturesCount; i++) {
          std::fill(TmpRow.begin(), TmpRow.end(), 0);
          for (int j = 0; j < kFeaturesCount; j++) {
            for (int k = 0; k < kFeaturesCount; k++) {
              float I_val = (k == j) ? 1 : 0;
              float q_val =
                  (j >= curent_last_row || k >= curent_last_row) ? I_val : Q[k * kFeaturesCount + j];
              TmpRow[j] +=
                  eigen_vectors_cpu[matrix_offset + i * kFeaturesCount + k] *
                  q_val;
            }
          }
          for (int k = 0; k < kFeaturesCount; k++) {
            eigen_vectors_cpu[matrix_offset + i * kFeaturesCount + k] =
                TmpRow[k];
            if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX)
              qq_matrix_file << eigen_vectors_cpu[matrix_offset + i * kFeaturesCount + k]
                  << " ";
          }
          if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) qq_matrix_file << "\n";
        }
        if (DEBUG && matrix_index == DEBUG_MATRIX_INDEX) qq_matrix_file << "\n";

        for (int j = 0; j < curent_last_row - 1; j++) {
          if (std::fabs(
                  a_matrix_cpu[matrix_offset + (curent_last_row - 1) * kFeaturesCount + j]) >
              k_zero_threshold) {
            close_to_zero = false;
            break;
          }
        }

        if (close_to_zero && curent_last_row == KDEFLIM) {
          total_iteration += li + 1;
          break;
        } else if (close_to_zero) {
          curent_last_row -= 1;
        }
      }
    }
    rq_matrix_file.close();
    qq_matrix_file.close();
    a_matrix_file.close();

    // exit(0);

    /////////////////////////////////////////////////////////////////////
    /////////  Sorting and matching with golden value ///////////////////
    /////////////////////////////////////////////////////////////////////

    int passsed_marixes = 0;
    int KernelFlagCount = 0;
    for (int matrix_index = 0; matrix_index < kPCAsToCompute; matrix_index++) {
      int matrix_offset = matrix_index * kAMatrixSize;
      int Eigmatrix_offset = matrix_index * (kFeaturesCount + 1);
      // int evec_offset = matrix_index * kFeaturesCount;

      // Initialize the idexes for sorting
      // the eigen values. Pyhton implmentation
      // could use different algorithm, hence
      // the order of eigen values might be different

      if (eig_matrix[kFeaturesCount + Eigmatrix_offset] == 1) {
        KernelFlagCount++;
        continue;
      }

      for (int i = 0; i < kFeaturesCount; i++) {
        sIndex[i] = i;
        sIndexSYCL[i] = i;
      }

      // sorting the eigen values
      std::sort(sIndex.begin(), sIndex.end(), [=](int a, int b) {
        return fabs(a_matrix_cpu[matrix_offset + a * kFeaturesCount + a]) >
               fabs(a_matrix_cpu[matrix_offset + b * kFeaturesCount + b]);
      });

      // std::sort(sIndexSYCL.begin(), sIndexSYCL.end(), [=](int a, int b) \
    //   { return fabs(rq_matrix[matrix_offset+a*kFeaturesCount+a]) > fabs(rq_matrix[matrix_offset+b*kFeaturesCount+b]);});

      // Relative error is used in error calculation of eigen values
      // This is beacuse eigen values can come in 1000s

      constexpr float k_diff_threshold = 1e-3;
      int rq_ecount_SYCL = 0;
      if (DEBUG) std::cout << "\nEigen values are:\n";
      for (int i = 0; i < kFeaturesCount; i++) {
        int sI = sIndex[i];
        if (DEBUG)
          std::cout << a_matrix_cpu[matrix_offset + sI * kFeaturesCount + sI]
                    << " ";

        if (fabs(fabs(a_matrix_cpu[matrix_offset + sI * kFeaturesCount + sI]) -
                 fabs(eig_matrix[Eigmatrix_offset + i])) /
                    (fabs(a_matrix_cpu[matrix_offset + sI * kFeaturesCount +
                                       sI])) >
                k_diff_threshold ||
            isnan(a_matrix_cpu[matrix_offset + sI * kFeaturesCount + sI]) ||
            isnan(eig_matrix[i + Eigmatrix_offset])) {
          rq_ecount_SYCL++;
          std::cout << "Mis matched CPU and SYCL eigen values are: "
                    << a_matrix_cpu[matrix_offset + sI * kFeaturesCount + sI]
                    << ", " << eig_matrix[i + Eigmatrix_offset]
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
            std::cout << eigen_vectors_cpu[matrix_offset + j * kFeaturesCount +
                                           sIndex[i]]
                      << " ";

          if (fabs(fabs(eigen_vectors_cpu[matrix_offset + j * kFeaturesCount +
                                          sIndex[i]]) -
                   fabs(qq_matrix[matrix_offset + i * kFeaturesCount +
                                  sIndexSYCL[j]])) > k_diff_threshold ||
              isnan(qq_matrix[matrix_offset + i * kFeaturesCount +
                              sIndexSYCL[j]]) ||
              isnan(eigen_vectors_cpu[matrix_offset + j * kFeaturesCount +
                                      sIndex[i]])) {
            qq_ecountSYCL++;
            std::cout
                << "Mis matched CPU and SYCL QQ values and corr eigen value "
                   "are: "
                << eigen_vectors_cpu[matrix_offset + j * kFeaturesCount +
                                     sIndex[i]]
                << ", "
                << qq_matrix[matrix_offset + i * kFeaturesCount + sIndexSYCL[j]]
                << " " << eig_matrix[i + Eigmatrix_offset] << " at i,j:" << i
                << "," << j << "\n";
          }
        }
        if (DEBUG) std::cout << "\n";
      }

      if (qq_ecountSYCL == 0) {
        passsed_marixes++;
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
              << kPCAsToCompute - passsed_marixes - KernelFlagCount << "\n";
    std::cout << "Passed matrix percenage is "
              << (100.0 * passsed_marixes) / (kPCAsToCompute - KernelFlagCount)
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
