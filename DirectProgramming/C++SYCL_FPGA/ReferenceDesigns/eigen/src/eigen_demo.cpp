#include <math.h>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>


#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>


#include <list>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#define KTHRESHOLD 1e-5
#define KDEFLIM 2
#define KETHRESHOLD 1e-3
#define KETHRESHOLD_Eigen 1e-3
#define RELSHIFT 1
#define SHIFT_NOISE 1e-2
#define SHIFT_NOISE_CPU 1e-2
#define ITER_PER_EIGEN 8

#define DEBUGEN 1
#define DEBUGMINDEX 0
#define DEBUG 1

#include "exception_handler.hpp"

#include "eigen.hpp"
#include "qr_MGS.hpp"
// #include "qr_decom.hpp"
#include "hessenberg_qrd.hpp"


typedef double DTypeCPU;

/*
  COMPLEX, COLS_COMPONENT, ROWS_COMPONENT and FIXED_ITERATIONS are defined
  by the build system.
  Depending on the value of COMPLEX, the real or complex QRDecomposition is
  defined

  Function arguments:
  - a_matrix:    The input matrix. Interpreted as a transposed matrix.
  - q_matrix:    The Q matrix. The function will overwrite this matrix.
  - r_matrix     The R matrix. The function will overwrite this matrix.
                 The vector will only contain the upper triangular elements
                 of the matrix, in a row by row fashion.
  - q:           The device queue.
  - matrix_count: Number of matrices to decompose.
  - repetitions: The number of repetitions of the computation to execute.
                 (for performance evaluation)
*/



#if COMPLEX == 0
// Real single precision floating-point QR Decomposition
void QRDecomposition(std::vector<float> &a_matrix, std::vector<float> &q_matrix,
                     std::vector<float> &r_matrix, sycl::queue &q,
                     int matrix_count,
                     int repetitions) {
  constexpr bool is_complex = false;
  QRDecompositionImpl<COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS,
                       is_complex, float>(a_matrix, q_matrix, r_matrix, q,
                                          matrix_count, repetitions);

}

// Real double precision floating-point QR Decomposition
// void QRDecomposition(std::vector<double> &a_matrix, std::vector<double> &q_matrix,
//                      std::vector<double> &r_matrix, sycl::queue &q,
//                      int matrix_count,
//                      int repetitions) {
//   constexpr bool is_complex = false;
//   QRDecompositionImpl<COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS,
//                        is_complex, double>(a_matrix, q_matrix, r_matrix, q,
//                                           matrix_count, repetitions);

// }
#else
// Complex single precision floating-point QR Decomposition
void QRDecomposition(std::vector<ac_complex<float> > &a_matrix,
                     std::vector<ac_complex<float> > &q_matrix,
                     std::vector<ac_complex<float> > &r_matrix, sycl::queue &q,
                     int matrix_count,
                     int repetitions) {
  constexpr bool is_complex = true;
  QRDecompositionImpl<COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS,
                       is_complex, float>(a_matrix, q_matrix, r_matrix, q,
                                          matrix_count, repetitions);
}
#endif

/*
  returns if both the real and complex parts of the given ac_complex
  value are finite
*/
bool IsFinite(ac_complex<float> val) {
  return std::isfinite(val.r()) && std::isfinite(val.i());
}

/*
  returns if the given value is finite
*/
bool IsFinite(float val) { return std::isfinite(val); }

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 100;
  constexpr size_t kRows = ROWS_COMPONENT;
  constexpr size_t kColumns = COLS_COMPONENT;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kRQMatrixSize = kRows * kColumns;
  constexpr size_t kQQMatrixSize = kRows * kColumns;
  constexpr bool kComplex = COMPLEX != 0;

  int iter = kRows*ITER_PER_EIGEN;


  // Get the number of times we want to repeat the decomposition
  // from the command line.
#if defined(FPGA_EMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;
#else
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;
#endif
  if (repetitions < 1) {
    std::cout << "Number of repetitions given is lower that 1." << std::endl;
    std::cout << "The decomposition must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kMatricesToDecompose = 1;

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
    sycl::property_list
                    queue_properties{sycl::property::queue::enable_profiling()};
    sycl::queue q = sycl::queue(selector,
                                fpga_tools::exception_handler,
                                queue_properties);

    sycl::device device = q.get_device();
    std::cout << "Device name: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Select a type for this compile depending on the value of COMPLEX
    // using T = std::conditional_t<kComplex, ac_complex<double>, double>;
    using T = std::conditional_t<kComplex, ac_complex<float>, float>;

    // Create vectors to hold all the input and output matrices
    std::vector<T> a_matrix;
    std::vector<T> rq_matrix;
    std::vector<T> qq_matrix;

    a_matrix.resize(kAMatrixSize * kMatricesToDecompose);
    rq_matrix.resize(kRQMatrixSize * kMatricesToDecompose);
    qq_matrix.resize(kQQMatrixSize * kMatricesToDecompose);

    std::cout << "Generating " << kMatricesToDecompose << " random ";
    if constexpr (kComplex) {
      std::cout << "complex ";
    } else {
      std::cout << "real ";
    }
    std::cout << "matri" << (kMatricesToDecompose > 1 ? "ces" : "x")
              << " of size "
              << kRows << "x" << kColumns << " " << std::endl;

    // Generate the random symmetric square matrices
    srand(kRandomSeed);

    for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
                                                                matrix_index++){
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col <= row; col++) {
          float random_real = (rand() % (kRandomMax - kRandomMin) + kRandomMin); // * 1.0/kRandomMax;
  #if COMPLEX == 0
          a_matrix[matrix_index * kAMatrixSize
                 + col * kRows + row] = random_real;
          a_matrix[matrix_index * kAMatrixSize
                 + row * kRows + col] = random_real;
  #else
          float random_imag = rand() % (kRandomMax - kRandomMin) + kRandomMin;
          ac_complex<float> random_complex{random_real, random_imag};
          a_matrix[matrix_index * kAMatrixSize
                 + col * kRows + row] = random_complex;
  #endif
        }  // end of col
      }    // end of row

      // a_matrix[0] = 112;
      // a_matrix[1] = 0;
      // a_matrix[2] = 0;
      // a_matrix[3] = 0;

      // a_matrix[4] = 0;
      // a_matrix[5] = 90;
      // a_matrix[6] = 0;
      // a_matrix[7] = 0;


      // a_matrix[8] = 0;
      // a_matrix[9] = 0;
      // a_matrix[10] = 30;
      // a_matrix[11] = 0;

      // a_matrix[12] = 0;
      // a_matrix[13] = 0;
      // a_matrix[14] = 0;
      // a_matrix[15] = 8;


  #ifdef DEBUG
      std::cout << "A MATRIX " << matrix_index << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << a_matrix[matrix_index * kAMatrixSize
                              + col * kRows + row] << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row
  #endif

    } // end of matrix_index


    std::cout << "Running QR decomposition of " << kMatricesToDecompose
              << " matri" << (kMatricesToDecompose > 1 ? "ces " : "x ")
              << repetitions << " times" << std::endl;

    QRDecomposition(a_matrix, rq_matrix, qq_matrix, q, kMatricesToDecompose,
                                                                  repetitions);

    // eigen value & vector computation on CPU for same data
    std::vector<DTypeCPU> a_matrix_cpu(kAMatrixSize * kMatricesToDecompose);
    std::vector<DTypeCPU> eigen_vectors_cpu(kAMatrixSize * kMatricesToDecompose);
    std::vector<DTypeCPU> TmpRow(kRows);
    std::vector<int> sIndex(kRows);
    std::vector<int> sIndexSYCL(kRows);


    if(DEBUG){
      std::cout << "\n RQ Matrix: \n";
      for(int i = 0; i < kRows; i++){
        for(int j = 0; j < kRows; j++){
          std::cout << rq_matrix[i*kRows+j] << " ";
        }
        std::cout << "\n";
      }

      std::cout << "\n QQ Matrix: \n";
      for(int i = 0; i < kRows; i++){
        for(int j = 0; j < kRows; j++){
          std::cout << qq_matrix[i*kRows+j]  << " ";
        }
        std::cout << "\n";
      }
    }

    // data strucutre for golden results from numpy 
    // std::vector<T> py_w(kRows*kMatricesToDecompose);
    // std::vector<T> py_V(kRows*kRows*kMatricesToDecompose);



    // copying input matrix and initial eigen vectors for
    // CPU based computation 
    for(int matrix_index = 0; matrix_index <kMatricesToDecompose; matrix_index++){
      int matrix_offset = matrix_index * kAMatrixSize;

      // copy A matrix to CPU data
      // column major to row major conversion 
      for(int i = 0; i < kRows; i++){
        for(int j = 0; j < kRows; j++){
          a_matrix_cpu[matrix_offset+ i*kRows+j] = a_matrix[matrix_offset+ j*kRows+i];
        }
      }

      //initialize the eigen vectors to identity mtrix
      for(int i = 0; i < kRows; i++){
        for(int j = 0; j < kRows; j++){
          eigen_vectors_cpu[matrix_offset + i*kRows+j] = (i == j) ? 1 : 0;
        }
      }
    }


      // Writing the input matrix to a file
      // python script will read the file and process
    std::ofstream osA("mat_A.txt");
    for(int matrix_index = 0; matrix_index <kMatricesToDecompose; matrix_index++){
      int matrix_offset = matrix_index * kAMatrixSize;
      for(int i = 0; i < kRows; i++){
        for(int j = 0; j < kRows; j++){
          osA << std::setprecision(15) << a_matrix[matrix_offset+j*kRows+i];
          if(j != kRows-1 || i != kRows-1 || matrix_index != kMatricesToDecompose-1){
            osA << ",";
          }
        }
      }
    }
    osA.close();

    // executing the python script 
    // it gets input matrices from the mat_A.txt file 
    // and write the eigen vectors and eigen values to mat_W.txt and mat_V.txt  
    // std::string cmd = "python2 ../src/eig_IQR.py " + std::to_string(kMatricesToDecompose) + " " + std::to_string(kRows);
    // if(system(cmd.c_str()) != 0){
    //   std::cout << "Error occured when trying to execute the python script\n";
    // }

    // reading back golden results: eigen values and eigen vectors
    // std::ifstream osW("mat_W.txt");
    // std::ifstream osV("mat_V.txt");  
    // for(int matrix_index = 0; matrix_index <kMatricesToDecompose; matrix_index++){
    //   int matrix_offset = matrix_index * kAMatrixSize;
    //   int evec_offset = matrix_index * kRows;
    //   for(int i = 0; i < kRows; i++){
    //     float tmp;
    //     osW >> tmp; //py_w[i+evec_offset];
    //     py_w[i+evec_offset] = tmp;
    //   }
      
    //   // reading back golden results
    //   for(int i = 0; i < kRows; i++){
    //     for(int j = 0; j < kRows; j++){
    //       float tmp;
    //       osV >> tmp; // py_V[matrix_offset+i*kRows+j];
    //       py_V[matrix_offset+i*kRows+j] = tmp;
    //     }
    //   }
      
    // }
    // osW.close();
    // osV.close();


////////////////////////////////////////////////////////////////
////////  QRD Iteration ////////////////////////////////////////
////////////////////////////////////////////////////////////////

    std::ofstream dRQ("Debug_RQ_CPU.txt");
    std::ofstream dQQ("Debug_QQ_CPU.txt");
    std::ofstream dQMat("Debug_Q_CPU.txt");
    std::ofstream dRMat("Debug_R_CPU.txt");
    std::ofstream dAMat("Debug_A_CPU.txt");


    DTypeCPU *R, *Q; // pointerfor Q and R matrix after QR decomposition 
    for(int matrix_index = 0; matrix_index <kMatricesToDecompose; matrix_index++){
      int matrix_offset = matrix_index * kAMatrixSize;
      // QR decomposition on CPU 
      QR_Decmp<DTypeCPU> qrd_cpu(&a_matrix_cpu[matrix_offset], kRows, matrix_index);
      // iter = 10000;
      int kP = kRows;
      for(int li = 0; li < iter; li++){

        // convergence test 
        bool close2zero = 1;

        // check zero thereshold for lower part 


        // Wilkinson shift computation 
        T a_wilk = a_matrix_cpu[matrix_offset+(kP-2)*kRows+kP-2];
        T b_wilk = a_matrix_cpu[matrix_offset+(kP-1)*kRows+kP-2];
        T c_wilk = a_matrix_cpu[matrix_offset+(kP-1)*kRows+kP-1];

        T lamda = (a_wilk - c_wilk)/2.0;
        T sign_lamda = (lamda > 0) ? 1.0 : -1.0;
        // T sign_lamda = (int)((lamda > 0) - (lamda < 0));

        T shift = RELSHIFT ? c_wilk : c_wilk - (sign_lamda*b_wilk*b_wilk)/(fabs(lamda) + sqrt(lamda * lamda + b_wilk*b_wilk));

        shift -= shift*SHIFT_NOISE;


        if(DEBUGEN && matrix_index == DEBUGMINDEX) {dAMat << "\n\nA Matrix before shift at iteration: " << li << "\n";}
        for(int i = 0; i < kP; i++){
          for(int j = 0; j < kP; j++){
              if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << a_matrix_cpu[matrix_offset+i*kRows+j]  << " ";
          }
          if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << "\n";
        }


        for(int i = 0; i < kP; i++){
          a_matrix_cpu[matrix_offset+i*kRows+i] -= shift;
        }

          if(DEBUGEN && matrix_index == DEBUGMINDEX) {dAMat << "\n\nA Matrix after shift at iteration: " << li << "\n";}
        for(int i = 0; i < kP; i++){
          for(int j = 0; j < kP; j++){
              if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << a_matrix_cpu[matrix_offset+i*kRows+j]  << " ";
          }
          if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << "\n";
        }

        qrd_cpu.QR_decompose(kP);
        R = qrd_cpu.get_R();
        Q = qrd_cpu.get_Q();
        // RQ computation and updating A 
        
        for(int i = 0; i < kP; i++){
          for(int j = 0; j < kP; j++){
            a_matrix_cpu[matrix_offset+i*kRows+j] = 0;
            for(int k = 0; k < kP; k++){
              a_matrix_cpu[matrix_offset+i*kRows+j] += R[i*kRows+k]*Q[k*kRows+j];
            } 
          }
        }

        if(DEBUGEN && matrix_index == DEBUGMINDEX) {dQMat << "\n\nQ Matrix at iteration: " << li << "\n";}
        if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRMat << "\n\nR Matrix at iteration: " << li << "\n";}
        for(int i = 0; i < kP; i++){
          for(int j = 0; j < kP; j++){
              if(DEBUGEN && matrix_index == DEBUGMINDEX) dQMat << Q[i*kRows+j] << " ";
              if(DEBUGEN && matrix_index == DEBUGMINDEX) dRMat << R[i*kRows+j] << " ";
          }
          if(DEBUGEN && matrix_index == DEBUGMINDEX) dQMat << "\n";
          if(DEBUGEN && matrix_index == DEBUGMINDEX) dRMat << "\n";
        }

        // adding back the shift from the matrix
        for(int i = 0; i < kP; i++){
          a_matrix_cpu[matrix_offset+i*kRows+i] += shift;
        }

        if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRQ << "\n\nRQ Matrix at iteration: " << li << "\n";}
        for(int i = 0; i < kRows; i++){
          for(int j = 0; j < kRows; j++){
            if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRQ << a_matrix_cpu[matrix_offset+i*kRows+j] << " ";}
          }
          if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRQ << "\n";}
        }

        // Eigen vector accumulation 
        if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << "QQ Matrix at iteration: " << li << "\n";
        for(int i = 0; i < kRows; i++){
          std::fill(TmpRow.begin(), TmpRow.end(), 0);
          for(int j = 0; j < kRows; j++){
            for(int k = 0; k < kRows; k++){
              T I_val = (k==j) ? 1 : 0;
              T q_val = (j >= kP || k >= kP) ? I_val : Q[k*kRows+j];
              TmpRow[j] += eigen_vectors_cpu[matrix_offset+i*kRows+k]*q_val;
            }
          }
          for(int k = 0; k < kRows; k++) {
            eigen_vectors_cpu[matrix_offset+i*kRows+k] = TmpRow[k];
            if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << eigen_vectors_cpu[matrix_offset+i*kRows+k] << " ";
          }
          if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << "\n";
        }
        if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << "\n";

        for(int j = 0; j < kP-1; j++){
          if(std::fabs(a_matrix_cpu[matrix_offset + (kP-1)*kRows+j]) > KTHRESHOLD){
            close2zero = 0;
            break;
          }
        }

        if(close2zero && kP == KDEFLIM){
          break;
        } else if(close2zero){
          kP -= 1;
        }
      
      }
    }
    dRQ.close();
    dQQ.close();
    dAMat.close();


    int passsed_marixes = 0;


/////////////////////////////////////////////////////////////////////
/////////  Sorting and matching with golden value ///////////////////
/////////////////////////////////////////////////////////////////////

  for(int matrix_index = 0; matrix_index <kMatricesToDecompose; matrix_index++){
    int matrix_offset = matrix_index * kAMatrixSize;
    // int evec_offset = matrix_index * kRows;

    // Initialize the idexes for sorting 
    // the eigen values. Pyhton implmentation
    // could use different algorithm, hence 
    // the order of eigen values might be different 

    for(int i = 0; i < kRows; i++){
      sIndex[i] = i;
      sIndexSYCL[i] = i;
    }

    // sorting the eigen values 
    std::sort(sIndex.begin(), sIndex.end(), [=](int a, int b) \
      { return fabs(a_matrix_cpu[matrix_offset+a*kRows+a]) > fabs(a_matrix_cpu[matrix_offset+b*kRows+b]);});

    std::sort(sIndexSYCL.begin(), sIndexSYCL.end(), [=](int a, int b) \
      { return fabs(rq_matrix[matrix_offset+a*kRows+a]) > fabs(rq_matrix[matrix_offset+b*kRows+b]);});


    // Relative error is used in error calculation of eigen values 
    // This is beacuse eigen values can come in 1000s

    T diff_threshold = KETHRESHOLD;
    int rq_ecount_SYCL = 0;
    for(int i = 0; i < kRows; i++){
      int sI = sIndex[i];
      int sIS = sIndexSYCL[i];
      if(fabs(fabs(a_matrix_cpu[matrix_offset + sI*kRows+sI])- fabs(rq_matrix[matrix_offset + sIS*kRows+sIS]))   \
      /(fabs(a_matrix_cpu[matrix_offset + sI*kRows+sI])) > KETHRESHOLD_Eigen 
      || isnan(a_matrix_cpu[matrix_offset + sI*kRows+sI]) || isnan(rq_matrix[matrix_offset + sIS*kRows+sIS])){
        rq_ecount_SYCL++;
        std::cout << "Mis matched CPU and SYCL eigen values are: " << a_matrix_cpu[matrix_offset + sI*kRows+sI] \
        << ", " << rq_matrix[matrix_offset + sIS*kRows+sIS] << " at i: " << sIS << "\n";
      }
    }

    if(rq_ecount_SYCL == 0){
    } else {
      std::cout << "\nMatrix: " << matrix_index << " Error is found between kernel and numpy eigen values, Mismatch count: " \
       << rq_ecount_SYCL << "\n";
    }

    if(rq_ecount_SYCL > 0) std::cout  << "\n\n\n";



    int qq_ecountSYCL = 0;
    for(int i = 0; i < kRows; i++){
      for(int j = 0; j < kRows; j++){
        if(fabs(fabs(eigen_vectors_cpu[matrix_offset + j*kRows+sIndex[i]]) - fabs(qq_matrix[matrix_offset + j*kRows+sIndexSYCL[i]])) > diff_threshold 
        || isnan(qq_matrix[matrix_offset + j*kRows+sIndexSYCL[i]]) || isnan(eigen_vectors_cpu[matrix_offset + j*kRows+sIndex[i]])){
          qq_ecountSYCL++;
          std::cout << "Mis matched CPU and SYCL QQ values and corr eigen value are: " << eigen_vectors_cpu[matrix_offset + j*kRows+sIndex[i]] << ", " << 
          qq_matrix[matrix_offset + j*kRows+sIndexSYCL[i]]  <<  " " << rq_matrix[matrix_offset + sIndex[i]*kRows+sIndex[i]]  << " at i,j:"
           << i << "," << j << "\n";
        }
      }
    }
 

    if(qq_ecountSYCL == 0){
      passsed_marixes++;
      // std::cout << "Matrix: " << matrix_index \
      // << " passed:  SYCL and numpy Eigen vectors are matched\n";
    } else {
      std::cout  << "Matrix: " << matrix_index \
      << "  Error: Mismatch is found between SYCL and numpy QQ, count: " << qq_ecountSYCL << "\n";
    }

    if(qq_ecountSYCL > 0)  std::cout << "\n\n\n";

  }
    std::cout << "Mis Matched matrix count is " << kMatricesToDecompose - passsed_marixes << "\n";
    std::cout << "Passed matrix percenage is " << (100.0 *passsed_marixes)/kMatricesToDecompose << "\n";
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
              << ((kAMatrixSize *3) * 2 * kMatricesToDecompose
                 * sizeof(float)) / pow(2, 30)
              << " GBs of memory was requested for the decomposition of a "
              << "matrix of size " << kRows << " x " << kColumns
              << std::endl;
    std::terminate();
  }
}  // end of main
