#include "qrd.hpp"

/*
  COMPLEX, COLS_COMPONENT, ROWS_COMPONENT and FIXED_ITERATIONS are defined
  by the build system.
  Depending on the value of COMPLEX, the real or complex QRDecomposition

  Function arguments:
  - AMatrix:    The input matrix. Interpreted as a transposed matrix.
  - QMatrix:    The Q matrix. The function will overwrite this matrix.
  - RMatrix     The R matrix. The function will overwrite this matrix.
                The vector will only contain the upper triangular elements
                of the matrix, in a row by row fashion.
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read sequentially from the AMatrix 
                vector.
  - reps:       The number of repetitions of the computation to execute.
                (for performance evaluation)
*/

#if COMPLEX == 0
// Real single precision floating-point QR Decomposition
void FloatQRDecomposition(std::vector<float> &AMatrix, 
                          std::vector<float> &QMatrix,
                          std::vector<float> &RMatrix,
                          sycl::queue &q, 
                          size_t matrices, 
                          size_t reps) {
  constexpr bool isComplex = false;
  QRDecomposition_impl< COLS_COMPONENT, 
                        ROWS_COMPONENT, 
                        FIXED_ITERATIONS, 
                        isComplex, 
                        float>(AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}
#else
// Complex single precision floating-point QR Decomposition
void ComplexFloatQRDecomposition( std::vector<ac_complex<float>> &AMatrix, 
                                  std::vector<ac_complex<float>> &QMatrix,
                                  std::vector<ac_complex<float>> &RMatrix,
                                  sycl::queue &q, 
                                  size_t matrices, 
                                  size_t reps) {
  constexpr bool isComplex = true;
  QRDecomposition_impl< COLS_COMPONENT, 
                        ROWS_COMPONENT, 
                        FIXED_ITERATIONS, 
                        isComplex, 
                        float>(AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}
#endif
