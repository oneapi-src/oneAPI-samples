#include <math.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

#include "qr_MGS.hpp"

/*
This file implements the steps to
identify principal components (Eigen vectors)
of a matrix and finally transform input matrix
along the directions of the principal components

Following are the main steps in order to transform an
matrix A made of multiple samples with different features.
The matrix A will contain n samples with p features
making it an n row, p columns matrix

Here are the steps performed by this file:
1. Generate random matrices
2. Standardize the matrices
3. Compute the covariance matrices of these matrices
4. Compute Eigen vectors and Eigen values of the covariance matrices using the
QR iteration method

*/
template <typename T>
class PCA {
 public:
  int samples;                           // number of samples
  int features;                          // number of features
  int matrix_count;                      // number of matrices
  bool debug;                            // print debug information if true
  std::vector<T> a_matrix;               // storage for input matrices
  std::vector<T> standardized_a_matrix;  // storage for standardized matrices
  std::vector<T> covariance_matrix;      // storage for covariance matrices
  std::vector<T> eigen_values;           // storage for the Eigen values
  std::vector<T> eigen_vectors;          // storage for the Eigen vectors

 private:
  std::default_random_engine gen;

 public:
  // Constructor
  PCA(int n, int p, int count, bool d) {
    samples = n;
    features = p;
    matrix_count = count;
    debug = d;

    a_matrix.resize(n * p * matrix_count);
    standardized_a_matrix.resize(n * p * matrix_count);
    covariance_matrix.resize(p * p * matrix_count);

    eigen_values.resize(p * matrix_count);
    eigen_vectors.resize(p * p * matrix_count);
  }

  // Generate the input matrix with index matrix_index with random numbers
  void populateIthA(int matrix_index) {
    constexpr float kRandomMin = -5;
    constexpr float kRandomMax = 5;

    std::uniform_real_distribution<float> distribution(kRandomMin, kRandomMax);

    for (int k = 0; k < a_matrix.size(); k++) {
      float value = distribution(gen);
      a_matrix[k] = value;
    }

    if (debug) {
      std::cout << "A matrix #" << matrix_index << std::endl;
      for (int row = 0; row < samples; row++) {
        for (int column = 0; column < features; column++) {
          std::cout << a_matrix[matrix_index * (samples * features) +
                                row * features + column]
                    << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  // Generate all the input matrices
  void populateA() {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      populateIthA(matrix_index);
    }
  }

  // Standardize the A matrix with index matrix_index
  void standardizeIthA(int matrix_index) {
    // The standardized matrix is defined as:
    // standardized_a_matrix[i][j] = (a_matrix[i][j] - mean[j])/(sd[j])
    // where mean[j] is the mean value of the column j and sd[j] is
    // the standard deviation of this column.
    // The standard deviation is defined as
    // sd[j] =  sqrt(sum((a[k][j] - mean[j])^2)/(N-1))

    if (debug)
      std::cout << "\nStandardizing A matrix #" << matrix_index << std::endl;

    // The current matrix offset in a_matrix
    int offset = matrix_index * samples * features;

    // Compute the mean of each column
    double mean[features];

    if (debug) std::cout << "\nMean of each column: " << std::endl;
    for (int column = 0; column < features; column++) {
      mean[column] = 0;
      for (int row = 0; row < samples; row++) {
        mean[column] += a_matrix[offset + row * features + column];
      }
      mean[column] /= samples;
      if (debug) std::cout << mean[column] << " ";
    }
    if (debug) std::cout << std::endl;

    // Compute the standard deviation of each column
    double standard_deviation[features];

    if (debug)
      std::cout << "\nStandard deviation of each column: " << std::endl;
    for (int column = 0; column < features; column++) {
      standard_deviation[column] = 0;
      for (int row = 0; row < samples; row++) {
        standard_deviation[column] +=
            (a_matrix[offset + row * features + column] - mean[column]) *
            (a_matrix[offset + row * features + column] - mean[column]);
      }
      standard_deviation[column] /= (samples - 1);
      standard_deviation[column] = sqrt(standard_deviation[column]);
      if (debug) std::cout << standard_deviation[column] << " ";
    }
    if (debug) std::cout << std::endl;

    // Compute the standardized matrix A
    if (debug) std::cout << "\nStandardized A matrix: " << std::endl;
    for (int row = 0; row < samples; row++) {
      for (int column = 0; column < features; column++) {
        standardized_a_matrix[offset + row * features + column] =
            (a_matrix[offset + row * features + column] - mean[column]) /
            standard_deviation[column];
        if (debug)
          std::cout << standardized_a_matrix[offset + row * features + column]
                    << " ";
      }
      if (debug) std::cout << std::endl;
    }
  }

  // Standardize all the A matrices
  void standardizeA() {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      standardizeIthA(matrix_index);
    }
  }

  // Compute the covariance matrix of the matrix with index matrix_index
  void computeCovarianceIthMatrix(int matrix_index) {
    // The covariance matrix is defined as the product of the
    // transposition of A by A. This matrix product then needs to be divided
    // by the number of samples-1.
    // This will result in a matrix of size features x features

    if (debug)
      std::cout << "\nCovariance matrix #" << matrix_index << std::endl;
    int a_matrix_offset = matrix_index * samples * features;
    int matrix_c_offset = matrix_index * features * features;
    for (int row = 0; row < features; row++) {
      for (int column = 0; column < features; column++) {
        double dot_product = 0;
        for (int k = 0; k < samples; k++) {
          dot_product +=
              standardized_a_matrix[a_matrix_offset + k * features + row] *
              standardized_a_matrix[a_matrix_offset + k * features + column];
        }
        covariance_matrix[matrix_c_offset + row * features + column] =
            dot_product / (samples - 1);
        if (debug)
          std::cout
              << covariance_matrix[matrix_c_offset + row * features + column]
              << " ";
      }
      if (debug) std::cout << std::endl;
    }
  }

  // Compute the covariance matrix of all the standardized A matrices
  void computeCovarianceMatrix() {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      computeCovarianceIthMatrix(matrix_index);
    }
  }

  // Compute the covariance matrix of the standardized A matrix
  void computeEigenValuesAndVectors() {
    // Compute the Eigen values and Eigen vectors using the QR iteration method
    // This implementation uses the Wilkinson shift to speedup the convergence

    constexpr float kZeroThreshold = 1e-7;

    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      if (debug)
        std::cout << "\nComputing Eigen values and vectors of matrix #"
                  << matrix_index << std::endl;

      int offset = matrix_index * features * features;

      // Compute the QR decomposition of the current matrix
      std::vector<double> q, r, rq;
      q.resize(features * features);
      r.resize(features * features);
      rq.resize(features * features);

      // Copy the covariance matrix into the input matrix to the QR
      // decomposition
      for (int k = 0; k < features * features; k++) {
        rq[k] = covariance_matrix[offset + k];
      }

      // Initialize the Eigen vectors matrix to the identity matrix
      for (int row = 0; row < features; row++) {
        for (int column = 0; column < features; column++) {
          eigen_vectors[offset + row * features + column] =
              row == column ? 1 : 0;
        }
      }

      // Count the number of iterations to abort if there is no convergence
      int iterations = 0;
      bool converged = false;
      while (!converged) {
        // Compute the shift value of the current matrix
        double shift_value = 0;

        // First find where the shift should be applied
        // Start from the last submatrix
        int shift_row = features - 2;
        for (int row = features - 1; row >= 1; row--) {
          bool row_is_zero = true;
          for (int col = 0; col < row; col++) {
            row_is_zero &= (fabs(rq[row * features + col]) < kZeroThreshold);
          }
          if (!row_is_zero) {
            break;
          }
          shift_row--;
        }

        if (shift_row >= 0) {
          // Compute the shift value
          // Take the submatrix:
          // [a b]
          // [b c]
          // and compute the shift such as
          // mu = c - (sign(d)* b*b)/(abs(d) + sqrt(d*d + b*b))
          // where d = (a - c)/2

          double a = rq[shift_row + features * shift_row];
          double b = rq[shift_row + features * (shift_row + 1)];
          double c = rq[(shift_row + 1) + features * (shift_row + 1)];

          double d = (a - c) / 2;
          double b_squared = b * b;
          double d_squared = d * d;
          double b_squared_signed = d < 0 ? -b_squared : b_squared;
          shift_value =
              c - b_squared_signed / (abs(d) + sqrt(d_squared + b_squared));
        }

        // Use the 90% percentage of the shift value to avoid
        // massive cancellations in the QRD
        shift_value *= 0.99;

        // if(debug) std::cout << "Shift value " << shift_value << std::endl;

        // if (debug){
        //   std::cout << "Before shift" << std::endl;
        //   for (int row = 0; row < features; row++) {
        //     for (int col = 0; col < features; col++) {
        //       std::cout << rq[row * features + col] << " ";
        //     }
        //     std::cout << std::endl;
        //   }
        //   std::cout << std::endl;
        // }

        // Subtract the shift value from the diagonal of RQ
        for (int row = 0; row < features; row++) {
          rq[row + features * row] -= shift_value;
        }

        // if (debug){
        //   std::cout << "Input matrix to QR" << std::endl;
        //   for (int row = 0; row < features; row++) {
        //     for (int col = 0; col < features; col++) {
        //       std::cout << rq[row * features + col] << " ";
        //     }
        //     std::cout << std::endl;
        //   }
        //   std::cout << std::endl;
        // }

        // Compute the actual QR decomposition
        {
          for (int row = 0; row < features; row++) {
            for (int column = 0; column < features; column++) {
              r[row * features + column] = 0;
              q[row * features + column] = 0;
            }
          }
          for (int i = 0; i < features; i++) {
            double norm = 0;
            for (int k = 0; k < features; k++) {
              norm +=
                  double(rq[k * features + i]) * double(rq[k * features + i]);
            }
            double rii = std::sqrt(norm);
            r[i * features + i] = rii;  // r_ii = ||a_i||

            for (int k = 0; k < features; k++) {
              q[k * features + i] = rq[k * features + i] / rii;
            }

            for (int j = i + 1; j < features; j++) {
              double dp = 0;
              for (int k = 0; k < features; k++) {
                dp += q[k * features + i] * rq[k * features + j];
              }
              r[i * features + j] = dp;

              for (int k = 0; k < features; k++) {
                rq[k * features + j] -= (dp * q[k * features + i]);
              }
            }
          }
        }

        // if (debug){
        //   std::cout << "Q matrix" << std::endl;
        //   for (int row = 0; row < features; row++) {
        //     for (int col = 0; col < features; col++) {
        //       std::cout << q[row* features + col] << " ";
        //     }
        //     std::cout << std::endl;
        //   }
        //   std::cout << std::endl;

        //   std::cout << "R matrix" << std::endl;
        //   for (int row = 0; row < features; row++) {
        //     for (int col = 0; col < features; col++) {
        //       std::cout << r[row* features + col] << " ";
        //     }
        //     std::cout << std::endl;
        //   }
        //   std::cout << std::endl;
        // }

        // Compute the updated Eigen vectors
        std::vector<T> eigen_vectors_q_product;
        eigen_vectors_q_product.resize(features * features);
        for (int row = 0; row < features; row++) {
          for (int col = 0; col < features; col++) {
            double prod = 0;
            for (int k = 0; k < features; k++) {
              prod += eigen_vectors[offset + row + features * k] *
                      q[k + features * col];
            }
            eigen_vectors_q_product[offset + row + features * col] = prod;
          }
        }
        // if (debug) std::cout << "Eigen vectors at iteration " << iterations
        // << std::endl;
        for (int row = 0; row < features; row++) {
          for (int col = 0; col < features; col++) {
            eigen_vectors[offset + row + features * col] =
                eigen_vectors_q_product[offset + row + features * col];
            // if (debug) std::cout << eigen_vectors[offset + row + features *
            // col] << " ";
          }
          // if (debug) std::cout << std::endl;
        }

        // Compute RQ
        // if (debug) std::cout << "RQ at iteration " << iterations <<
        // std::endl;
        for (int row = 0; row < features; row++) {
          for (int col = 0; col < features; col++) {
            double prod = 0;
            for (int k = 0; k < features; k++) {
              prod += r[row * features + k] * q[k * features + col];
            }
            rq[row * features + col] = prod;
            // if (debug) std::cout << prod << " ";
          }
          // if (debug) std::cout << std::endl;
        }

        // Add the shift value back to the diagonal of RQ
        for (int row = 0; row < features; row++) {
          rq[row + features * row] += shift_value;
        }

        // Check if we found all Eigen Values
        bool all_below_threshold = true;
        for (int row = 1; row < features; row++) {
          for (int col = 0; col < row; col++) {
            all_below_threshold &=
                (std::fabs(rq[row * features + col]) < kZeroThreshold);
          }
        }
        converged = all_below_threshold;

        iterations++;
        if (iterations > (features * features * features * 4096)) {
          std::cout << "Number of iterations too high" << std::endl;
          break;
        }
      }

      if (iterations > features * features * features * 4096) {
        std::cout
            << "One of the input matrices required too many iterations, we "
               "should regenerate a new matrix"
            << std::endl;
        populateIthA(matrix_index);
        standardizeIthA(matrix_index);
        computeCovarianceIthMatrix(matrix_index);
        matrix_index--;
      } else {
        if (debug)
          std::cout << "QR iteration stopped after " << iterations
                    << " iterations" << std::endl;
        if (debug)
          std::cout << "Eigen values for matrix #" << matrix_index << std::endl;
        for (int k = 0; k < features; k++) {
          eigen_values[k + matrix_index * features] = rq[k + features * k];
          if (debug) std::cout << rq[k + features * k] << " ";
        }
        if (debug) std::cout << std::endl;

        if (debug) {
          std::cout << "Eigen vectors for matrix #" << matrix_index
                    << std::endl;
          for (int row = 0; row < features; row++) {
            for (int col = 0; col < features; col++) {
              std::cout << eigen_vectors[offset + row + features * col] << " ";
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
        }
      }
    }  // end for:matrix_index
  }

};  // class PCA