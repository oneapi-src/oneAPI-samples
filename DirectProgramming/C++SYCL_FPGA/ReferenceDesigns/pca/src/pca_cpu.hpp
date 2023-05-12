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
1. Compute the mean of individual features
   u = (F_0, F_1, ..., F_(p-1))

2. Compute the standardized matrix
   B = (A - u) / s
   where s is the standard deviation of each feature column

3. Compute the covariance matrix of size pxp
   C = (1.0/p) * transpose(B) *B

4. Compute Eigen vectors and Eigen values using the QR iteration method

5. Sort the Eigen values and vector based
*/
template <typename T>
class PCA {
 public:
  int samples;                           // number of samples
  int features;                          // number of features
  int matrix_count;                      // number of matrices
  int debug;                             // in debug mode if !=0
  std::vector<T> matrix_a;               // storage for input matrices
  std::vector<T> standardized_matrix_a;  // storage for standardized matrices
  std::vector<T> covariance_matrix;      // storage for covariance matrices
  std::vector<T> eigen_values;           // storage for the Eigen values
  std::vector<T> eigen_vectors;          // storage for the Eigen vectors

 public:
  PCA(int n, int p, int count, int debug = 0);
  void populateA();
  void standardizeA();
  void computeCovarianceMatrix();
  void QRIteration();
};

template <typename T>
PCA<T>::PCA(int n, int p, int count, int d) {
  samples = n;
  features = p;
  matrix_count = count;
  debug = d;

  matrix_a.resize(n * p * matrix_count);
  standardized_matrix_a.resize(n * p * matrix_count);
  covariance_matrix.resize(p * p * matrix_count);

  eigen_values.resize(p * p * matrix_count);
  eigen_vectors.resize(p * matrix_count);
}

// Generate the input matrices with random numbers
template <typename T>
void PCA<T>::populateA() {
  constexpr float kRandomMin = -5;
  constexpr float kRandomMax = 5;

  std::default_random_engine gen;
  std::uniform_real_distribution<float> distribution(kRandomMin, kRandomMax);

  for (int k = 0; k < matrix_a.size(); k++) {
    float value = distribution(gen);
    matrix_a[k] = value;
  }

  if (debug) {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      std::cout << "A matrix #" << matrix_index << std::endl;
      for (int row = 0; row < samples; row++) {
        for (int column = 0; column < features; column++) {
          std::cout << matrix_a[matrix_index * (samples * features) +
                                row * features + column]
                    << " ";
        }
        std::cout << std::endl;
      }
    }
  }
}

// Standardize the A matrices
template <typename T>
void PCA<T>::standardizeA() {
  // The standardized matrix is defined as:
  // standardized_matrix_a[i][j] = (matrix_a[i][j] - mean[j])/(sd[j])
  // where mean[j] is the mean value of the column j and sd[j] is
  // the standard deviation of this column.
  // The standard deviation is defined as
  // sd[j] =  sqrt(sum((a[k][j] - mean[j])^2)/(N-1))

  for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
    if (debug)
      std::cout << "\nStandardizing A matrix #" << matrix_index << std::endl;

    // The current matrix offset in matrix_a
    int offset = matrix_index * samples * features;

    // Compute the mean of each column
    double mean[features];

    if (debug) std::cout << "\nMean of each column: \n";
    for (int column = 0; column < features; column++) {
      mean[column] = 0;
      for (int row = 0; row < samples; row++) {
        mean[column] += matrix_a[offset + row * features + column];
      }
      mean[column] /= samples;
      if (debug) std::cout << mean[column] << " ";
    }
    if (debug) std::cout << std::endl;

    // Compute the standard deviation of each column
    double standard_deviation[features];

    if (debug) std::cout << "\nStandard deviation of each column: \n";
    for (int column = 0; column < features; column++) {
      standard_deviation[column] = 0;
      for (int row = 0; row < samples; row++) {
        standard_deviation[column] +=
            (matrix_a[offset + row * features + column] - mean[column]) *
            (matrix_a[offset + row * features + column] - mean[column]);
      }
      standard_deviation[column] /= (samples - 1);
      standard_deviation[column] = sqrt(standard_deviation[column]);
      if (debug) std::cout << standard_deviation[column] << " ";
    }
    if (debug) std::cout << std::endl;

    // Compute the standardized matrix A
    if (debug) std::cout << "\nStandardized A matrix: \n";
    for (int row = 0; row < samples; row++) {
      for (int column = 0; column < features; column++) {
        standardized_matrix_a[offset + row * features + column] =
            (matrix_a[offset + row * features + column] -
             mean[column]) /
            standard_deviation[column];
        if (debug) std::cout << standardized_matrix_a[offset + row * features + column] << " ";
      }
      if (debug) std::cout << std::endl;
    }
  }
}

// Compute the covariance matrix of the standardized A matrix
template <typename T>
void PCA<T>::computeCovarianceMatrix() {
  // covariance matrix matdA^{T} * matdA
  // this corresponds to matrix order pxp
  if (debug) std::cout << "\nCovariance matrix is: \n";

  for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
    int offsetUA = matrix_index * samples * features;
    int offsetC = matrix_index * features * features;
    for (int i = 0; i < features; i++) {
      for (int j = 0; j < features; j++) {
        covariance_matrix[offsetC + i * features + j] = 0;
        for (int k = 0; k < samples; k++) {
          covariance_matrix[offsetC + i * features + j] +=
              standardized_matrix_a[offsetUA + k * features + i] *
              standardized_matrix_a[offsetUA + k * features + j];
        }
        covariance_matrix[offsetC + i * features + j] =
            (1.0 / (samples)) * covariance_matrix[offsetC + i * features + j];
        if (debug)
          std::cout << covariance_matrix[offsetC + i * features + j] << " ";
      }
      if (debug) std::cout << "\n";
    }
  }
}