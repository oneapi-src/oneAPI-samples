//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// utils: Intel® oneAPI DPC++ utility functions
// used in motionsim.cpp and motionsim_kernel.cpp
//

// This function displays correct usage and parameters
void Usage(string program_name) {
  if (!WINDOWS) {
    cout << "\nUsage: ";
    cout << "./<binary_name> "
         << "-i <Number of Iterations> "
         << "-p <Number of Particles> "
         << "-g <Size of Square Grid> "
         << "-r <Seed for RNG> "
         << "-c <1/0 Flag for CPU Comparison> "
         << "-o <1/0 Flag for Grid Output>\n\n";
  } else {
    cout << "\nUsage: ";
    cout << "./<binary_name> "
         << "<Number of Iterations> "
         << "<Number of Particles> "
         << "<Size of Square Grid> "
         << "<Seed for RNG> "
         << "<1/0 Flag for CPU Comparison> "
         << "<1/0 Flag for Grid Output>\n\n";
  }
}

// Returns true for numeric strings, used for argument parsing
bool IsNum(const char* str) {
  for (int i = 0; i < strlen(str); ++i)
    if (!isdigit(str[i])) return false;
  return true;
}

// This function prints a 1D vector as a matrix
template <typename T>
void PrintVectorAsMatrix(T* vector, size_t size_X, size_t size_Y) {
  cout << "\n";
  for (size_t j = 0; j < size_X; ++j) {
    for (size_t i = 0; i < size_Y; ++i) {
      cout << std::setw(3) << vector[j * size_Y + i] << " ";
    }
    cout << "\n";
  }
}

// This function prints a 2D matrix
template <typename T>
void PrintMatrix(T** matrix, size_t size_X, size_t size_Y) {
  cout << "\n";
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      cout << std::setw(3) << matrix[i][j] << " ";
    }
    cout << "\n";
  }
}

// This function prints a vector
template <typename T>
void PrintVector(T* vector, size_t n) {
  cout << "\n";
  for (size_t i = 0; i < n; ++i) {
    cout << vector[i] << " ";
  }
  cout << "\n";
}

// This function checks the two matricies to see if their computations are equal
bool ValidateDeviceComputation(size_t* grid_device, size_t* grid_cpu,
                               size_t grid_size, const size_t planes) {
  for (int c = 0; c < grid_size; ++c)
    for (int r = 0; r < grid_size; ++r)
      for (int d = 0; d < planes; ++d)
        if (grid_device[r + grid_size * c + grid_size * grid_size * d] !=
            grid_cpu[r + grid_size * c + grid_size * grid_size * d])
          return false;
  return true;
}

// Returns false for NxN matrices which do not contain the same values
bool CompareMatrices(size_t* grid1, size_t* grid2, size_t grid_size) {
  for (int c = 0; c < grid_size; ++c)
    for (int r = 0; r < grid_size; ++r)
      if (grid1[r + c * grid_size] != grid2[r + c * grid_size]) return false;
  return true;
}
