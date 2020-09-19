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
#if !defined(WINDOWS)
  cout << "\nUsage: ";
  cout << "./<binary_name> "
       << "-i <Number of Iterations> "
       << "-p <Number of Particles> "
       << "-g <Size of Square Grid> "
       << "-r <Seed for RNG> "
       << "-c <1/0 Flag for CPU Comparison> "
       << "-o <1/0 Flag for Grid Output>\n\n";
#else   // WINDOWS
  cout << "\nUsage: ";
  cout << "./<binary_name> "
       << "<Number of Iterations> "
       << "<Number of Particles> "
       << "<Size of Square Grid> "
       << "<Seed for RNG> "
       << "<1/0 Flag for CPU Comparison> "
       << "<1/0 Flag for Grid Output>\n\n";
#endif  // WINDOWS
}

// Returns true for numeric strings, used for argument parsing
int IsNum(const char* str) {
  for (int i = 0; i < strlen(str); ++i)
    if (!isdigit(str[i])) return 1;
  return 0;
}

// This function checks the two matricies to see if their computations are equal
bool ValidateDeviceComputation(const size_t* grid_device,
                               const size_t* grid_cpu, const size_t grid_size,
                               const size_t planes) {
  for (int c = 0; c < grid_size; ++c)
    for (int r = 0; r < grid_size; ++r)
      for (int d = 0; d < planes; ++d)
        if (grid_device[r + grid_size * c + grid_size * grid_size * d] !=
            grid_cpu[r + grid_size * c + grid_size * grid_size * d])
          return false;
  return true;
}

// Returns false for NxN matrices which do not contain the same values
bool CompareMatrices(const size_t* grid1, const size_t* grid2,
                     const size_t grid_size) {
  for (int c = 0; c < grid_size; ++c)
    for (int r = 0; r < grid_size; ++r)
      if (grid1[r + c * grid_size] != grid2[r + c * grid_size]) return false;
  return true;
}

// This function prints a vector
template <typename T>
void PrintVector(const T* vector, const size_t n) {
  cout << "\n";
  for (size_t i = 0; i < n; ++i) {
    cout << vector[i] << " ";
  }
  cout << "\n";
}

// This function prints a 2D matrix
template <typename T>
void PrintMatrix(const T** matrix, const size_t size_X, const size_t size_Y) {
  cout << "\n";
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      cout << std::setw(3) << matrix[i][j] << " ";
    }
    cout << "\n";
  }
}

// This function prints a 1D vector as a matrix
template <typename T>
void PrintVectorAsMatrix(const T* vector, const size_t size_X,
                         const size_t size_Y) {
  cout << "\n";
  for (size_t j = 0; j < size_X; ++j) {
    for (size_t i = 0; i < size_Y; ++i) {
      cout << std::setw(3) << vector[j * size_Y + i] << " ";
    }
    cout << "\n";
  }
}

#if !defined(WINDOWS)
// Command line argument parser
int parse_cl_args(const int argc, char* argv[], size_t* n_iterations,
                  size_t* n_particles, size_t* grid_size, int* seed,
                  unsigned int* cpu_flag, unsigned int* grid_output_flag) {
  int retv = 0, negative_seed = 0, cl_option;
  // Parse user-specified parameters
  while ((cl_option = getopt(argc, argv, "i:p:g:r:c:o:h")) != -1 && retv == 0) {
    if (optarg) {
      if (cl_option == 'r' && optarg[0] == '-') negative_seed = 1;
      if (negative_seed == 0) retv = IsNum(optarg);
      if (retv == 1) goto usage_label;
    }
    switch (cl_option) {
      case 'i':
        *n_iterations = stoi(optarg);
        break;
      case 'p':
        *n_particles = stoi(optarg);
        break;
      case 'g':
        *grid_size = stoi(optarg);
        break;
      case 'r':
        *seed = stoi(optarg);
        break;
      case 'c':
        *cpu_flag = stoul(optarg);
        break;
      case 'o':
        *grid_output_flag = stoul(optarg);
        break;
      case 'h':
        cout << "Particle Diffusion DPC++ code sample help message:\n";
      case ':':
      case '?':
      default:
      usage_label : {
        Usage(argv[0]);
        break;
      }
    }
  }
  return retv;
}
#else   // WINDOWS
// Windows command line argument parser
int parse_cl_args_windows(char* argv[], size_t* n_iterations,
                          size_t* n_particles, size_t* grid_size, int* seed,
                          unsigned int* cpu_flag,
                          unsigned int* grid_output_flag) {
  // Parse user-specified parameters
  try {
    *n_iterations = stoi(argv[1]);
    *n_particles = stoi(argv[2]);
    *grid_size = stoi(argv[3]);
    *seed = stoi(argv[4]);
    *cpu_flag = stoul(argv[5]);
    *grid_output_flag = stoul(argv[6]);
  } catch (...) {
    Usage(argv[0]);
    return 1;
  }
  return 0;
}
#endif  // WINDOWS

// Prints the 3D grids holding the simulation results
void print_grids(const size_t* grid, const size_t* grid_cpu,
                 const size_t grid_size, const unsigned int cpu_flag,
                 const unsigned int grid_output_flag) {
  // Index variables for 3rd dimension of grid
  size_t layer = 0;
  const size_t gs2 = grid_size * grid_size;

  // Displays final grid only if grid small
  if (grid_size <= 45 && grid_output_flag == 1) {
    cout << "\n**********************************************************\n";
    cout << "*                           DEVICE                       *\n";
    cout << "**********************************************************\n";

    cout << "\n ********************** FULL GRID: **********************\n";

    // Counter 1 layer of grid (0 * grid_size * grid_size)
    layer = 0;
    PrintVectorAsMatrix<size_t>(&grid[layer], grid_size, grid_size);

    cout << "\n ***************** FINAL SNAPSHOT: *****************\n";

    // Counter 2 layer of grid (1 * grid_size * grid_size)
    layer = gs2;
    PrintVectorAsMatrix<size_t>(&grid[layer], grid_size, grid_size);

    cout << "\n ************* NUMBER OF PARTICLE ENTRIES: ************* \n";

    // Counter 3 layer of grid (2 * grid_size * grid_size)
    layer = gs2 + gs2;
    PrintVectorAsMatrix<size_t>(&grid[layer], grid_size, grid_size);

    cout << "**********************************************************\n";
    cout << "*                        END DEVICE                      *\n";
    cout << "**********************************************************\n\n\n";
  }

  if (cpu_flag == 1) {
    // Displays final grid on cpu only if grid small
    if (grid_size <= 45 && grid_output_flag == 1) {
      cout << "\n";
      cout << "**********************************************************\n";
      cout << "*                           CPU                          *\n";
      cout << "**********************************************************\n";

      cout << "\n ********************** FULL GRID: **********************\n";

      // Counter 1 layer of grid (0 * grid_size * grid_size)
      layer = 0;
      PrintVectorAsMatrix<size_t>(&grid_cpu[layer], grid_size, grid_size);

      cout << "\n ***************** FINAL SNAPSHOT: *****************\n";

      // Counter 2 layer of grid (1 * grid_size * grid_size)
      layer = gs2;
      PrintVectorAsMatrix<size_t>(&grid_cpu[layer], grid_size, grid_size);

      cout << "\n ************* NUMBER OF PARTICLE ENTRIES: ************* \n";

      // Counter 3 layer of grid (2 * grid_size * grid_size)
      layer = gs2 + gs2;
      PrintVectorAsMatrix<size_t>(&grid_cpu[layer], grid_size, grid_size);

      cout << "**********************************************************\n";
      cout << "*                          END CPU                       *\n";
      cout << "**********************************************************\n";
      cout << "\n\n";
    }
  }
}

// Compares the matrices generated by the CPU and device and prints results.
void print_validation_results(const size_t* grid, const size_t* grid_cpu,
                              const size_t grid_size, const size_t planes,
                              const unsigned int cpu_flag,
                              const unsigned int grid_output_flag) {
  // Index variables for 3rd dimension of grid
  size_t layer = 0;
  const size_t gs2 = grid_size * grid_size;
  bool retv = false;

  /* ********** Counter 1: device v.s host comparison ********** */

  // Counter 1 layer of grid (0 * grid_size * grid_size)
  layer = 0;
  retv = CompareMatrices(&grid[layer], &grid_cpu[layer], grid_size);

  if (!retv) {
    cout << "Device Counter 1 != CPU Counter 1\n";
  }
  if (retv) {
    cout << "Device Counter 1 == CPU Counter 1\n";
  }

  /* ********** Counter 2: device v.s host comparison ********** */

  // Counter 2 layer of grid (1 * grid_size * grid_size)
  layer = gs2;
  retv = CompareMatrices(&grid[layer], &grid_cpu[layer], grid_size);

  if (!retv) {
    cout << "Device Counter 2 != CPU Counter 2\n";
  }
  if (retv) {
    cout << "Device Counter 2 == CPU Counter 2\n";
  }

  /* ********** Counter 3: device v.s host comparison ********** */

  // Counter 3 layer of grid (2 * grid_size * grid_size)
  layer = gs2 + gs2;
  retv = CompareMatrices(&grid[layer], &grid_cpu[layer], grid_size);

  if (!retv) {
    cout << "Device Counter 3 != CPU Counter 3\n";
  }
  if (retv) {
    cout << "Device Counter 3 == CPU Counter 3\n";
  }

  retv = ValidateDeviceComputation(grid, grid_cpu, grid_size, planes);

  if (!retv)
    cout << "ERROR cpu computation does not match that of the device\n";
  if (retv) cout << "Success.\n";
}
