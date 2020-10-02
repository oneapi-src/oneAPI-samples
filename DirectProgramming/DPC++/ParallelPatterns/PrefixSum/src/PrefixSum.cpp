//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// PrefixSum: this code sample implements the inclusive scan (prefix sum) in
// parallel. That is, given a randomized sequence of numbers x0, x1, x2, ...,
// xn, this algorithm computes and returns a new sequence y0, y1, y2, ..., yn so
// that
//
// y0 = x0
// y1 = x0 + x1
// y2 = x0 + x1 + x2
// .....
// yn = x0 + x1 + x2 + ... + xn
//
// Below is the pseudo code for computing prefix sum in parallel:
//
// n is power of 2 (1, 2, 4 , 8, 16, ...):
//
// for i from 0 to  [log2 n] - 1 do
//   for j from 0 to (n-1) do in parallel
//     if j<2^i then
//       x_{j}^{i+1} <- x_{j}^{i}}
//     else
//       x_{j}^{i+1} <- x_{j}^{i} + x_{j-2^{i}}^{i}}
//
// In the above, the notation x_{j}^{i} means the value of the jth element of
// array x in timestep i. Given n processors to perform each iteration of the
// inner loop in constant time, the algorithm as a whole runs in O(log n) time,
// the number of iterations of the outer loop.
//

#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

void Show(int a[], int arraysize) {
  for (int i = 0; i < arraysize; ++i) {
    cout << a[i] << " ";
    if ((i % 16) == 15) cout << "\n";
  }

  cout << "\n";
  return;
}

int* ParallelPrefixSum(int* current, int* next, unsigned int nb, queue& q) {
  unsigned int two_power = 1;
  unsigned int num_iter = log2(nb);
  // unsigned int uintmax = UINT_MAX;
  int* result = NULL;

  //  cout << "uintmax " << uintmax << " " << log2(uintmax) << "\n";
  // Buffer scope
  {
    buffer sequence_buf(current, range(nb));
    buffer sequence_next_buf(next, range(nb));

    // Iterate over the necessary iterations.
    for (unsigned int iter = 0; iter < num_iter; iter++, two_power *= 2) {
      // Submit command group for execution
      q.submit([&](auto& h) {
        // Create accessors
        accessor sequence(sequence_buf, h);
        accessor sequence_next(sequence_next_buf, h);

        if (iter % 2 == 0) {
          h.parallel_for(nb, [=](id<1> j) {
            if (j < two_power) {
              sequence_next[j] = sequence[j];
            } else {
              sequence_next[j] = sequence[j] + sequence[j - two_power];
            }
          });  // end parallel for loop in kernel
          result = next;
        } else {
          h.parallel_for(nb, [=](id<1> j) {
            if (j < two_power) {
              sequence[j] = sequence_next[j];
            } else {
              sequence[j] = sequence_next[j] + sequence_next[j - two_power];
            }
          });  // end parallel for loop in kernel
          result = current;
        }
      });  // end device queue
    }      // end iteration
  }        // Buffer scope

  // Wait for commands to complete. Enforce synchronization on the command queue
  q.wait_and_throw();

  return result;
}
/*
void PrefixSum(int* x, unsigned int nb)
{
  unsigned int two_power = 1;
  unsigned int num_iter = log2(nb);
  int temp = 0;

  // Iterate over the necessary iterations
  for (unsigned int iter = 0; iter < num_iter; iter++, two_power*=2) {
    //Show(x, nb);
    //    cout << "two_power: " << two_power << "\n";
    for (unsigned int j = nb; j > 0; j--) {
      if (j < two_power) {
        x[j] = x[j];
      }
      else {
        x[j] = x[j] + x[j - two_power];
      }
    }
  }
}
*/
void Usage(string prog_name, int exponent) {
  cout << " Incorrect parameters\n";
  cout << " Usage: " << prog_name << " n k \n\n";
  cout << " n: Integer exponent presenting the size of the input array.\n";
  cout << "    The number of element in the array must be power of 2\n";
  cout << "    (e.g., 1, 2, 4, ...). Please enter the corresponding exponent\n";
  cout << "    betwwen 0 and " << exponent - 1 << ".\n";
  cout << " k: Seed used to generate a random sequence.\n";
}

int main(int argc, char* argv[]) {
  unsigned int nb, seed;
  int n, exp_max = log2(numeric_limits<int>::max());

  // Read parameters.
  try {
    n = stoi(argv[1]);

    // Verify the boundary of acceptance.
    if (n < 0 || n >= exp_max) {
      Usage(argv[0], exp_max);
      return -1;
    }

    seed = stoi(argv[2]);
    nb = pow(2, n);
  } catch (...) {
    Usage(argv[0], exp_max);
    return -1;
  }

  cout << "\nSequence size: " << nb << ", seed: " << seed;

  int num_iter = log2(nb);
  cout << "\nNum iteration: " << num_iter << "\n";

  // Define device selector as 'default'
  default_selector device_selector;

  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler);

  cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  int* data = new int[nb];
  int* prefix_sum1 = new int[nb];
  int* prefix_sum2 = new int[nb];
  int* result = NULL;

  srand(seed);

  // Initialize data arrays
  for (int i = 0; i < nb; i++) {
    data[i] = prefix_sum1[i] = rand() % 10;
    prefix_sum2[i] = 0;
  }

  // Start timer
  dpc_common::TimeInterval t;

  result = ParallelPrefixSum(prefix_sum1, prefix_sum2, nb, q);

  auto elapsed_time = t.Elapsed();

  cout << "Elapsed time: " << elapsed_time << " s\n";

  // cout << "\ndata after transforming using parallel prefix sum result:";
  // Show(result, nb);

  bool equal = true;

  if (result[0] != data[0])
    equal = false;
  else {
    for (int i = 1; i < nb; i++) {
      if (result[i] != result[i - 1] + data[i]) {
        equal = false;
        break;
      }
    }
  }

  delete[] data;
  delete[] prefix_sum1;
  delete[] prefix_sum2;

  if (!equal) {
    cout << "\nFailed: " << std::endl;
    return -2;
  } else {
    cout << "\nSuccess!" << std::endl;
    return 0;
  }
}
