//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// PrefixSum: this code sample implements the inclusive scan (prefix sum) in parallel. That
// is, given a randomized sequence of numbers x0, x1, x2, ..., xn, this algorithm computes and
// returns a new sequence y0, y1, y2, ..., yn so that
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
// In the above, the notation x_{j}^{i} means the value of the jth element of array x in timestep i.
// Given n processors to perform each iteration of the inner loop in constant time, the algorithm as
// a whole runs in O(log n) time, the number of iterations of the outer loop.
//

#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std; 

void Show(int a[], int arraysize) 
{ 
  for (int i = 0; i < arraysize; ++i) 
  {
    std::cout << a[i] << " ";
    if ((i % 16) == 15) std::cout << "\n";
  }

  std::cout << "\n";
  return;  
} 

int* ParallelPrefixSum(int* prefix1, int* prefix2, unsigned int nb, queue &q)
{
  unsigned int two_power = 1;
  unsigned int num_iter = log2(nb);
  //unsigned int uintmax = UINT_MAX;
  int* result = NULL;
  
  //  std::cout << "uintmax " << uintmax << " " << log2(uintmax) << "\n";
  // Buffer scope
  {
    buffer<int, 1> prefix1_buf(prefix1, range<1>{nb});
    buffer<int, 1> prefix2_buf(prefix2, range<1>{nb});

    // Iterate over the necessary iterations.
    for (unsigned int iter = 0; iter < num_iter; iter++, two_power*=2) {

      // Submit command group for execution
      q.submit([&](handler& h) {
	// Create accessors
	auto prefix1_acc = prefix1_buf.get_access<access::mode::read_write>(h);
	auto prefix2_acc = prefix2_buf.get_access<access::mode::read_write>(h);

	if (iter % 2 == 0) {
	  h.parallel_for(range<1>(nb), [=](id<1> j) {
	    if (j < two_power) {
	      prefix2_acc[j] = prefix1_acc[j];
	    }
	    else {
	      prefix2_acc[j] = prefix1_acc[j] + prefix1_acc[j - two_power];
	    }
	  }); // end parallel for loop in kernel
	  result = prefix2;
	  //std::cout << "return prefix2\n";
	}
	else {
	  h.parallel_for(range<1>(nb), [=](id<1> j) {
	    if (j < two_power) {
	      prefix1_acc[j] = prefix2_acc[j];
	    }
	    else {
	      prefix1_acc[j] = prefix2_acc[j] + prefix2_acc[j - two_power];
	    }
	  }); // end parallel for loop in kernel
	  result = prefix1;
          //std::cout << "return prefix1\n";
	}
      }); // end device queue
    } // end iteration
  } // Buffer scope

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
    //    std::cout << "two_power: " << two_power << "\n";
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
void Usage(std::string prog_name, int exponent) {
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: " << prog_name << " n k \n\n";
  std::cout << " n: Integer exponent presenting the size of the input array. The number of el\
ement in\n";
  std::cout << "    the array must be power of 2 (e.g., 1, 2, 4, ...). Please enter the corre\
sponding\n";
  std::cout << "    exponent betwwen 0 and " << exponent - 1 << ".\n";
  std::cout << " k: Seed used to generate a random sequence.\n";
}

int main(int argc, char* argv[]) {
  unsigned int nb, seed;
  int n, exp_max = log2(std::numeric_limits<int>::max());

 // Read parameters.
  try {
    n = std::stoi(argv[1]);

    // Verify the boundary of acceptance.
    if (n < 0 || n >= exp_max) {
      Usage(argv[0], exp_max);
      return -1;
    }

    seed = std::stoi(argv[2]);
    nb = pow(2, n);
  } catch (...) {
    Usage(argv[0], exp_max);
    return -1;
  }

  std::cout << "\nSequence size: " << nb << ", seed: " << seed;

  int num_iter = log2(nb);
  std::cout << "\nNum iteration: " << num_iter << "\n";

  // Define device selector as 'default'
  default_selector device_selector;

  // exception handler
  auto exception_handler = [](exception_list exceptionList) {
    for (std::exception_ptr const& e : exceptionList) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::terminate();
      }
    }
  };
  
  // Create a device queue using DPC++ class queue
  queue q(device_selector, exception_handler);

  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  int *data = new int[nb];
  int *prefix_sum1 = new int[nb];
  int *prefix_sum2 = new int[nb];
  int *result = NULL;
  
  srand(seed);
  
  // Initialize data arrays
  for (int i = 0; i < nb; i++) {
    data[i] = prefix_sum1[i] = rand() % 10;
    prefix_sum2[i] = 0;
  }

  // Start timer
  auto start = std::chrono::steady_clock::now();

  result = ParallelPrefixSum(prefix_sum1, prefix_sum2, nb, q);

  auto end = std::chrono::steady_clock::now();
  auto timeKern = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Kernel time: " << timeKern << " ms" << "\n";

  //std::cout << "\ndata after transforming using parallel prefix sum result:"; 
  //Show(result, nb);
  
  bool equal = true;
  
  if (result[0] != data[0])
    equal = false;
  else
  {
    for (int i = 1; i < nb; i++) {
      if (result[i] != result[i - 1] + data[i])
      {
	equal = false;
	break;
      }
    }
  }
  
  delete[] data;
  delete[] prefix_sum1;
  delete[] prefix_sum2;  

  if (!equal) {
    std::cout << "\nFailed: " << std::endl;
    return -2;
  }
  else {
    std::cout << "\nSuccess!" << std::endl;
    return 0;
  }
}
