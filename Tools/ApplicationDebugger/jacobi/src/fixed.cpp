//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// This file is included only into jacobi-fixed program.
// This file contains methods where actual computation takes place.
// Here we explain and show how to fix Bugs 1, 2, and 3.


// Compute x_k1 and write the result to its accessor.

void compute_x_k1_kernel (id<1> &index, float *b, 
                          float *x_k, float *x_k1) {
  // Current index.
  int i = index[0];

  // The vector x_k1 should be computed as:
  //     x_k1 = D^{-1}(b - (A - D)x_k),
  // where A is our matrix, D is its diagonal, b is right hand
  // side vector, and x_k is the result of the previous iteration.
  //
  // Matrices (A - D) and D are hardcoded as:
  //     (A - D) is a stencil matrix [1 1 0 1 1];
  //     D is a diagonal matrix with all elements equal to 5.

  float result = b[i];

  // Non-diagonal elements of matrix A are all 1s, so to substract
  // i-th element of (A - D)x_k, we need to substract the sum of elements
  // of x_k with indices i - 2, i - 1, i + 1, i + 2. We do not substract
  // the i-th element, as it gets multiplied by 0 in (A - D)x_k.

  // Fix Bug 1: out-of-bounds access: all indices below can trigger
  // out-of-bounds access and thus garbage values will be read.
  // Fix it by adding checks that the index exists:
  if (i > 1)
    result -= x_k[i - 2];
  if (i > 0)
    result -= x_k[i - 1];
  if (i < n - 1)
    result -= x_k[i + 1];
  if (i < n - 2)
    result -= x_k[i + 2];

  // In our case the diagonal matrix has only 5s on the diagonal, so
  // division by 5 gives us its invert.
  result /= 5;

  // Save the value to the output buffer.
  x_k1[i] = result;
}

// Submits the kernel which updates x_k1 at every iteration.

void compute_x_k1 (queue &q, buffer_args &buffers) {
  q.submit([&](auto &h) {
    accessor acc_b(buffers.b, h, read_only);
    accessor acc_x_k(buffers.x_k, h, read_only);
    accessor acc_x_k1(buffers.x_k1, h, write_only);

    h.parallel_for(range{n}, [=](id<1> index) {
      compute_x_k1_kernel (index, acc_b.get_pointer(), acc_x_k.get_pointer(), 
                           acc_x_k1.get_pointer());
    });
  });
}

// Here we compute values which are used for relative error computation
// and copy the vector x_k1 over the vector x_k.

void prepare_for_next_iteration (queue &q, buffer_args &buffers) {
  constexpr size_t l = 16;

  q.submit([&](auto &h) {
    accessor acc_abs_error(buffers.abs_error, h, read_write);
    accessor acc_x_k(buffers.x_k, h, read_write);
    accessor acc_x_k1(buffers.x_k1, h, read_only);

    // To compute the relative error we need to prepare two values:
    // absolute error and L1-norm of x_k1.
    // Absolute error is computed as L1-norm of (x_k - x_k1).
    // To compute the L1-norms of x_k1 and (x_k - x_k1) vectors 
    // we use the reduction API with std::plus operator.
    auto r_abs_error = reduction(buffers.abs_error, h, std::plus<>());
    auto r_l1_norm_x_k1 = reduction(buffers.l1_norm_x_k1, h, std::plus<>());

    h.parallel_for(nd_range<1>{n, l},
                   r_abs_error, r_l1_norm_x_k1,
                   [=](nd_item<1> index, auto &abs_error, auto &l1_norm_x_k1) {
      auto gid = index.get_global_id();
      float x_k = acc_x_k[gid];
      float x_k1 = acc_x_k1[gid];

      // Execute reduction sums.
      float local_abs_error = abs(x_k - x_k1);
      abs_error += local_abs_error; // Bug 2 challenge: breakpoint here.
      l1_norm_x_k1 += abs(x_k1);

      // Copy the vector x_k1 over x_k.
      acc_x_k[gid] = x_k1;
    });
  });
}

// Iterate until the algorithm converges (success) or the maximum number 
// of iterations is reached (fail).

int iterate(queue &q, float *b, float *x_k, float *x_k1, float &rel_error) {
  // Absolute error, ||x_k - x_k1||_1, L1-norm of (x_k - x_k1).
  float abs_error = 0;
  // ||x_k1||_1, L1-norm of x_k1.
  float l1_norm_x_k1 = 0;

  int k = 0;

  // Jacobi iteration begins.
  do {// k-th iteration of Jacobi.
    // Fix bug 2: we have to reset the error values at each iteration, otherwise
    // the relative error accumulates through iterations and does not fall
    // below the tolerance.
    abs_error = 0;
    l1_norm_x_k1 = 0;

    { // Fix bug 3: the host values of abs_error and l1_norm_x_k1
      // were not synchronised with their new values on device.
      // Open new scope for buffers. Once the scope is ended, the destructors
      // of buffers will write the data back from device to host.

      // Create SYCL buffers.
      buffer_args buffers {b, x_k, x_k1, &l1_norm_x_k1, &abs_error};

      compute_x_k1(q, buffers);
      prepare_for_next_iteration(q, buffers);
    }

    // Compute relative error based on reduced values from this iteration.
    rel_error = abs_error / (l1_norm_x_k1 + 1e-32);

    if (abs_error < 0 || l1_norm_x_k1 < 0
        || (abs_error + l1_norm_x_k1) < 1e-32) {
      cout << "\nfail; Bug 3. Fix it on GPU. The relative error has invalid value "
           << "after iteration " << k << ".\n"
           << "Hint 1: inspect reduced error values. With the challenge scenario\n"
           << "    from bug 2 you can verify that reduction algorithms compute\n"
           << "    the correct values inside kernel on GPU. Take into account\n"
           << "    SIMD lanes: on GPU each thread processes several work items\n"
           << "    at once, so you need to modify your commands and update\n"
           << "    the convenience variable for each SIMD lane, e.g. using \n"
	   << "    `thread apply :*`.\n"
           << "Hint 2: why don't we get the correct values at the host part of\n"
           << "    the application?\n";
      exit(0);
    }

    // Periodically print out how the algorithm behaves.
    if (k % 20 == 0) {
      std::cout << "Iteration " << k << ", relative error = "
                << rel_error << "\n";
    }

    k++;
  } while (rel_error > tolerance && k < max_number_of_iterations);
  // Jacobi iteration ends.

  return k;
}

