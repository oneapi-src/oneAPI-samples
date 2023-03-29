//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// matrix multiply routines
#include "multiply.hpp"

#include <CL/sycl.hpp>
#include <array>

using namespace std;

template <typename T> class Matrix1;

template <typename T> class Matrix1_1;

template <typename T> class Matrix1_2;

// Basic matrix multiply
void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM],
               TYPE c[][NUM], TYPE t[][NUM]) {
  int i, j, k;

  // Declare a deviceQueue
  sycl::queue q(sycl::default_selector_v, exception_handler);
  cout << "Running on " << q.get_device().get_info<sycl::info::device::name>()
       << "\n";
  // Declare a 2 dimensional range
  sycl::range<2> matrix_range{NUM, NUM};

  // Declare 3 buffers and Initialize them
  sycl::buffer<TYPE, 2> bufferA((TYPE *)a, matrix_range);
  sycl::buffer<TYPE, 2> bufferB((TYPE *)b, matrix_range);
  sycl::buffer<TYPE, 2> bufferC((TYPE *)c, matrix_range);
  // Submit our job to the queue
  q.submit([&](auto &h) {
     // Declare 3 accessors to our buffers. The first 2 read and the last
     // read_write
     sycl::accessor accessorA(bufferA, h, sycl::read_only);
     sycl::accessor accessorB(bufferB, h, sycl::read_only);
     sycl::accessor accessorC(bufferC, h);

     // Execute matrix multiply in parallel over our matrix_range
     // ind is an index into this range
     h.parallel_for(matrix_range, [=](sycl::id<2> ind) {
       int k;
       for (k = 0; k < NUM; k++) {
         // Perform computation ind[0] is row, ind[1] is col
         accessorC[ind[0]][ind[1]] +=
             accessorA[ind[0]][k] * accessorB[k][ind[1]];
       }
     });
   }).wait_and_throw();
} // multiply1

// Replaces accessorC reference with a local variable
void multiply1_1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM],
                 TYPE c[][NUM], TYPE t[][NUM]) {

  int i, j, k;

  // Declare a deviceQueue
  sycl::queue q(sycl::default_selector_v, exception_handler);
  cout << "Running on " << q.get_device().get_info<sycl::info::device::name>()
       << "\n";

  // Declare a 2 dimensional range
  sycl::range<2> matrix_range{NUM, NUM};

  // Declare 3 buffers and Initialize them
  sycl::buffer<TYPE, 2> bufferA((TYPE *)a, matrix_range);
  sycl::buffer<TYPE, 2> bufferB((TYPE *)b, matrix_range);
  sycl::buffer<TYPE, 2> bufferC((TYPE *)c, matrix_range);

  // Submit our job to the queue
  q.submit([&](auto &h) {
     // Declare 3 accessors to our buffers. The first 2 read and the last
     // read_write
     sycl::accessor accessorA(bufferA, h, sycl::read_only);
     sycl::accessor accessorB(bufferB, h, sycl::read_only);
     sycl::accessor accessorC(bufferC, h);

     // Execute matrix multiply in parallel over our matrix_range
     // ind is an index into this range
     h.parallel_for(matrix_range, [=](sycl::id<2> ind) {
       int k;
       TYPE acc = 0.0;
       for (k = 0; k < NUM; k++) {
         // Perform computation ind[0] is row, ind[1] is col
         acc += accessorA[ind[0]][k] * accessorB[k][ind[1]];
       }
       accessorC[ind[0]][ind[1]] = acc;
     });
   }).wait_and_throw();
}

// Replaces accessorC reference with a local variable and adds matrix tiling
void multiply1_2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM],
                 TYPE c[][NUM], TYPE t[][NUM]) {
  int i, j, k;

  // Declare a deviceQueue
  sycl::queue q(sycl::default_selector_v, exception_handler);
  cout << "Running on " << q.get_device().get_info<sycl::info::device::name>()
       << "\n";

  // Declare a 2 dimensional range
  sycl::range<2> matrix_range{NUM, NUM};
  sycl::range<2> tile_range{MATRIXTILESIZE, MATRIXTILESIZE};

  // Declare 3 buffers and Initialize them
  sycl::buffer<TYPE, 2> bufferA((TYPE *)a, matrix_range);
  sycl::buffer<TYPE, 2> bufferB((TYPE *)b, matrix_range);
  sycl::buffer<TYPE, 2> bufferC((TYPE *)c, matrix_range);

  // Submit our job to the queue
  q.submit([&](auto &h) {
     // Declare 3 accessors to our buffers. The first 2 read and the last
     // read_write
     sycl::accessor accessorA(bufferA, h, sycl::read_only);
     sycl::accessor accessorB(bufferB, h, sycl::read_only);
     sycl::accessor accessorC(bufferC, h);

     // Create matrix tiles
     sycl::local_accessor<TYPE, 2> aTile(
         sycl::range<2>(MATRIXTILESIZE, MATRIXTILESIZE), h);
     sycl::local_accessor<TYPE, 2> bTile(
         sycl::range<2>(MATRIXTILESIZE, MATRIXTILESIZE), h);
     // Execute matrix multiply in parallel over our matrix_range
     // ind is an index into this range
     h.parallel_for(sycl::nd_range<2>(matrix_range, tile_range),
                    [=](cl::sycl::nd_item<2> it) {
                      int k;
                      const int numTiles = NUM / MATRIXTILESIZE;
                      const int row = it.get_local_id(0);
                      const int col = it.get_local_id(1);
                      const int globalRow =
                          MATRIXTILESIZE * it.get_group(0) + row;
                      const int globalCol =
                          MATRIXTILESIZE * it.get_group(1) + col;
                      TYPE acc = 0.0;
                      for (int t = 0; t < numTiles; t++) {
                        const int tiledRow = MATRIXTILESIZE * t + row;
                        const int tiledCol = MATRIXTILESIZE * t + col;
                        aTile[row][col] = accessorA[globalRow][tiledCol];
                        bTile[row][col] = accessorB[tiledRow][globalCol];
                        it.barrier(sycl::access::fence_space::local_space);
                        for (k = 0; k < MATRIXTILESIZE; k++) {
                          // Perform computation ind[0] is row, ind[1] is col
                          acc += aTile[row][k] * bTile[k][col];
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                      }
                      accessorC[globalRow][globalCol] = acc;
                    });
   }).wait_and_throw();
} // multiply1_2

void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM],
                      TYPE t[][NUM]) {
  int NTHREADS = MAXTHREADS;
  int MSIZE = NUM;

  MULTIPLY(MSIZE, NTHREADS, 0, a, b, c, t);
}
