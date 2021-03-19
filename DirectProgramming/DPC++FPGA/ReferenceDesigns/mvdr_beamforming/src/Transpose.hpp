#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "Tuple.hpp"

using namespace sycl;

// SubmitTransposeKernel
// Accept k_pipe_width elements of type T wrapped in NTuple
// on each read from DataInPipe.  Store elements locally until we can write
// to the output array of pipes with data in transposed order.
// For simplicity of terminology, incoming data is defined to enter in 'row' 
// order and exit in 'column' order, although for the purposes of the transpose
// the terms 'row' and 'column' could be interchanged.
// Row order means all data from a given row is received before any data from
// the next row is received.
// This kernel also performs flow control and ensures that no partial matrices
// will be written into the output pipe.
template <
  typename  TransposeKernelName,      // Name to use for the Kernel
  typename  T,                        // type of element to transpose
  size_t    k_num_cols_in,            // number of columns in the input matrix
  size_t    k_pipe_width,             // number of elements read/written
                                      // (wrapped in NTuple) from/to pipes
  typename  MatrixInPipe,             // Receive the input matrix in row order
                                      // Receive k_pipe_width elements of type
                                      // T wrapped in NTuple on each read
  typename  MatrixOutPipe             // Send the output matrix in column order.
                                      // Send k_pipe_width elements of type T
                                      // wrapped in NTuple on each write.
>
event SubmitTransposeKernel( queue& q ) {

  // Template parameter checking
  static_assert(std::numeric_limits<short>::max() > k_num_cols_in,
    "k_num_cols_in must fit in a short" );
  static_assert( k_num_cols_in % k_pipe_width == 0,
    "k_num_cols_in must be evenly divisible by k_pipe_width" );

  using PipeType = NTuple< T, k_pipe_width >;

  auto e = q.submit([&](handler& h) {

    h.single_task< TransposeKernelName > ( [=] {

      while( 1 ) {

        // TODO
        // for now, DO NOT do the transpose, just accept the data already in
        // column order and pass it through
        for ( short col = 0; col < k_num_cols_in; col++ ) {
          PipeType data_in = MatrixInPipe::read();
          MatrixOutPipe::write( data_in );
        }

      }   // end of while( 1 ) 

    });   // end of h.single_task

  });   // end of q.submit

  return e;

}   // end of SubmitTransposeKernel()

#endif  // ifndef __TRANSPOSE_HPP__
