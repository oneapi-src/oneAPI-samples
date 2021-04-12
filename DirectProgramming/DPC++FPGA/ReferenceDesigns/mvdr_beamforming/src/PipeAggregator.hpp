//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#ifndef __PIPE_AGGREGATOR_HPP__
#define __PIPE_AGGREGATOR_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <array>

#include "pipe_array.hpp"
#include "UnrolledLoop.hpp"

// PipeAggregator
// Translate a pipe_array where each element transacts a std::array of type
// T per transaction into what appears as a single pipe which transacts a
// single std::array of type T.
template <
  typename  PipeArrayType,              // 1D pipe_array class
                                        // Each pipe in the array must transact
                                        // k_elements_per_array_pipe elelements
                                        // of type T wrapped in std::array
  typename  T,                          // type of individual elements
  size_t    k_elements_per_array_pipe,  // number of T elements read/written
                                        // to each sub-pipe in PipeArrayType
  size_t    k_num_pipes                 // number of pipes in PipeArrayType
> struct PipeAggregator {

  using SmallPipeElement = std::array< T, k_elements_per_array_pipe >;
  using LargePipeElement = 
    std::array< T, k_elements_per_array_pipe * k_num_pipes >;

  // disable constructor, copy constructor and operator=
  PipeAggregator() = delete;
  PipeAggregator(const PipeAggregator &) = delete;
  PipeAggregator& operator=(PipeAggregator const &) = delete;

  // Non-blocking write
  // success = true if ALL individual pipe writes were successful, otherwise
  // success = false.
  static void write( const LargePipeElement &data, bool &success ) {
    bool small_pipe_success[k_num_pipes];
    bool large_pipe_success = true;

    UnrolledLoop<k_num_pipes>([&](auto i) {
      SmallPipeElement pipe_data;

      UnrolledLoop<k_elements_per_array_pipe>([&](auto j) {
        pipe_data[j] = data[i*k_elements_per_array_pipe + j];
      });

      PipeArrayType::template PipeAt<i>::write( 
        pipe_data, small_pipe_success[i] );
    });

    UnrolledLoop<k_num_pipes>([&](auto i) {
      large_pipe_success &= small_pipe_success[i];
    });
    
    success = large_pipe_success;
  }

  // Blocking write
  // Return after all individual pipe writes have completed
  static void write( const LargePipeElement &data ) {
    UnrolledLoop<k_num_pipes>([&](auto i) {
      SmallPipeElement pipe_data;
      
      UnrolledLoop<k_elements_per_array_pipe>([&](auto j) {
        pipe_data[j] = data[i*k_elements_per_array_pipe + j];
      });

      PipeArrayType::template PipeAt<i>::write( pipe_data );
    });
  }

  // Non-blocking read
  // success = true if ALL individual pipe reads were successful, otherwise
  // success = false.
  static LargePipeElement read( bool &success ) {
    LargePipeElement data;
    bool small_pipe_success[k_num_pipes];
    bool large_pipe_success = true;

    UnrolledLoop<k_num_pipes>([&](auto i) {
      SmallPipeElement pipe_data;
      pipe_data = PipeArrayType::template PipeAt<i>::read( 
        small_pipe_success[i] );
      UnrolledLoop<k_elements_per_array_pipe>([&](auto j) {
        data[i*k_elements_per_array_pipe + j] = pipe_data[j];
      });
    });
    
    UnrolledLoop<k_num_pipes>([&](auto i) {
      large_pipe_success &= small_pipe_success[i];
    });

    success = large_pipe_success;
    return( data );
  }

  // Blocking read
  // Perform a blocking read in each pipe in the pipe array
  static LargePipeElement read() {
    LargePipeElement data;
    
    UnrolledLoop<k_num_pipes>([&](auto i) {
      SmallPipeElement pipe_data;
      pipe_data = PipeArrayType::template PipeAt<i>::read();
      UnrolledLoop<k_elements_per_array_pipe>([&](auto j) {
        data[i*k_elements_per_array_pipe + j] = pipe_data[j];
      });
    });

    return( data );
  }

};  // end of struct PipeAggregator

#endif  // ifndef __PIPE_AGGREGATOR_HPP__
