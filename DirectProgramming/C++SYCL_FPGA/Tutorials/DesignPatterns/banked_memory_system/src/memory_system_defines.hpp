#pragma once

#include <array>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

#include "constexpr_math.hpp"

using namespace sycl;

// Constants for the simple kernels.
constexpr size_t kNumRows = 5;
constexpr size_t kNumCols = 500;

constexpr size_t kNumRowsOptimized = 500;

using SimpleInputT = int;
using SimpleOutputT = std::array<int, 5>;

/////////////////////////////////////////////
// Define input/output streaming interfaces
/////////////////////////////////////////////
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>));

// Interfaces for the illustrative simple kernels.
class ID_InStream_NaiveKernel;
using InStream_NaiveKernel =
    sycl::ext::intel::experimental::pipe<ID_InStream_NaiveKernel, SimpleInputT, 0,
                                         PipePropertiesT>;

class ID_OutStream_NaiveKernel;
using OutStream_NaiveKernel =
    sycl::ext::intel::experimental::pipe<ID_OutStream_NaiveKernel, SimpleOutputT, 0,
                                         PipePropertiesT>;

class ID_InStream_OptKernel;
using InStream_OptKernel =
    sycl::ext::intel::experimental::pipe<ID_InStream_OptKernel, SimpleInputT, 0,
                                         PipePropertiesT>;
class ID_OutStream_OptKernel;
using OutStream_OptKernel =
    sycl::ext::intel::experimental::pipe<ID_OutStream_OptKernel, SimpleOutputT, 0,
                                         PipePropertiesT>;
