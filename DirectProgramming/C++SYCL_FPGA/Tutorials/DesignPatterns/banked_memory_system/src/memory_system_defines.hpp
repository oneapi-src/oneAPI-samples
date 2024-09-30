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
class IDInStreamNaiveKernel;
using InStreamNaiveKernel =
    sycl::ext::intel::experimental::pipe<IDInStreamNaiveKernel, SimpleInputT, 0,
                                         PipePropertiesT>;

class IDOutStreamNaiveKernel;
using OutStreamNaiveKernel =
    sycl::ext::intel::experimental::pipe<IDOutStreamNaiveKernel, SimpleOutputT, 0,
                                         PipePropertiesT>;

class IDInStreamOptKernel;
using InStreamOptKernel =
    sycl::ext::intel::experimental::pipe<IDInStreamOptKernel, SimpleInputT, 0,
                                         PipePropertiesT>;
class IDOutStreamOptKernel;
using OutStreamOptKernel =
    sycl::ext::intel::experimental::pipe<IDOutStreamOptKernel, SimpleOutputT, 0,
                                         PipePropertiesT>;
