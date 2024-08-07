#pragma once

#include <array>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

#include "constexpr_math.hpp"

using namespace sycl;

// size of the line buffer that holds the image pixels.
#define LB_SZ 1024

// size of the filter window.
#define WINDOW_SZ 5

// input image size.
#define IMG_SZ 400

using pixel_t = unsigned int;

// Definition of input and output pipes.
using PixelPipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>));

class ID_InStream;
using InputPixelStream =
    sycl::ext::intel::experimental::pipe<ID_InStream, pixel_t, 0,
                                         PixelPipePropertiesT>;

class ID_OutStream;
using OutputPixelStream =
    sycl::ext::intel::experimental::pipe<ID_OutStream, pixel_t, 0,
                                         PixelPipePropertiesT>;


class ID_BoxFilter;


using SimplePipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
  sycl::ext::intel::experimental::bits_per_symbol<8>,
  sycl::ext::intel::experimental::uses_valid<true>,
  sycl::ext::intel::experimental::ready_latency<0>,
  sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>
));

using SimpleInStream =
    sycl::ext::intel::experimental::pipe<class ID_SimpleInPipe, int, 0,
                                         SimplePipePropertiesT>;

using SimpleOutStream =
    sycl::ext::intel::experimental::pipe<class ID_SimpleOutPipe, std::array<int, 5>, 0,
                                         SimplePipePropertiesT>;

using SimpleInStream_Optimized =
    sycl::ext::intel::experimental::pipe<class ID_SimpleInPipe, int, 0,
                                         SimplePipePropertiesT>;

using SimpleOutStream =
    sycl::ext::intel::experimental::pipe<class ID_SimpleOutPipe, std::array<int, 5>, 0,
                                         SimplePipePropertiesT>;
