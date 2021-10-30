#include <CL/sycl.hpp>
SYCL_EXTERNAL extern "C" unsigned rtl_byteswap (unsigned x) {
  return x << 16 | x >> 16;
}
