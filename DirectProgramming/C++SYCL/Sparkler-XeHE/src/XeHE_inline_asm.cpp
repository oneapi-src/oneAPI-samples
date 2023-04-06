/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

/*
Utility to verify the correctness (+performance) of
inlined assembly implementations against xehe::native operations
*/

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>

#include "util/perf_gpu.hpp"

int main(int argc, char* argv[]) 
{  

  std::cout << "Beginning Inline Assembly Verification utility\n" 
            << "...................................................................\n\n" ;

  xehe::util::BaseInlineAsm<uint32_t> inline_util_32(64, 16);

  inline_util_32.verify_inline_add_uint();
  inline_util_32.verify_inline_addmod();
  inline_util_32.verify_inline_addmod_opt();

  inline_util_32.verify_inline_mul_uint_low();
  inline_util_32.verify_inline_mul_uint();


  xehe::util::BaseInlineAsm<uint64_t> inline_util_64(64, 16);

  inline_util_64.verify_inline_add_uint();
  inline_util_64.verify_inline_addmod();
  inline_util_64.verify_inline_addmod_opt();

  inline_util_64.verify_inline_mul_uint_low();
  inline_util_64.verify_inline_mul_uint();


      
  return 1;
}
