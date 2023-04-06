//==============================================================
// Copyright � 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <array>
#include <iostream>
#include <iso646.h>
#include <random>

//#define VERBOSE_TEST

#include "../tests/tests_gpu/polyops_ntv.cpp"
#include "../tests/tests_gpu/add_ntv.cpp"
#include "../tests/tests_gpu/mul_ntv.cpp"
#include "../tests/tests_gpu/evaluator_seal.cpp"
#include "../src/include/util/sanity_check_gpu.hpp"
#include "../src/perf_app.hpp"
#include "../src/XeHE.cpp"
//************************************
// Demonstrate summation of arrays both in scalar on CPU and parallel on device
//************************************
int main() {




    xehe::util::XeHE_sanity_check<uint32_t>(64);
    xehe::util::XeHE_sanity_check<uint64_t>(64);

    xehetest::util::XeHETests_Poly_Ops<uint32_t>(14, 6, (size_t(1) << 12));
    xehetest::util::XeHETests_Poly_Ops<uint64_t>(14, 6, (size_t(1) << 12));

    /*
    Basic_static_native_uint_add();
    Basic_static_native_w64_add();
    Basic_static_native_uint_sub();
    Basic_static_native_w64_sub();
    */

    //Basic_static_native_uint_mul();
    Basic_negate_mod_test();
    //Basic_uint64_mod_add_test();
    //Basic_w64_mod_add_test();
    //Basic_uint32_mod_add_test();

    //Basic_uint64_mod_sub_test();
    
    //Basic_uint32_mod_sub_test();

    //Basic_mul_inv_mod_test();

    //Basic_barret_red1_test();

    //Basic_barret_red2_test();
    //Basic_mod_mul2_test();
    


    //Basic_static_bench_native_uint_mul();
    //xehetest::XeTests_Evaluator<uint64_t>();

    // int outer_loop, int inner_loop, int q_base_sz, int log_n
    //DumpUtil<uint32_t>(64, 16, 30);
    //DumpUtil<uint64_t>(64, 16, 60);
    //PerfUtil<uint32_t>(10, 100, 64, 16, 30);
    //
   //PerfUtil<uint64_t>(10, 100, 64, 16, 60);

    return 0;
}
