/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef _PERF_APP_HPP_
#define _PERF_APP_HPP_

#include <array>
#include <vector>
#include <iostream>
#include <iso646.h>
#include <random>
#include <iomanip>

#define PERF_INSTS  7 // number of instructions performance is reported for
#ifdef BUILD_WITH_IGPU

// XeHE
#include "include/util/perf_gpu.hpp"


/*
* TODO:
* real HW parameters
* correct calibrator
* SPIR-V/asm dump
* other ops: double mulmod, NTT butterfly, other ?
*
* Parmeters:
*
* number of kernel launchs
* internal computational loop
* RNS base size
* lon N
* prime bitwidth
*/

enum instructions { AddInstr, AddModInstr, MulInstr, MulModInstr, Mul2Instr, Mul2ModInstr, MulNativeInstr};
std::vector<std::string> inst_names {"Add", "AddMod", "Mul", "MulMod", "Mul2", "Mul2Mod", "MulNative"};
std::vector<std::string> inst_names_enh {"Add", "Add_Mod", "Mul_Low", "Mul_Mod", "Mul_High_Low", "Mul_High_Low_Mod", "Mul_High_Low_Native"};


struct inst_data {
    std::string name;
    double duration;
    uint32_t clc;
};

struct perf_util_result{
    int n_EUs;
    double spec_max_eng_freq;
    uint64_t actual_eng_freq;
    double duration;
    double n_calib_instruction;
    uint32_t total_clc;
    std::vector<inst_data> insts;

    perf_util_result(){ insts.resize(PERF_INSTS);}
};

template <typename T>
perf_util_result PerfUtil(int outer_loop, int inner_loop, int q_base_sz, int log_n, int p_size)
{
    perf_util_result result;
    std::string bits;

    
    if (sizeof(T) == 4)
    {
        bits = "32";
    }
    else 
    {
       bits = "64";
    }
    

    //std::cout << "||=================================== Perf Utility for " << bits << "bits ===================================||" << std::endl;

    double n_EUs = 24;
    double max_eng_freq = 1000000000;
    double calibrated_clock = 1;
    uint32_t simd_width = 8;
    uint32_t unrolled_inner_loop = 1024;
    uint64_t time_scale = 1000000;


    // instantiate per class
    xehe::util::BasePerf<T> perf_util(outer_loop, inner_loop, q_base_sz, log_n);

    auto n = (size_t(1) << log_n);
    sycl::device dev = perf_util.get_queue().get_device();
    n_EUs = dev.get_info<cl::sycl::info::device::max_compute_units>();
    double spec_max_eng_freq = dev.get_info<cl::sycl::info::device::max_clock_frequency>();

    double duration;
    perf_util.calibrate_float_perf(duration); 
    //perf_util.calibrate_float_perf(duration);

    auto range = n * q_base_sz;
    //calibrated number
    auto n_calib_instruction256 = double(outer_loop) * 256 * range;
    auto n_calib_instruction128 = double(outer_loop) * 128 * range;
    //auto n_calib_instruction64 = double(outer_loop) * 64 * range;
    //auto n_calib_instruction16 = double(outer_loop) * 16 * range;
    auto n_calib_instruction =  double(outer_loop) * unrolled_inner_loop * range;
    //    double comp_throuput = (n_EUs * 8 * max_eng_freq);

    double throuput = (n_calib_instruction * time_scale) / (duration);

    double clocks = calibrated_clock; // comp_throuput / throuput;

    max_eng_freq = clocks * throuput / (n_EUs * simd_width);
    double comp_throuput = (n_EUs * simd_width * max_eng_freq);

    
    result.n_EUs = n_EUs;
    result.spec_max_eng_freq = spec_max_eng_freq;
    result.actual_eng_freq =  uint64_t(max_eng_freq/1000000);
    result.duration = duration;
    result.n_calib_instruction = n_calib_instruction;
    result.total_clc =  uint32_t(clocks + 0.5);

    // real number
    //auto n_instruction = double(outer_loop) * inner_loop * range;
    {
        double duration;
        perf_util.add_perf(duration);
        double throuput = (n_calib_instruction256 * time_scale) / (duration);
        double clocks = comp_throuput / throuput;
        auto int_clocks = std::max(uint32_t(1), uint32_t(clocks + 0.5));
        // std::cout << "Add:"
        //     << " duration " << duration << " "
        //     << int_clocks << " clc"
        //     << std::endl;
        result.insts[AddInstr].name = inst_names[AddInstr];
        result.insts[AddInstr].duration = duration;
        result.insts[AddInstr].clc = int_clocks;
    }

    {
        double duration;
        perf_util.add_mod_perf(duration);
        double throuput = (n_calib_instruction256 * time_scale) / (duration);
        double clocks = comp_throuput / throuput;
        // std::cout << "AddMod:"
        //     << " duration " << duration << " "
        //     << uint32_t(clocks + 0.5) << " clc"
        //     << std::endl;
        result.insts[AddModInstr].name = inst_names[AddModInstr];
        result.insts[AddModInstr].duration = duration;
        result.insts[AddModInstr].clc =  uint32_t(clocks + 0.5);
    }

    {
        double duration;
        perf_util.mul_perf(duration);
        double throuput = (n_calib_instruction256 * time_scale) / (duration);
        double clocks = comp_throuput / throuput;
        auto int_clocks = std::max(uint32_t(1), uint32_t(clocks + 0.5));
        // std::cout << "Mul:"
        //     << " duration " << duration << " "
        //     << int_clocks << " clc"
        //     << std::endl;

        result.insts[MulInstr].name = inst_names[MulInstr];
        result.insts[MulInstr].duration = duration;
        result.insts[MulInstr].clc =  int_clocks;
    }

    {
        double duration;
        perf_util.mul_mod_perf(duration);
        double throuput = (n_calib_instruction256 * time_scale) / (duration);
        double clocks = comp_throuput / throuput;
        // std::cout << "MulMod:"
        //     << " duration " << duration << " "
        //     << uint32_t(clocks + 0.5) << " clc"
        //     << std::endl;

        result.insts[MulModInstr].name = inst_names[MulModInstr];
        result.insts[MulModInstr].duration = duration;
        result.insts[MulModInstr].clc = uint32_t(clocks + 0.5);
    }

    {
        double duration;
        perf_util.mul2_perf(duration);
        double throuput = (n_calib_instruction256 * time_scale) / (duration);
        double clocks = comp_throuput / throuput;

        // std::cout << "Mul2:"
        //     << " duration " << duration << " "
        //     << uint32_t(clocks + 0.5) << " clc"
        //     << std::endl;

        result.insts[Mul2Instr].name = inst_names[Mul2Instr];
        result.insts[Mul2Instr].duration = duration;
        result.insts[Mul2Instr].clc = uint32_t(clocks + 0.5);
    }

    {
        // durection is ciulated to give a consistent view: more clocks -> more time
        double duration;
        perf_util.mul2_mod_perf(duration);
        double throuput = (n_calib_instruction128 * time_scale) / (duration);
        double clocks = comp_throuput / throuput;
        // std::cout << "Mul2Mod:"
        //     << " duration " << duration * (n_calib_instruction256/ n_calib_instruction128) << " "
        //     << uint32_t(clocks + 0.5) << " clc"
        //     << std::endl;    

        result.insts[Mul2ModInstr].name = inst_names[Mul2ModInstr];
        result.insts[Mul2ModInstr].duration = duration;
        result.insts[Mul2ModInstr].clc = uint32_t(clocks + 0.5);
    }

    {
        // durection is ciulated to give a consistent view: more clocks -> more time
        double duration;
        perf_util.mul_native_perf(duration);
        double throuput = (n_calib_instruction128 * time_scale) / (duration);
        double clocks = comp_throuput / throuput;

        result.insts[MulNativeInstr].name = inst_names[MulNativeInstr];
        result.insts[MulNativeInstr].duration = duration;
        result.insts[MulNativeInstr].clc = uint32_t(clocks + 0.5);
    }

#if 0
    XeHE_IF_CONSTEXPR(is_uint64_v<T>)
    {
        double duration;
        perf_util.ntt_negacyclic_forward_perf(duration);
        // lon_N rounds
        double n_ntt_instructions = outer_loop* range * log_n;
        double throuput = (n_ntt_instructions * time_scale) / (duration);
        double clocks = comp_throuput / throuput;
        std::cout << "NTT forward:"
            << " duration " << duration << " "
            << uint32_t(clocks + 0.5) << " clc"
            << std::endl;
    }
#endif

    return result;
}

template <typename T>
void DumpUtil(int q_base_sz, int log_n, int p_size)
{
    std::string bits;

    if (sizeof(T) == 4)
    {
        bits = "32";
    }
    else
    {
        bits = "64";
    }

    std::cout << "Dump asm utility " << bits << "bits ==============================================================" << std::endl;
   
    std::vector<T> xe_modulus;

    std::vector<T> xe_const_ratio;
    // size_t const_ratio_sz = 3;
    auto n = (size_t(1) << log_n);
    std::vector<xehe::native::Modulus<T>> gen_modulus = xehe::native::get_primes<T>(n, p_size, q_base_sz);

    // const_ratio_sz = gen_modulus[0].const_ratio().size();

    // for (int i = 0; i < q_base_sz; ++i)
    // {
    //     xe_modulus.push_back(gen_modulus[i].value());
    //     auto temp = gen_modulus[i].const_ratio().data();
    //     for (int j = 0; j < const_ratio_sz; ++j)
    //     {
    //         xe_const_ratio.push_back(temp[i]);
    //     }
    // }

    // xehe::util::BaseAsmDump<T> dump_util(q_base_sz, log_n);
    xehe::util::BaseAsmDump<T> dump_util(q_base_sz, log_n, gen_modulus);

    dump_util.add_code();
    dump_util.add_mod_code();
    dump_util.mul_code();
    dump_util.mul_mod_code();
    dump_util.mul2_code();
    dump_util.mul2_mod_code();

    //inline assembly dumps
    dump_util.inline_add_code();
}
#endif //#ifdef BUILD_WITH_IGPU

#endif // #ifndef _PERF_APP_HPP_