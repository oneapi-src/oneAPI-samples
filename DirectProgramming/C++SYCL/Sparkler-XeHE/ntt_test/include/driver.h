/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef _DRIVER_H_
#define _DRIVER_H_

#include <vector>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <cstdint>
#include <random>
#include <iso646.h>
#include <cstddef>
#include <string>

#ifdef DEBUG_BENCHMARKING
#include <numeric>
#endif // DEBUG_BENCHMARKING

#ifdef BUILD_WITH_IGPU
#include <XeHE.hpp>


#ifdef BUILD_WITH_SEAL
#include "seal/batchencoder.h"
#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/keygenerator.h"
#include "seal/modulus.h"
#endif // BUILD_WITH_SEAL

#endif // BUILD_WITH_IGPU

#define _NTT_MT_ false

void run_driver(int n = 1024 * 8, int time_loop = 10, int rns_base_sz = 6, size_t queue_num = 1);
template<typename T>
T calc_quotient(const seal::Modulus& modulus, T operand)
{
#ifdef SEAL_DEBUG
    if (operand >= T(modulus.value()))
    {
        throw std::invalid_argument("input must be less than modulus");
    }
#endif
    T quotient = 0;
    if constexpr (sizeof(T) == 8)
    {
        T num2[2]{ 0, operand };
        T quo2[2];
        seal::util::divide_uint128_inplace(num2, modulus.value(), quo2);
        quotient = quo2[0];
    }
    else  if constexpr (sizeof(T) == 4)
    {
        uint64_t wide_op = (uint64_t(operand) << 32);
        quotient = T(wide_op / modulus.value());            
    }   

    return (quotient);
}

template<typename T>
void calc_quotient2(T * quo2, const seal::Modulus& modulus, T operand)
{
    auto mod = modulus.value();
#ifdef SEAL_DEBUG
    if (operand >= mod)
    {
        throw std::invalid_argument("input must be less than modulus");
    }
#endif
    if constexpr (sizeof(T) == 8)
    {
        T num3[3]{ 0, 0, operand };
        T quo3[3];
        seal::util::divide_uint192_inplace(num3, mod, quo3);
        quo2[0] = quo3[0];
        quo2[1] = quo3[1];
    }
    else  if constexpr (sizeof(T) == 4)
    {
        *((uint64_t*)quo2) = calc_quotient<uint64_t>(modulus, uint64_t(operand));            
    }        
}


template <typename T>
class Driver{
public:
    ~Driver(void);
    // calls to benchmark interfaces
    void benchmark_interface(int n = 1024*8, int time_loop=10, int prime_length=sizeof(T)*8-4, std::vector<int>* p_mods=nullptr, size_t queue_num = 1);

    /***
    benchmarking functions
    ***/
    void NTT_benchmark(std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, int time_loop, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
        const std::vector<T>& modulus);
    void invNTT_benchmark(std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, int time_loop, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
        const std::vector<T>& modulus);
    void NTT_correctness(std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, std::vector<std::vector<T>> input, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
        const std::vector<T>& modulus);

private:
    xehe::ext::XeHE_mem_context<T> gpu_context_entry_;
    std::vector<T> modulus_;
    // number of physucal queues
    size_t ph_q_num_ = 2;
};

#define MS_in_1SEC 1000.0
class Timer
{
public:
    void start()
    {
        StartTime = std::chrono::steady_clock::now();
        Running = true;
    }

    void stop()
    {
        EndTime = std::chrono::steady_clock::now();
        Running = false;
    }

    double elapsedMicroseconds()
    {
        std::chrono::time_point<std::chrono::steady_clock> endTime;

        if (Running)
        {
            endTime = std::chrono::steady_clock::now();
        }
        else
        {
            endTime = EndTime;
        }

        return std::chrono::duration_cast<std::chrono::microseconds>(endTime - StartTime).count();
    }

    double elapsedMilliseconds()
    {
        std::chrono::time_point<std::chrono::steady_clock> endTime;

        if (Running)
        {
            endTime = std::chrono::steady_clock::now();
        }
        else
        {
            endTime = EndTime;
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - StartTime).count();
    }

    double elapsedSeconds()
    {
        return elapsedMilliseconds() / MS_in_1SEC;
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> StartTime;
    std::chrono::time_point<std::chrono::steady_clock> EndTime;
    bool Running = false;
};

#endif //#ifndef _DRIVER_H_