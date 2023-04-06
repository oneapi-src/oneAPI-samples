/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XEHE_ROTATE_HPP
#define XEHE_ROTATE_HPP

#include <vector>
#include <ctime>
#include <assert.h>
#include <iostream>
#include <cstdint>
#include <cstdint>
#include <random>
#include <iso646.h>

#ifdef DEBUG_BENCHMARKING
#include <numeric>
#endif

#ifdef __JETBRAINS_IDE__
#define BUILD_WITH_IGPU
#endif

// XeHE
#include "native/xe_uintarith_core.hpp"


namespace xehe {
	namespace eval {


        void kernel_permute_index(uint64_t coeff_id,
            uint32_t coeff_count,
            int coeff_count_power,
            uint32_t galois_elt,
            uint32_t *index
            )
        {
            uint32_t coeff_count_minus_one = coeff_count - 1;
            uint32_t in_index = static_cast<uint32_t>(coeff_id) + coeff_count;
            uint32_t reversed = native::reverse_bits<uint32_t>(static_cast<uint32_t>(in_index), coeff_count_power + 1);
            auto index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
            index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
            index[coeff_id] = native::reverse_bits<uint32_t>(static_cast<uint32_t>(index_raw), coeff_count_power);
        }

        template<typename T>
        void kernel_permute(uint64_t ct_id, uint64_t q_id, uint64_t coeff_id,
            uint32_t encryp_sz, uint32_t q_base_sz, uint32_t coeff_count,
            const uint32_t* permute_table,
            const T* input,
            T* output)
        {
            uint32_t mod_mask = (coeff_count - 1);
            uint32_t prm_idx = (coeff_id & mod_mask);
            auto inp_ptr = input + ct_id * q_base_sz * coeff_count + q_id * coeff_count + permute_table[prm_idx];
            auto out_ptr = output + ct_id * q_base_sz * coeff_count + q_id * coeff_count + coeff_id;
            *out_ptr = *inp_ptr;
        }


/*******************************************************************
    CPU interface

*/
        void permute_index(
            uint32_t coeff_count,
            int coeff_count_power,
            uint32_t galois_elt,
            uint32_t* index
        )
        {
            for (uint64_t coeff_id = 0; coeff_id < coeff_count; coeff_id++)
            {

                kernel_permute_index(coeff_id,
                    coeff_count,
                    coeff_count_power,
                    galois_elt,
                    index
                );
            }
        }

        template<typename T>
        void permute(
            uint32_t encryp_sz, uint32_t q_base_sz, uint32_t coeff_count,
            const uint32_t* permute_table,
            const T* input,
            T* output)
        {
            for (uint64_t ct_id = 0; ct_id < encryp_sz; ++ct_id)
            {
                for (uint64_t q_id = 0; q_id < q_base_sz; ++q_id)
                {
                    for (uint64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
                    {


                        kernel_permute<T>(ct_id, q_id, coeff_id,
                            encryp_sz, q_base_sz, coeff_count,
                            permute_table,
                            input,
                            output);
                    }
                }
            }
        }


#ifdef BUILD_WITH_IGPU
 


        void permute_index(cl::sycl::queue& queue,
            uint32_t coeff_count,
            int coeff_count_power,
            uint32_t galois_elt,
            uint32_t* index,
            bool wait = false
        ) {
            auto grid_range = sycl::range<1>(coeff_count);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class name_permute_index>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {

                    auto coeff_id = it.get_id(0);
                    kernel_permute_index(coeff_id,
                        coeff_count,
                        coeff_count_power,
                        galois_elt,
                        index
                    );

                });
            });//.wait();
            EventCollector::add_event("kernel_permute_index", e);
            if (wait){
                queue.wait();
            }
        }

        template<typename T>
        class name_permute;

        template<typename T>
        void permute(cl::sycl::queue& queue,
            uint32_t encryp_sz, uint32_t q_base_sz, uint32_t coeff_count,
            const uint32_t* permute_table,
            const T* input,
            T* output,
            bool wait = false)
        {
            auto grid_range = sycl::range<3>(encryp_sz, q_base_sz, coeff_count);
            auto e = queue.submit([&](cl::sycl::handler& h) {
                h.parallel_for<class name_permute<T>>({ grid_range }, [=](auto it)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    auto coeff_id = it.get_id(2);
                    auto q_id = it.get_id(1);
                    auto ct_id = it.get_id(0);

                        kernel_permute<T>(ct_id, q_id, coeff_id,
                            encryp_sz, q_base_sz, coeff_count,
                            permute_table,
                            input,
                            output);
                    });
                });//.wait();
            EventCollector::add_event("kernel_permute", e);
            if (wait){
                queue.wait();
            }
        }


#endif // #ifdef BUILD_WITH_IGPU

	} // namespace util

}; // namespace xehe



#endif // XEHE_ROTATE_HPP