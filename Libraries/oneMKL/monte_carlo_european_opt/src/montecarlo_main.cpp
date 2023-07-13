//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <cmath>
#include <limits>

#include <iostream>

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include "montecarlo.hpp"
#include "timer.hpp"

template<typename Type, int>
class k_MonteCarlo; // can be useful for profiling

int main(int argc, char** argv)
{
    try {
        std::cout << "MonteCarlo European Option Pricing in " <<
            (std::is_same_v<DataType, double> ? "Double" : "Single") <<
            " precision using " <<
#if USE_PHILOX
            "PHILOX4x32x10" <<
#elif USE_MRG
            "MRG32k3a" <<
#else
            "MCG59" <<
#endif
            " generator." <<
        std::endl;

        std::cout <<
            "Pricing " << num_options <<
            " Options with Path Length = " << path_length <<
            ", sycl::vec size = " << VEC_SIZE <<
            ", Options Per Work Item = " << ITEMS_PER_WORK_ITEM <<
            " and Iterations = " << num_iterations <<
        std::endl;

        sycl::queue my_queue;
        sycl::usm_allocator<DataType, sycl::usm::alloc::shared> alloc(my_queue);
        std::vector<DataType, decltype(alloc)> h_call_result(num_options, alloc);
        std::vector<DataType, decltype(alloc)> h_call_confidence(num_options, alloc);
        std::vector<DataType, decltype(alloc)> h_stock_price(num_options, alloc);
        std::vector<DataType, decltype(alloc)> h_option_strike(num_options, alloc);
        std::vector<DataType, decltype(alloc)> h_option_years(num_options, alloc);
        DataType* h_call_result_ptr = h_call_result.data();
        DataType* h_call_confidence_ptr = h_call_confidence.data();
        DataType* h_stock_price_ptr = h_stock_price.data();
        DataType* h_option_strike_ptr = h_option_strike.data();
        DataType* h_option_years_ptr = h_option_years.data();

        // calculate the number of blocks
        constexpr DataType fpath_lengthN = static_cast<DataType>(path_length);
        constexpr DataType stddev_denom = 1.0 / (fpath_lengthN * (fpath_lengthN - 1.0));
        DataType confidence_denom = 1.96 / std::sqrt(fpath_lengthN);

        constexpr int rand_seed = 777;
        constexpr std::size_t local_size = 256;
        const std::size_t global_size = (num_options * local_size) / ITEMS_PER_WORK_ITEM; // It requires num_options be divisible by ITEMS_PER_WORK_ITEM
        const int block_n = path_length / (local_size * VEC_SIZE);

        timer tt{};
        double total_time = 0.0;

        namespace mkl_rng = oneapi::mkl::rng;
        mkl_rng::mcg59 engine(
#if !INIT_ON_HOST
            my_queue,
#else
            sycl::queue{sycl::cpu_selector_v},
#endif
            rand_seed); // random number generator object

        auto rng_event_1 = mkl_rng::generate(mkl_rng::uniform<DataType>(5.0, 50.0), engine, num_options, h_stock_price_ptr);
        auto rng_event_2 = mkl_rng::generate(mkl_rng::uniform<DataType>(10.0, 25.0), engine, num_options, h_option_strike_ptr);
        auto rng_event_3 = mkl_rng::generate(mkl_rng::uniform<DataType>(1.0, 5.0), engine, num_options, h_option_years_ptr);

        std::size_t n_states = global_size;
        using EngineType =
#if USE_PHILOX
            mkl_rng::device::philox4x32x10<VEC_SIZE>;
#elif USE_MRG
            mkl_rng::device::mrg32k3a<VEC_SIZE>;
#else
            mkl_rng::device::mcg59<VEC_SIZE>;
#endif

        // initialization needs only on first step
        auto deleter = [my_queue](auto* ptr) {sycl::free(ptr, my_queue);};
        auto rng_states_uptr = std::unique_ptr<EngineType, decltype(deleter)>(sycl::malloc_device<EngineType>(n_states, my_queue), deleter);
        auto* rng_states = rng_states_uptr.get();

        my_queue.parallel_for<class k_initialize_state>(
            sycl::range<1>(n_states),
            std::vector<sycl::event>{rng_event_1, rng_event_2, rng_event_3},
            [=](sycl::item<1> idx) {
                auto id = idx[0];
#if USE_MRG
                constexpr std::uint32_t seed = 12345u;
                rng_states[id] = EngineType({ seed, seed, seed, seed, seed, seed }, { 0, (4096 * id) });
#else
                rng_states[id] = EngineType(rand_seed, id * ITEMS_PER_WORK_ITEM * VEC_SIZE * block_n);
#endif
        })
        .wait_and_throw();

        // main cycle
        for (int i = 0; i < num_iterations; i++)
        {
            tt.start();

            my_queue.parallel_for<k_MonteCarlo<DataType, ITEMS_PER_WORK_ITEM>>(
                sycl::nd_range<1>({global_size}, {local_size}),
                [=](sycl::nd_item<1> item)
                {
                    auto local_state = rng_states[item.get_global_id()];

                    for(std::size_t i = 0; i < ITEMS_PER_WORK_ITEM; ++i)
                    {
                        const std::size_t i_options = item.get_group_linear_id() * ITEMS_PER_WORK_ITEM + i;
                        DataType option_years = h_option_years_ptr[i_options];
                        const DataType VBySqrtT = VLog2E * sycl::sqrt(option_years);
                        const DataType MuByT = MuLog2E * option_years;
                        const DataType Y = h_stock_price_ptr[i_options];
                        const DataType Z = h_option_strike_ptr[i_options];
                        DataType v0 = 0, v1 = 0;

                        mkl_rng::device::gaussian<DataType> distr(MuByT, VBySqrtT);

                        for (int block = 0; block < block_n; ++block)
                        {
                            auto rng_val_vec = mkl_rng::device::generate(distr, local_state);
                            auto rng_val = Y * sycl::exp2(rng_val_vec) - Z;
                            for (int lane = 0; lane < VEC_SIZE; ++lane)
                            {
                                DataType rng_element = sycl::max(rng_val[lane], DataType{});

                                // reduce within the work-item
                                v0 += rng_element;
                                v1 += rng_element * rng_element;
                            }
                        }

                        // reduce within the work-group
                        v0 = sycl::reduce_over_group(item.get_group(), v0, std::plus<>());
                        v1 = sycl::reduce_over_group(item.get_group(), v1, std::plus<>());

                        const DataType exprt = sycl::exp2(RLog2E * option_years);
                        DataType call_result = exprt * v0 * (DataType(1) / fpath_lengthN);

                        const DataType std_dev = sycl::sqrt((fpath_lengthN * v1 - v0 * v0) * stddev_denom);
                        DataType call_confidence = static_cast<DataType>(exprt * std_dev * confidence_denom);

                        if(item.get_local_id() == 0)
                        {
                            h_call_result_ptr[i_options] = call_result;
                            h_call_confidence_ptr[i_options] = call_confidence;
                        }
                    }
            }).wait_and_throw();

            tt.stop();
            if(i != 0)
                total_time += tt.duration();
        }
        std::cout << "Completed in " << total_time << " seconds. Options per second = " << static_cast<double>(num_options * (num_iterations - 1)) / total_time << std::endl;

        // check results
        check(h_call_result, h_call_confidence, h_stock_price, h_option_strike, h_option_years);
    }
    catch (sycl::exception e) {
        std::cout << e.what();
        exit(1);
    }
    return 0;
}
