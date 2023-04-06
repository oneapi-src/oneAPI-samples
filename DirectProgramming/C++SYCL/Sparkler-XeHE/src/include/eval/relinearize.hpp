/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XEHE_RELINEARIZE_HPP
#define XEHE_RELINEARIZE_HPP

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
#include "native/xe_polyops.hpp"


namespace xehe {
	namespace eval {



		// qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
		template<typename T>
		T kernel_ctmodqi_ctmodqk_modqi_internal(T qi, T ntt, const T& inv_q_last_mod_q_op, const T& inv_q_last_mod_q_quo, T input) {
			// Since SEAL uses at most 60bit moduli, 8*qi < 2^63.
			auto qi_lazy = qi << 2;
			input += qi_lazy - ntt;			
			return(native::mul_quotent_mod<T>(input, inv_q_last_mod_q_op, qi, inv_q_last_mod_q_quo));
		}
		// (ct mod qk) mod qi + qi - (qk/2) mod qi 

		template<typename T>
		T kernel_mod4qk_modqi_internal(T qk, T qi, const T* qi_inverse, T input)
		{
			T qk_half = qk >> 1;
			auto temp2 = (qk > qi) ? native::barrett_reduce<T>(input, qi, qi_inverse[1]) : input;
			// Lazy substraction, results in [0, 2*qi), since fix is in [0, qi].
			T fix = qi - native::barrett_reduce<T>(qk_half, qi, qi_inverse[1]);
			T output = temp2 + fix;
			return(output);
		}

		// Add (P-1)/2 to change from flooring to rounding.
		template<typename T>
		T kernel_add_p_1div2_internal(T qk, const T* qk_inverse, T input)
		{
			T qk_half = qk >> 1;
			return(native::barrett_reduce<T>(input + qk_half, qk, qk_inverse[1]));
		}

		/*
		*  Rescaling
		*/

		// ((ct mod qi) - (ct mod qk)) mod qi
		template<typename T>
		void
			kernel_rscl_ctmodqi_ctmodqk_modqi(uint64_t comp_id, uint64_t decomp_id, uint64_t coeff_id,
				int n_components, int decomp_base_sz, int coeff_count,
				const T* modulus,
				const T* modswitch_factors_op,
				const T* modswitch_factors_quo,
				const T* ntt,
				const T* input,
				T* output) {
			T qi = modulus[decomp_id];
			size_t inp_off = (uint64_t(comp_id) * decomp_base_sz* coeff_count + decomp_id * coeff_count + coeff_id);
			uint64_t ntt_off = uint64_t(comp_id) * (decomp_base_sz-1) * coeff_count + decomp_id * coeff_count + coeff_id;
			auto inp = input[inp_off];
			auto ntt_val = ntt[ntt_off];
			// qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
			auto out = kernel_ctmodqi_ctmodqk_modqi_internal(qi, ntt_val, modswitch_factors_op[decomp_id], modswitch_factors_quo[decomp_id], inp);
			// reuse output memory space
			auto out_off = ntt_off;
			output[out_off] = out;

		}


		template<typename T>
		void
			kernel_rscl_mod4qk_modqi(uint64_t ct_id, uint64_t coeff_id,
				int enc_sz, int q_base_sz, int coeff_count,
				const T* q_modulus,
				const T* q_inv2,
				const T* input,
				T* ntt) {

			auto k = q_base_sz - 1;
			T qk = q_modulus[k];
			auto qk_inverse = (q_inv2 + k * 2);
			uint64_t last_off = (ct_id * q_base_sz + k) * coeff_count + coeff_id;
			auto last_inp = input[last_off];
			// Add (P-1)/2 to change from flooring to rounding.
			auto last_plushalf = kernel_add_p_1div2_internal(qk, qk_inverse, last_inp);
			for (uint64_t q_id = 0; q_id < q_base_sz-1; ++q_id)
			{
				T qi = q_modulus[q_id];
				auto qi_inverse = (q_inv2 + q_id * 2);
				uint64_t ntt_off = uint64_t(ct_id) * (q_base_sz-1) * coeff_count + q_id * coeff_count + coeff_id;
				ntt[ntt_off] = kernel_mod4qk_modqi_internal(qk, qi, qi_inverse, last_plushalf);
			}
		}


		/*
		*  Relinearization
		*/


		// ((ct mod qi) - (ct mod qk)) mod qi
		template<typename T>
		void
			kernel_rln_ctmodqi_ctmodqk_modqi(uint64_t comp_id, uint64_t decomp_id, uint64_t coeff_id,
				int rns_base_sz,int n_components, int decomp_base_sz, int coeff_count,
				const T* modulus,
				const T* modswitch_factors_op,
				const T* modswitch_factors_quo,
				const T* ntt,
				T* poly_prod,
				T* encrypt) {
			T qi = modulus[decomp_id];
			size_t poly_prod_off = (uint64_t(comp_id) * rns_base_sz * coeff_count + decomp_id * coeff_count + coeff_id);
			uint64_t ntt_off = uint64_t(comp_id) * decomp_base_sz * coeff_count + decomp_id * coeff_count + coeff_id;
			auto p_prod = poly_prod[poly_prod_off];
			auto ntt_val = ntt[ntt_off];
			// qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
			auto p_prod_mul_fctr = kernel_ctmodqi_ctmodqk_modqi_internal(qi, ntt_val, modswitch_factors_op[decomp_id], modswitch_factors_quo[decomp_id], p_prod);
			// add to ct0, ct1
			uint64_t encrypt_off = ntt_off;
			auto encr_in = encrypt[encrypt_off];
			auto encr_ot = native::add_mod(p_prod_mul_fctr, encr_in, qi);
			encrypt[encrypt_off] = encr_ot;

		}



		template<typename T>
		void
			kernel_rln_mod4qk_modqi(uint64_t comp_id, uint64_t coeff_id,
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* key_modulus,
				const T* key_inv2,
				const T* temp,
				T* ntt) {

			auto k = key_mod_base - 1;
			T qk = key_modulus[k];
			auto qk_inverse = (key_inv2 + k * 2);
			uint64_t temp_off = (comp_id * rns_base_sz + q_base_sz) * coeff_count + coeff_id;
			auto inp = temp[temp_off];
			// Add (P-1)/2 to change from flooring to rounding.
			auto tmp = kernel_add_p_1div2_internal(qk, qk_inverse, inp);
			for (int decomp_id = 0; decomp_id < q_base_sz; ++decomp_id)
			{
				T qi = key_modulus[decomp_id];

				auto qi_inverse = (key_inv2 + decomp_id * 2);

				uint64_t ntt_off = uint64_t(comp_id) * q_base_sz * coeff_count + decomp_id * coeff_count + coeff_id;
				ntt[ntt_off] = kernel_mod4qk_modqi_internal(qk, qi, qi_inverse, tmp);

			}

		}


		// Add (P-1)/2 to change from flooring to rounding.
		template<typename T>
		void
			kernel_add_p_1div2(uint64_t comp_id, uint64_t coeff_id,
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* key_modulus,
				const T* key_inv2,
				T* temp) {

			auto k = key_mod_base - 1;
			T qk = key_modulus[k];
			auto qk_inverse = (key_inv2 + k * 2);
			uint64_t temp_off = (comp_id * rns_base_sz + q_base_sz) * coeff_count + coeff_id;
			auto inp = temp[temp_off];
			temp[temp_off] = kernel_add_p_1div2_internal(qk, qk_inverse, inp);
		}


		template<typename T>
		void
			kernel_rln_keys_dotprod_mod_lazy(int rns_id, int comp_id,int coeff_id,
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* keys,
				const T* key_modulus,
				const T* key_inv2,
				const T* operand,
				T* poly_prod,
				int count_bound) {

			int lazy_reduce_count = count_bound;
			T accum2[2] = { 0,0 };
			// switch key id
			int key_id = (rns_id == q_base_sz) ? key_mod_base - 1 : rns_id;
			auto qi = key_modulus[key_id];
			auto qi_key_ratio = (key_inv2 + key_id * 2);
			// operand
			uint64_t oprnd_off = uint64_t(rns_id) * q_base_sz * coeff_count  + coeff_id;
			uint64_t key_off = uint64_t(comp_id) * key_mod_base * coeff_count + key_id * coeff_count + coeff_id;

			// loop over RNS decomp 
			// accumulate
			// reduce by qi
			for (int j = 0; j < q_base_sz; ++j)
			{
				// j is decomp mod index
				int q_id = j;

				T key = keys[key_off + uint64_t(q_id) * n_components * key_mod_base * coeff_count];

				T coeff = operand[oprnd_off + q_id * coeff_count];
				T word2[2];
                native::mul_uint2<T>(coeff, key, word2);

				// Accumulate product and reduce
				native::add_uint<T>(word2, accum2, word2);
				--lazy_reduce_count;
				accum2[0] = (!lazy_reduce_count) ? native::barrett_reduce2<T>(word2, qi, qi_key_ratio) : word2[0];
				accum2[1] = (!lazy_reduce_count) ? 0 : word2[1];
				lazy_reduce_count = (lazy_reduce_count == 0) ? count_bound : lazy_reduce_count;
			}
			// final reduce
			accum2[0] = (lazy_reduce_count) ? native::barrett_reduce2<T>(accum2, qi, qi_key_ratio) : accum2[0];
			size_t poly_prod_off = (uint64_t(comp_id) * rns_base_sz * coeff_count + rns_id * coeff_count + coeff_id);
			poly_prod[poly_prod_off] = accum2[0];

		}

		template<typename T>
		void
			kernel_rln_keys_mod(int64_t rns_id, int64_t coeff_id,
				int rns_base_sz, int q_base_sz, int key_mod_base, int coeff_count,
				const T* keys,
				const T* key_modulus,
				const T* key_inv2,
				const T* input,
				T* operand) {

			// switch key id
			int key_id = (rns_id == q_base_sz) ? key_mod_base - 1 : rns_id;
			auto qk = key_modulus[key_id];
			auto qk_ratio = (key_inv2 + key_id * 2);
			// operand
			uint64_t oprnd_off = uint64_t(rns_id) * q_base_sz * coeff_count + coeff_id;

			// loop over RNS decomp 
			for (int j = 0; j < q_base_sz; ++j)
			{
				// j is decomp mod index
				int q_id = j;
				uint64_t input_off = uint64_t(q_id) * coeff_count + coeff_id;

				T qi = key_modulus[q_id];

				T inp = input[input_off];
				T oprnd;
				if (qi <= qk)
				{
					oprnd = inp;
				}
				else
				{
					oprnd = native::barrett_reduce<T>(inp, qk, qk_ratio[1]);
				}

				operand[oprnd_off + q_id * coeff_count] = oprnd;

			}


		}


/*******************************************************************
    CPU interface

*/

/*
*  Rescaling
*/

		template<typename T>
		void
			rscl_ctmodqi_ctmodqk_modqi(
				int encrypted_size, int q_base_sz, int coeff_count,
				const T* xe_modulus,
                const T* xe_inv_q_last_mod_q_op,
                const T* xe_inv_q_last_mod_q_quo,
				const T* ntt,
				const T* input,
				T* output)
		{
			auto k = (q_base_sz - 1);
			for (uint64_t ct_id = 0; ct_id < encrypted_size; ++ct_id)
			{
				for (uint64_t q_id = 0; q_id < k; ++q_id)
				{
					for (uint64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
					{
						kernel_rscl_ctmodqi_ctmodqk_modqi(ct_id, q_id, coeff_id,
							encrypted_size, q_base_sz, coeff_count,
							xe_modulus,
							xe_inv_q_last_mod_q_op,
							xe_inv_q_last_mod_q_quo,
							ntt,
							input,
							output);
					}
				}
			}
		}
		



// (ct mod 4qk) mod qi
// Lazy substraction, results in [0, 2*qi), since fix is in [0, qi].
		template<typename T>
		void
			rscl_mod4qk_modqi(
				int encrypted_size, int q_base_sz, int coeff_count,
				const T* xe_modulus,
                const T* xe_inv2,
				const T* input,
				T* output)
		{
			for (uint64_t ct_id = 0; ct_id < encrypted_size; ++ct_id)
			{
				for (uint64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
				{
					kernel_rscl_mod4qk_modqi<T>(ct_id, coeff_id,
						encrypted_size, q_base_sz, coeff_count,
						xe_modulus,
						xe_inv2,
						input, //last_input
						output); // reuse output
				}
			}

		}


/*
*  Relinearization
*/
		template<typename T>
		void
			rln_ctmodqi_ctmodqk_modqi(
				int rns_base_sz,  int n_components, int decomp_base_sz, int coeff_count,
				const T* key_modulus,
                const T* modswitch_factors_op,
                const T* modswitch_factors_quo,
				const T* ntt,
				T* poly_prod,
				T* encrypt) {

			for (uint64_t comp_id = 0; comp_id < n_components; ++comp_id)
			{
				for (uint64_t decomp_id = 0; decomp_id < decomp_base_sz; ++decomp_id)
				{
					for (uint64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
					{
						kernel_rln_ctmodqi_ctmodqk_modqi(comp_id, decomp_id, coeff_id,
							rns_base_sz, n_components, decomp_base_sz, coeff_count,
							key_modulus,
							modswitch_factors_op,
                            modswitch_factors_quo,
							ntt,
							poly_prod,
							encrypt);
					}
				}
			}

		}

		// (ct mod 4qk) mod qi
		// Lazy substraction, results in [0, 2*qi), since fix is in [0, qi].
		template<typename T>
		void
			rln_mod4qk_modqi(
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* key_modulus,
                const T* key_inv2,
				const T* temp,
				T* ntt)
		{
			for (uint64_t comp_id = 0; comp_id < n_components; ++comp_id)
			{
				for (uint64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
				{
					kernel_rln_mod4qk_modqi(comp_id, coeff_id,
						rns_base_sz, q_base_sz, n_components, key_mod_base, coeff_count,
						key_modulus,
						key_inv2,
						temp,
						ntt
					);
				}
			}
			
		}

		// Add (p-1)/2 to change from flooring to rounding.
		template<typename T>
		void
			add_p_1div2(
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* key_modulus,
                const T* key_inv2,
				T* temp)
		{
			for (uint64_t comp_id = 0; comp_id < n_components; ++comp_id)
			{
				for (uint64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
				{
					kernel_add_p_1div2(comp_id, coeff_id,
						rns_base_sz, q_base_sz, n_components, key_mod_base, coeff_count,
						key_modulus,
						key_inv2,
						temp);
				}
			}
		}

		template<typename T>
		void
			rln_keys_dotprod_mod_lazy(int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* keys,
				const T* key_modulus,
				const T* key_inv2,
				const T* operand,
				T* poly_prod,
				int count_bound) {

			for (uint64_t rns_id = 0; rns_id < rns_base_sz; ++rns_id)
			{
				for (uint64_t comp_id = 0; comp_id < n_components; ++comp_id)
				{
					for (uint64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
					{
						kernel_rln_keys_dotprod_mod_lazy(rns_id, comp_id, coeff_id,
							rns_base_sz, q_base_sz, n_components, key_mod_base, coeff_count,
							keys,
							key_modulus,
							key_inv2,
							operand,
							poly_prod,
							count_bound);
					}
				}
			}
		}


		template<typename T>
		void
			rln_keys_mod(
				int rns_base_sz, int q_base_sz, int key_mod_base, int coeff_count,
				const T* keys,
				const T* key_modulus,
                const T* key_inv2,
				const T* input,
				T* operand) {

			for (int64_t rns_id = 0; rns_id < rns_base_sz; ++rns_id)
			{
				for (int64_t coeff_id = 0; coeff_id < coeff_count; ++coeff_id)
				{
					kernel_rln_keys_mod(rns_id, coeff_id,
						rns_base_sz, q_base_sz, key_mod_base, coeff_count,
						keys,
						key_modulus,
						key_inv2,
						input,
						operand);
				}
			}
		}

#ifdef BUILD_WITH_IGPU

		/*
		*  Rescaling
		*/

		template<typename T>
		class name_rscl_ctmodqi_ctmodqk_modqi;

		template<typename T>
		void
			rscl_ctmodqi_ctmodqk_modqi(cl::sycl::queue& queue,
				int encrypted_size, int q_base_sz, int coeff_count,
				const T* xe_modulus,
				const T* xe_inv_q_last_mod_q_op,
				const T* xe_inv_q_last_mod_q_quo,
				const T* ntt,
				const T* input,
				T* output,
				bool wait = false)
		{
			auto k = (q_base_sz - 1);
			auto grid_range = sycl::range<3>(encrypted_size, k, coeff_count);
			auto e = queue.submit([&](cl::sycl::handler& h) {
				h.parallel_for<class name_rscl_ctmodqi_ctmodqk_modqi<T>>({ grid_range }, [=](auto it)
				[[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
				{
					auto coeff_id = it.get_id(2);
					auto q_id = it.get_id(1);
					auto ct_id = it.get_id(0);
					kernel_rscl_ctmodqi_ctmodqk_modqi(ct_id, q_id, coeff_id,
						encrypted_size, q_base_sz, coeff_count,
						xe_modulus,
						xe_inv_q_last_mod_q_op,
						xe_inv_q_last_mod_q_quo,
						ntt,
						input,
						output);
					});
				});//.wait();
            EventCollector::add_event("kernel_rscl_ctmodqi_ctmodqk_modqi", e);
            if (wait){
                queue.wait();
            }
		}

		template<typename T>
		class name_rscl_mod4qk_modqi;

		// (ct mod 4qk) mod qi
		// Lazy substraction, results in [0, 2*qi), since fix is in [0, qi].
		template<typename T>
		void
			rscl_mod4qk_modqi(cl::sycl::queue& queue,
				int encrypted_size, int q_base_sz, int coeff_count,
				const T* xe_modulus,
				const T* xe_inv2,
				const T* input,
				T* output,
				bool wait = false)
		{
			auto grid_range = sycl::range<2>(encrypted_size, coeff_count);
			auto e = queue.submit([&](cl::sycl::handler& h) {
				h.parallel_for<class name_rscl_mod4qk_modqi<T>>({ grid_range }, [=](auto it)
				[[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
				{
					auto coeff_id = it.get_id(1);
					auto ct_id = it.get_id(0);
					kernel_rscl_mod4qk_modqi<T>(ct_id, coeff_id,
						encrypted_size, q_base_sz, coeff_count,
						xe_modulus,
						xe_inv2,
						input, //last_input
						output); // reuse output
					});
				});//.wait();
            EventCollector::add_event("kernel_rscl_mod4qk_modqi", e);
            if (wait){
                queue.wait();
            }
		}


		/*
		*  Relinearization
		*/


		template<typename T>
		class name_rln_ctmodqi_ctmodqk_modqi;

		template<typename T>
		void
			rln_ctmodqi_ctmodqk_modqi(cl::sycl::queue& queue,
				int rns_base_sz, int n_components, int decomp_base_sz, int coeff_count,
				const T* key_modulus,
				const T* modswitch_factors_op,
				const T* modswitch_factors_quo,
				const T* ntt,
				T* poly_prod,
				T* encrypt,
				bool wait = false) {

			auto grid_range = sycl::range<3>(n_components, decomp_base_sz, coeff_count);
			auto e = queue.submit([&](cl::sycl::handler& h) {
				h.parallel_for<class name_rln_ctmodqi_ctmodqk_modqi<T>>({ grid_range }, [=](auto it)
				[[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
				{
					auto coeff_id = it.get_id(2);
					auto decomp_id = it.get_id(1);
					auto comp_id = it.get_id(0);
					kernel_rln_ctmodqi_ctmodqk_modqi(comp_id, decomp_id, coeff_id,
						rns_base_sz, n_components, decomp_base_sz, coeff_count,
						key_modulus,
						modswitch_factors_op,
						modswitch_factors_quo,
						ntt,
						poly_prod,
						encrypt);
					});
				});//.wait();
            EventCollector::add_event("kernel_rln_ctmodqi_ctmodqk_modqi", e);
            if (wait){
                queue.wait();
            }
		}


		// (ct mod 4qk) mod qi
		// Lazy substraction, results in [0, 2*qi), since fix is in [0, qi].
		template<typename T>
		class name_rln_mod4qk_modqi;


		template<typename T>
		void
			rln_mod4qk_modqi(cl::sycl::queue& queue,
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* key_modulus,
				const T* key_inv2,
				const T* temp,
				T* ntt,
				bool wait = false)
		{
			auto grid_range = sycl::range<2>(n_components, coeff_count );
			auto e = queue.submit([&](cl::sycl::handler& h) {
				h.parallel_for<class name_rln_mod4qk_modqi<T>>({ grid_range }, [=](auto it)
				[[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
				{
					auto coeff_id = it.get_id(1);
					auto comp_id = it.get_id(0);
					kernel_rln_mod4qk_modqi(comp_id, coeff_id,
						rns_base_sz, q_base_sz, n_components, key_mod_base, coeff_count,
						key_modulus,
						key_inv2,
						temp,
						ntt
					);
					});
				});//.wait();
            EventCollector::add_event("kernel_rln_mod4qk_modqi", e);
            if (wait){
                queue.wait();
            }
		}
		// Add (p-1)/2 to change from flooring to rounding.
		template<typename T>
		class name_add_p_1div2;

		template<typename T>
		void
			add_p_1div2(cl::sycl::queue& queue,
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* key_modulus,
				const T* key_inv2,
				T* temp)
		{
			auto grid_range = sycl::range<2>(n_components, coeff_count);
			auto e = queue.submit([&](cl::sycl::handler& h) {
				h.parallel_for<class name_add_p_1div2<T>>({ grid_range }, [=](auto it)
				[[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
				{
					auto coeff_id = it.get_id(1);
					auto comp_id = it.get_id(0);
					kernel_add_p_1div2(comp_id, coeff_id,
						rns_base_sz, q_base_sz, n_components, key_mod_base, coeff_count,
						key_modulus,
						key_inv2,
						temp);
					});
				}).wait();
            EventCollector::add_event("kernel_add_p_1div2", e);

		}

		template<typename T>
		class name_rln_keys_dotprod_mod;

		template<typename T>
		void
			rln_keys_dotprod_mod_lazy(cl::sycl::queue& queue,
				int rns_base_sz, int q_base_sz, int n_components, int key_mod_base, int coeff_count,
				const T* keys,
				const T* key_modulus,
				const T* key_inv2,
				const T* operand,
				T* poly_prod,
				int count_bound,
				bool wait = false) {

			auto grid_range = sycl::range<3>(rns_base_sz, n_components, coeff_count);
			auto e = queue.submit([&](cl::sycl::handler& h) {
				h.parallel_for<class name_rln_keys_dotprod_mod<T>>({ grid_range }, [=](auto it)
				[[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
				{
					auto coeff_id = it.get_id(2);
					auto comp_id  = it.get_id(1);
					auto rns_id   = it.get_id(0);
					kernel_rln_keys_dotprod_mod_lazy(rns_id, comp_id, coeff_id,
						rns_base_sz, q_base_sz, n_components, key_mod_base, coeff_count,
						keys,
						key_modulus,
						key_inv2,
						operand,
						poly_prod,
						count_bound);
				});
			});//.wait();
            EventCollector::add_event("kernel_rln_keys_dotprod_mod_lazy", e);
			if (wait){
			    queue.wait();
			}
		}

		template<typename T>
		class name_rln_keys_mod;

		template<typename T>
		void
			rln_keys_mod(cl::sycl::queue& queue,
				int rns_base_sz, int q_base_sz, int key_mod_base, int coeff_count,
				const T* keys,
				const T* key_modulus,
				const T* key_inv2,
				const T* input,
				T* operand,
				bool wait = false) {
			auto grid_range = sycl::range<2>(rns_base_sz, coeff_count);
			auto e = queue.submit([&](cl::sycl::handler& h) {
				h.parallel_for<class name_rln_keys_mod<T>>({ grid_range }, [=](auto it)
				[[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
				{
					auto coeff_id = it.get_id(1);
					auto rns_id = it.get_id(0);
					kernel_rln_keys_mod(rns_id, coeff_id,
						rns_base_sz, q_base_sz, key_mod_base, coeff_count,
						keys,
						key_modulus,
						key_inv2,
						input,
						operand);
				});
			});//.wait();
            EventCollector::add_event("kernel_rln_keys_mod", e);
			if (wait) {
			    queue.wait();
			}
		}

#endif // #ifdef BUILD_WITH_IGPU

	} // namespace util

}; // namespace xehe



#endif // XEHE_RELINEARIZE_HPP
