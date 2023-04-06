
#ifndef XEHE_EVALUATOR_HPP
#define XEHE_EVALUATOR_HPP

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

#ifdef BUILD_WITH_IGPU

// XeHE
#include "eval/relinearize.hpp"
#include "util/xe_ntt.h"
#include "util/xe_ntt_relin.h"
#include "util/xe_intt_rescale.h"

/**********************************
 * GPU kernels for XeHE Evaluator
 *
 */
namespace xehe {
    namespace eval {

        /**********************************************************
        *   CKKS mod switch
        *
        */
        template<typename T>
        void ckks_mod_switch_drop_to_next(cl::sycl::queue &queue,
            int dest_size,        // number of output compunents
            int dest_q_base_sz,   // destination rns base size
            int q_base_sz,        // input rns based size
            int coeff_count,                // order of poly
            const T* input,
            T* output,
            bool wait = true)
        {
            // TODO: WRAP INTO GPU KERNEL
            for (int d = 0; d < dest_size; ++d)
            {
                auto out_ptr = output + d * dest_q_base_sz * coeff_count;
                auto inp_ptr = input + d * q_base_sz * coeff_count;
                xehe::gpu_copy(queue, out_ptr, inp_ptr, (dest_q_base_sz * coeff_count), false);
            }
            if (wait){
                queue.wait();
            }
        }


        /**********************************************************
         *   CKKS square
         *
         */
        template<typename T>
        void ckks_square(
                cl::sycl::queue &queue,
                int dest_size,        // number of output components
                int q_base_sz,        // rns based size
                int coeff_count,                // order of poly
                const T* xe_modulus,            // RNS modules
                const T* inv_mod2,                  // inverse length
                T * inp_output,                // input/output inplace
                bool wait = true
                //T * temp
        )
        {
#if 1
            // launch the fused kernel function for a 2-sized polynomial
            native::ckks_coeff_square<T>(queue, q_base_sz, coeff_count, inp_output, xe_modulus, inv_mod2, wait);
#else
            // TODO: provide a non-GPU fused kernel function.
            // Compute c0^2
            auto encr0_ptr = (inp_output + 0 * coeff_count * q_base_sz);
            auto temp0_ptr = (temp + 0 * coeff_count * q_base_sz);
            native::poly_coeff_prod_mod<T>(queue, 1, q_base_sz, coeff_count, encr0_ptr, encr0_ptr, xe_modulus, inv_mod2, temp0_ptr);

            // Compute 2*c0*c1
            auto encr1_ptr = (inp_output + 1 * coeff_count * q_base_sz);
            auto temp1_ptr = (temp + 1 * coeff_count * q_base_sz);
            native::poly_coeff_prod_mod<T>(queue, 1, q_base_sz, coeff_count, encr0_ptr, encr1_ptr, xe_modulus, inv_mod2, temp1_ptr);
            native::poly_coeff_add_mod<T>(queue, 1, q_base_sz, coeff_count, temp1_ptr, temp1_ptr, xe_modulus, temp1_ptr);


            // Compute c1^2
            auto encr2_ptr = (inp_output + 2 * coeff_count * q_base_sz);
            native::poly_coeff_prod_mod<T>(queue, 1, q_base_sz, coeff_count, encr1_ptr, encr1_ptr, xe_modulus, inv_mod2, encr2_ptr);
            xehe::gpu_copy(queue, inp_output, temp, (2 * coeff_count * q_base_sz), wait);
#endif //BUILD_WITH_IGPU
        }



        /**********************************************************
        *   CKKS relinearization
        *
        */
        template<typename T>
        void ckks_relin(
                cl::sycl::queue &queue,
                int key_component_count,        // number of output compunents
                int rns_modulus_size,           // extended rns base size
                int decomp_modulus_size,        // rns based size
                int coeff_count,                // order of poly
                const T* xe_modulus,            // RNS modules
                const T* xe_roots_op,  // powers of prime roots of unity mod each RNS prime
                const T* xe_roots_quo,  // powers of prime roots of unity mod each RNS prime
                const T* xe_inv_roots_op, // powers of invert prime roots of unity mod each RNS prime
                const T* xe_inv_roots_quo, // powers of invert prime roots of unity mod each RNS prime
                const T* xe_inv_degree_modulus_op, // n^(-1) modulo q; scaled by in iNTT
                const T* xe_inv_degree_modulus_quo, // n^(-1) modulo q; scaled by in iNTT
                const T* xe_key_vector,
                int key_modulus_size,
                const T* xe_key_modulus,
                const T* xe_key_inv2,
                const T* xe_modswitch_factors_op,
                const T* xe_modswitch_factors_quo,
                T* input,
                T* output,
                //  temp prod
                T* temp_p_prod,
            // temp operand for parallel processing
                T *temp_operand_pll,
                bool wait = true
        )
        {
            int coeff_count_power = int(ilogb(coeff_count));
            // transform encripted into "time" domain
            // to do modulo ops
            xehe::util::inverse_ntt_negacyclic_harvey<T>(
                    queue,
                    1,
                    int(decomp_modulus_size),
                    coeff_count_power,
                    input,
                    xe_modulus,
                    xe_inv_roots_op, xe_inv_roots_quo,
                    xe_inv_degree_modulus_op, xe_inv_degree_modulus_quo);


            // ModUp to special prime P
            // rln modulus
            //
            rln_keys_mod(queue,
                         rns_modulus_size, decomp_modulus_size, key_modulus_size, coeff_count,
                         xe_key_vector,
                         xe_key_modulus,
                         xe_key_inv2,
                         input,
                         temp_operand_pll);

            // transform back into NTT domain
            //
#if 0
            for (size_t i = 0; i < rns_modulus_size; ++i)
            {
                // one modulus at a time
                // mod last key modulus
                size_t k = (i == decomp_modulus_size ? key_modulus_size - 1 : i);

                auto temp_operand = temp_operand_pll + i * decomp_modulus_size * coeff_count;
                // return into NTT space

                xehe::util::ntt_negacyclic_harvey<T>(
                    queue,
                    decomp_modulus_size,
                    1,
                    coeff_count_power,
                    temp_operand,
                    xe_modulus + k,
                    xe_roots_op + k * coeff_count,
                    xe_roots_quo + k * coeff_count,
                    nullptr, //scalar_op
                    nullptr, //scalar_quo
                    true); // lazy





            }// for (size_t i = 0; i < rns_modulus_size; ++i)
#else
            // legacy implementation
            // for (size_t i = 0; i < rns_modulus_size; ++i)
            // {
            //     // one modulus at a time
            //     // mod last key modulus
            //     size_t k = (i == decomp_modulus_size ? key_modulus_size - 1 : i);
            //     auto temp_operand = temp_operand_pll + i * decomp_modulus_size * coeff_count;
            //     // return into NTT space
            //     xehe::util::ntt_negacyclic_harvey<T>(
            //         queue,
            //         decomp_modulus_size,
            //         1,
            //         coeff_count_power,
            //         temp_operand,
            //         xe_modulus + k,
            //         xe_roots_op + k * coeff_count,
            //         xe_roots_quo + k * coeff_count,
            //         nullptr, //scalar_op
            //         nullptr, //scalar_quo
            //         true); // lazy
            // }// for (size_t i = 0; i < rns_modulus_size; ++i)

            xehe::util::ntt_negacyclic_harvey<T>(
                queue,
                key_modulus_size,
                decomp_modulus_size,
                rns_modulus_size,
                coeff_count_power,
                temp_operand_pll,
                xe_modulus,
                xe_roots_op,
                xe_roots_quo,
                nullptr, //scalar_op
                nullptr, //scalar_quo
                true);   //lazy

#endif

            // dot product with relianirization keys

            // HW loops over rns_modulus_size, key_component_count, coeff_count
            // SW loop over decomp_modulus_size

            // accumulate
            // reduce by qi
            int lazy_reduction_summand_bound = xehe::native::max_mul_accum_mod_count<T>();

            rln_keys_dotprod_mod_lazy<T>(queue,
                                         rns_modulus_size, decomp_modulus_size, key_component_count, key_modulus_size, coeff_count,
                                         xe_key_vector,
                                         xe_key_modulus,
                                         xe_key_inv2,
                                         temp_operand_pll,
                                         temp_p_prod,
                                         lazy_reduction_summand_bound);


            // ATTENTION: reuse input
            auto temp_ntt = input;

            //
            // ATTENTION: reuse 2 last poly spaces in temp_p_prod
            // TODO: to run all components in one launch

            // ops over the last P key

            // return to "time" domain
            // for direct mod operation on poly coefficents
#if 0
            for (int kc = 0; kc < key_component_count; ++kc)
            {
                auto k = key_modulus_size - 1;

                xehe::util::inverse_ntt_negacyclic_harvey<T>(
                        queue,
                        1,
                        1,
                        coeff_count_power,
                        temp_p_prod + (kc * rns_modulus_size + decomp_modulus_size) * coeff_count,
                        xe_modulus + k,
                        xe_inv_roots_op + k * coeff_count,
                        xe_inv_roots_quo + k * coeff_count,
                        xe_inv_degree_modulus_op + k, //scalar
                        xe_inv_degree_modulus_quo + k, //scalar
                        true); // lazy
            }
#else
            xehe::util::inverse_ntt_negacyclic_harvey<T>(
                    queue,
                    key_modulus_size,
                    key_component_count,
                    rns_modulus_size,
                    coeff_count_power,
                    temp_p_prod + decomp_modulus_size * coeff_count,
                    xe_modulus,
                    xe_inv_roots_op,
                    xe_inv_roots_quo,
                    xe_inv_degree_modulus_op, //scalar
                    xe_inv_degree_modulus_quo, //scalar
                    true); // lazy

#endif
            // Add (P-1)/2 to change from flooring to rounding.
            // (ct mod 4qk) mod qi
            // Lazy substraction, results in [0, 2*qi), since fix is in [0, qi].
            rln_mod4qk_modqi<T>(
                    queue,
                    rns_modulus_size, decomp_modulus_size,
                    key_component_count, key_modulus_size, coeff_count,
                    xe_key_modulus,
                    xe_key_inv2,
                    temp_p_prod,
                    temp_ntt);

            // all components in one launch

            // This ntt_negacyclic_harvey_lazy results in [0, 4*qi).
            xehe::util::ntt_negacyclic_harvey<T>(
                    queue,
                    key_component_count,
                    decomp_modulus_size,
                    coeff_count_power,
                    temp_ntt,
                    xe_modulus,
                    xe_roots_op,
                    xe_roots_quo,
                    nullptr, //scalar
                    nullptr, //scalar
                    true); // lazy


            rln_ctmodqi_ctmodqk_modqi(
                    queue,
                    rns_modulus_size, key_component_count, decomp_modulus_size, coeff_count,
                    xe_key_modulus,
                    xe_modswitch_factors_op,
                    xe_modswitch_factors_quo,
                    temp_ntt,
                    temp_p_prod,
                    output,
                    wait); // wait

        }

        /**********************************************************
        *   CKKS rescale
        *
        */
        template<typename T>
        void ckks_divide_and_round_q_last(
                cl::sycl::queue &queue,
                size_t encrypted_size,       // n_polys in encr message
                size_t q_base_sz,            // current RNS base size; will be decreased by 1
                size_t coeff_count,          // poly order
                size_t next_coeff_modulus_size, // next RNS base size
                const T* xe_modulus,            // RNS modules
                const T* xe_inv2,        // 2^64/prime values
                const T* xe_roots_op,  // powers of prime roots of unity mod each RNS prime
                const T* xe_roots_quo,  // powers of prime roots of unity mod each RNS prime
                const T* xe_inv_roots_op, // powers of invert prime roots of unity mod each RNS prime
                const T* xe_inv_roots_quo, // powers of invert prime roots of unity mod each RNS prime
                const T* xe_inv_degree_modulus_op, // n^(-1) modulo q; scaled by in iNTT
                const T* xe_inv_degree_modulus_quo, // n^(-1) modulo q; scaled by in iNTT
                const T* xe_inv_q_last_mod_q_op,  // 1/qi mod last qi for each qi in [0, q_base_sz-1]
                const T* xe_inv_q_last_mod_q_quo,  // 1/qi mod last qi for each qi in [0, q_base_sz-1]
                T* input,
                T* output,
                bool wait = true
        )
        {
            // Convert to non-NTT form
            int coeff_count_power = int(ilogb(coeff_count));
            auto k = (q_base_sz - 1);
#if 0
            for (int ct = 0; ct < encrypted_size; ++ct)
            {
                //auto last_inp_ptr = input + ct * RNS_sz + k * coeff_count;
                //std::cout << "xehe inv_nt1 " << last_inp_ptr[0] << " " << last_inp_ptr[coeff_count - 1] << std::endl;
                xehe::util::inverse_ntt_negacyclic_harvey<T>(
                        queue,
                        1,
                        1,
                        coeff_count_power,
                        input + ct * q_base_sz * coeff_count + k * coeff_count,//last_inp_ptr, //last_input
                        xe_modulus + k,
                        xe_inv_roots_op + k * coeff_count,
                        xe_inv_roots_quo + k * coeff_count,
                        xe_inv_degree_modulus_op + k,
                        xe_inv_degree_modulus_quo + k);

                //std::cout << "xehe inv_nt1 " << last_inp_ptr[0] << " " << last_inp_ptr[coeff_count-1] << std::endl;

            }
#else
            xehe::util::inverse_ntt_negacyclic_harvey_rescale<T>(
                    queue,
                    encrypted_size,
                    q_base_sz,
                    coeff_count_power,
                    input + k * coeff_count,//last_inp_ptr, //last_input
                    xe_modulus + k,
                    xe_inv_roots_op + k * coeff_count,
                    xe_inv_roots_quo + k * coeff_count,
                    xe_inv_degree_modulus_op + k,
                    xe_inv_degree_modulus_quo + k);
#endif
            rscl_mod4qk_modqi<T>(queue,
                    int(encrypted_size), int(q_base_sz), int(coeff_count),
                    xe_modulus,
                    xe_inv2,
                    (const T*)input, //last_input
                    output);

            // (q_sz -1) in one shot


            xehe::util::ntt_negacyclic_harvey<T>(queue,
                            encrypted_size,
                            int(k),
                            coeff_count_power,
                            output,
                            xe_modulus,
                            xe_roots_op,
                            xe_roots_quo);


            rscl_ctmodqi_ctmodqk_modqi<T>(queue,
                    encrypted_size, q_base_sz, coeff_count,
                    xe_modulus,
                    xe_inv_q_last_mod_q_op,
                    xe_inv_q_last_mod_q_quo,
                    (const T*)output,
                    (const T*)input,
                    output,
                    wait); // wait
        }

        /**********************************************************
        *   CKKS multiply
        *
        */
        // naive  m polys x n polys
        //
        template<typename T>
        void ckks_multiply(
                cl::sycl::queue &queue,
                size_t enc1_sz, // n polys
                size_t enc2_sz,           // m polys
                size_t max_sz,            // n*m
                int q_base_sz,            // RNS base size
                int n,                // log(order_of_poly)
                const T* poly1, const T* poly2, // enc1, enc2
                const T* xe_modulus, const T* inv_mod2, // RNS, inv RNS
                T* poly_res,
                bool wait = true
        )
        {

            xehe::native::ckks_coeff_fused_prod_mod<T>(queue,q_base_sz,n,poly1,poly2,xe_modulus,inv_mod2, poly_res, wait);
        }


        template<typename T>
        void ckks_multiply(
            cl::sycl::queue& queue,
            size_t enc1_sz, // n polys
            size_t enc2_sz,           // m polys
            size_t max_sz,            // n*m
            int q_base_sz,            // RNS base size
            int n,                // log(order_of_poly)
            const T* poly1, const T* poly2, // enc1, enc2
            const T* xe_modulus, const T* inv_mod2, // RNS, inv RNS
            T* poly_res,             // destination, result
            T* prod,                // temp prod
            bool wait = true
        )
        {

            size_t RNS_poly_len = n * q_base_sz;
            for (int p = 0; p < max_sz; ++p)
            {
                int curr_encrypted1_last = (p < (enc1_sz - 1)) ? p : enc1_sz - 1;
                int curr_encrypted2_first = (p < (enc2_sz - 1)) ? p : enc2_sz - 1;
                int curr_encrypted1_first = p - curr_encrypted2_first;

                // The total number of dyadic products is now easy to compute
                int steps = curr_encrypted1_last - curr_encrypted1_first + 1;

                for (int i = 0, s = curr_encrypted1_first, rs = curr_encrypted2_first; i < steps; i++, ++s, --rs)
                {
                    auto prod_ptr = (i == 0) ? (poly_res + p * RNS_poly_len) : prod;
                    //auto prod_h_ptr = (i == 0) ? (poly_res_h + p * RNS_poly_len) : prod_h;

                    xehe::native::poly_coeff_prod_mod<T>(queue,
                        1, q_base_sz, n,
                        poly1 + s * RNS_poly_len, poly2 + rs * RNS_poly_len,
                        xe_modulus, inv_mod2,
                        prod_ptr);

                    if (i > 0)
                    {
                        xehe::native::poly_coeff_add_mod<T>(queue,
                            1, 1, 1, q_base_sz, n,
                            prod, poly_res + p * RNS_poly_len, xe_modulus, poly_res + p * RNS_poly_len);

                    }

                }
            }

            if (wait) {
                queue.wait();
            }
        }



    } // namespace eval
} // namespace xehe

#endif // #ifdef BUILD_WITH_IGPU

#endif //XEHE_EVALUATOR_HPP