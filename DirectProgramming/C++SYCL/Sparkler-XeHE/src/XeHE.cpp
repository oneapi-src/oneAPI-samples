/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#include "../include/XeHE.hpp"

#define LIB_ON

#ifdef LIB_ON
#include "native/xe_polyops.hpp"
#include "eval/rotate.hpp"
#include "eval/xe_evaluator.hpp"
#include "util/memcache.hpp"
#endif

namespace xehe {
    namespace ext {
        class XeHE;


        class XeHE {
        public:
            XeHE(void)
            {               
            }

            ~XeHE(void)
            {
                // dealloc();
                for (auto memcache : memcache_) memcache->dealloc();
                // memcache_.clear();
            }
            //cl::sycl::free(get_data_ptr(), get_queue());
            void init(void)
            {
                if (!initialised_)
                {
                    std::cout << "Initializing " << ctx_.tiles() << " Memcache(s)" << std::endl;
                    for (size_t i = 0; i < ctx_.tiles(); i++) 
                        memcache_.push_back(new Memcache(get_context_queue(i)));
                    initialised_ = true;
                }
            }
           // size_t get_queue_size();

            cl::sycl::queue& get_queue(size_t index = 0)
            {
                init();

                if (index >= memcache_.size()) {
                    std::cout << "Error: Queue Index Out of Bound" << std::endl;
                    // TODO: throw exception
                }
                return memcache_[index]->get_queue();
            } 
            
            Memcache* get_memcache(size_t q_idx = 0)
            {
                init();

                if (q_idx >= memcache_.size()) {
                    std::cout << "Error: Queue Index Out of Bound" << std::endl;
                    // TODO: throw exception
                    return nullptr;
                }
                return memcache_[q_idx];
            } 
            
            template<typename T>
            T* pool_alloc(size_t buffer_size, size_t & capacity, size_t q_idx = 0, bool uncached = false){
                init();

                if (q_idx >= memcache_.size()) {
                    std::cout << "Queue Index Out of Bound Error: " << q_idx << std::endl;
                    // TODO: throw exception
                    return nullptr;
                }
                return memcache_[q_idx]->pool_alloc<T>(buffer_size, capacity, uncached);
            }

            
            void pool_free(void * data, size_t capacity, size_t q_idx = 0, bool uncached = false)
            {
                init();

                if (q_idx >= memcache_.size()) {
                    std::cout << "Queue Index Out of Bound Error: " << q_idx << std::endl;
                    // TODO: throw exception
                    return;
                }
                memcache_[q_idx]->pool_free(data, capacity, uncached);
            }

        private:

            bool initialised_ = false;

            const cl::sycl::queue& get_context_queue(size_t idx) const
            {
                return ctx_.queue(idx);
            }

            dpcpp::Context ctx_;
            std::vector<Memcache*> memcache_;
        };

        // cleaner initialization
        static XeHE s_pXeHELib;        
        static XeHE* pXeHELib = nullptr;

        XeHE& get_xehe_lib(void)
        {
            if (pXeHELib == nullptr)
            {
                pXeHELib = &s_pXeHELib;
            }
            return(*pXeHELib);
        }


        cl::sycl::queue& get_queue(size_t index = 0)
        {
            if (pXeHELib == nullptr)
            {
                pXeHELib = &s_pXeHELib;
            }
            return(pXeHELib->get_queue(index));
        }

        // interfaces for clearing the EventCollector data
        void clear_events(){
            EventCollector::clear_events();
        }

        // interface for summurizing the profilling data in EventCollector
        void process_events(){
            EventCollector::process_events();
        }

        // process all the events in EventCollector into an operation export record, append it to export table
        void add_operation(std::string op_name, double avg_external_time, int loops){
            EventCollector::add_operation(op_name, avg_external_time, loops);
        }

        // export the export table to a txt file
        void export_table(std::string file_name){
            EventCollector::export_table(file_name);
        }

        // clear the export table in record.
        void clear_export_table(){
            EventCollector::clear_export_table();
        }

        // add header for the export table, describing the parameter choice
        void add_header(int poly_order, int data_bound, int delta_bits, int rns_size){
            EventCollector::add_header(poly_order, data_bound, delta_bits, rns_size);
        }

        void XeHE_negate(
            int n_polys,
            const std::shared_ptr<Buffer<uint64_t>> values,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> result,
            bool wait)
        {

#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();
            // Negate every poly in the array
            xehe::native::poly_coeff_neg_mod<uint64_t>(queue,
                n_polys,
                xehe_mem_ctx.rns_base_size,
                xehe_mem_ctx.coeff_count,
                values->get_data_ptr(),
                xehe_mem_ctx.xe_modulus->get_data_ptr(),
                result->get_data_ptr(),
                wait);
#endif
        }

        void XeHE_add(
            int n_polys,
            int n_poly1,
            int n_poly2,
            const std::shared_ptr<Buffer<uint64_t>> oprnd1,
            const std::shared_ptr<Buffer<uint64_t>> oprnd2,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> result,
            bool wait
        )
        {
#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();

//#if _NO_WAIT_
//            // defer deletion
//            s_XeHE_DefferedDel[oprnd1->get_data_ptr()] = oprnd1;
//            s_XeHE_DefferedDel[oprnd2->get_data_ptr()] = oprnd2;
//            s_XeHE_DefferedDel[result->get_data_ptr()] = result;
//#endif
            xehe::native::poly_coeff_add_mod<uint64_t>(queue,
                n_polys, n_poly1, n_poly2,
                xehe_mem_ctx.rns_base_size,
                xehe_mem_ctx.coeff_count,
                oprnd1->get_data_ptr(),
                oprnd2->get_data_ptr(),
                xehe_mem_ctx.xe_modulus->get_data_ptr(),
                result->get_data_ptr(),
                wait);
#endif
        }



       void XeHE_sub(
                int n_polys,
                int n_poly1,
                int n_poly2,
                const std::shared_ptr<Buffer<uint64_t>> oprnd1,
                const std::shared_ptr<Buffer<uint64_t>> oprnd2,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> result,
                bool wait
        )
        {
#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();

//#if _NO_WAIT_
//           // defer deletion
//            s_XeHE_DefferedDel[oprnd1->get_data_ptr()] = oprnd1;
//            s_XeHE_DefferedDel[oprnd2->get_data_ptr()] = oprnd2;
//            s_XeHE_DefferedDel[result->get_data_ptr()] = result;
//#endif

            xehe::native::poly_coeff_sub_mod<uint64_t>(queue,
                n_polys, n_poly1, n_poly2,
                xehe_mem_ctx.rns_base_size,
                xehe_mem_ctx.coeff_count,
                oprnd1->get_data_ptr(),
                oprnd2->get_data_ptr(),
                xehe_mem_ctx.xe_modulus->get_data_ptr(),
                result->get_data_ptr(),
                wait);
#endif
        }


       void XeHE_multiply_plain_ckks(
            int n_polys,
           const std::shared_ptr<Buffer<uint64_t>> oprnd1,
           const std::shared_ptr<Buffer<uint64_t>> oprnd2,
           const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
           std::shared_ptr<Buffer<uint64_t>>  result,
           bool wait)
        {

#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();

//#if _NO_WAIT_
//           // defer deletion
//            s_XeHE_DefferedDel[oprnd1->get_data_ptr()] = oprnd1;
//            s_XeHE_DefferedDel[oprnd2->get_data_ptr()] = oprnd2;
//            s_XeHE_DefferedDel[result->get_data_ptr()] = result;
//#endif

#ifdef BUILD_WITH_IGPU
            xehe::native::poly_coeff_prod_mod_plain<uint64_t>(queue,
                n_polys,
                xehe_mem_ctx.rns_base_size,
                xehe_mem_ctx.coeff_count,
                (oprnd1->get_data_ptr()),
                oprnd2->get_data_ptr(),
                xehe_mem_ctx.xe_modulus->get_data_ptr(),
                xehe_mem_ctx.xe_inv2->get_data_ptr(),
                (result->get_data_ptr()),
                wait
            );
#else
            // TODO: FUSE IT ON HOST
            auto plain_len = xehe_mem_ctx.rns_base_size * xehe_mem_ctx.coeff_count;

            for (uint64_t ct_id = 0; ct_id < n_polys; ++ct_id)
            {
                xehe::native::poly_coeff_prod_mod<uint64_t>(queue,
                    1,
                    xehe_mem_ctx.rns_base_size,
                    xehe_mem_ctx.coeff_count,
                    (oprnd1->get_data_ptr() + ct_id * plain_len),
                    oprnd2->get_data_ptr(),
                    xehe_mem_ctx.xe_modulus->get_data_ptr(),
                    xehe_mem_ctx.xe_inv2->get_data_ptr(),
                    (result->get_data_ptr() + ct_id * plain_len)
                );
            }
#endif // BUILD_WITH_IGPU
#endif // LIB_ON
        }

        // MADPlain: result = oprnd_add + oprnd_mul * oprnd_plain
       void XeHE_multiply_plain_ckks_add(
            int n_polys,
            int n_poly_add,
            int n_poly_mul,
            const std::shared_ptr<Buffer<uint64_t>> oprnd_add,
            const std::shared_ptr<Buffer<uint64_t>> oprnd_mul,
            const std::shared_ptr<Buffer<uint64_t>> oprnd_plain,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>>  result,
            bool wait)
        {

#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();

//#if _NO_WAIT_
//            // defer deletion
//            s_XeHE_DefferedDel[oprnd_add->get_data_ptr()] = oprnd_add;
//            s_XeHE_DefferedDel[oprnd_mul->get_data_ptr()] = oprnd_mul;
//            s_XeHE_DefferedDel[oprnd_plain->get_data_ptr()] = oprnd_plain;
//            s_XeHE_DefferedDel[result->get_data_ptr()] = result;
//#endif

#ifdef BUILD_WITH_IGPU
            xehe::native::poly_coeff_prod_mod_plain_add<uint64_t>(queue,
                n_polys, n_poly_add, n_poly_mul,
                xehe_mem_ctx.rns_base_size,
                xehe_mem_ctx.coeff_count,
                oprnd_add->get_data_ptr(),
                oprnd_mul->get_data_ptr(),
                oprnd_plain->get_data_ptr(),
                xehe_mem_ctx.xe_modulus->get_data_ptr(),
                xehe_mem_ctx.xe_inv2->get_data_ptr(),
                result->get_data_ptr(),
                wait
            );
#else
            // TODO: the host implementation
#endif // BUILD_WITH_IGPU
#endif // LIB_ON
        }

        // MADPlain: result = oprnd_add + oprnd2 * oprnd3
        // here we assumed n_poly2 = 2, n_poly3 = 2, max_sz = 3
       void XeHE_multiply_ckks_add(
            int n_polys,
            int n_poly_add,
            int n_poly2,
            int n_poly3,
            const std::shared_ptr<Buffer<uint64_t>> oprnd_add,
            const std::shared_ptr<Buffer<uint64_t>> oprnd2,
            const std::shared_ptr<Buffer<uint64_t>> oprnd3,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> result,
            bool wait)
        {

#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();

//#if _NO_WAIT_
//            // defer deletion
//            s_XeHE_DefferedDel[oprnd_add->get_data_ptr()] = oprnd_add;
//            s_XeHE_DefferedDel[oprnd2->get_data_ptr()] = oprnd2;
//            s_XeHE_DefferedDel[oprnd3->get_data_ptr()] = oprnd3;
//            s_XeHE_DefferedDel[result->get_data_ptr()] = result;
//#endif

#ifdef BUILD_WITH_IGPU
            xehe::native::poly_coeff_prod_mod_add<uint64_t>(queue,
                n_polys, n_poly_add, n_poly2, n_poly3,
                xehe_mem_ctx.rns_base_size,
                xehe_mem_ctx.coeff_count,
                oprnd_add->get_data_ptr(),
                oprnd2->get_data_ptr(),
                oprnd3->get_data_ptr(),
                xehe_mem_ctx.xe_modulus->get_data_ptr(),
                xehe_mem_ctx.xe_inv2->get_data_ptr(),
                result->get_data_ptr(),
                wait
            );
#else
            // TODO: the host implementation
            auto h_op1 = oprnd1->get_host_ptr();
            auto h_op2 = oprnd2->get_host_ptr();
            auto h_op3 = oprnd3->get_host_ptr();
            auto h_mod = xehe_mem_ctx.xe_modulus->get_host_ptr();
            auto h_inv2 = xehe_mem_ctx.xe_inv2->get_host_ptr();
            auto res_sz = result->get_size();
            std::vector<uint64_t> h_res(res_sz, uint64_t(0xbeefbeefbeefbeef));
            xehe::native::poly_coeff_prod_mod_add<uint64_t>(
                n_polys,
                xehe_mem_ctx.rns_base_size,
                xehe_mem_ctx.coeff_count,
                h_op1,
                h_op2,
                h_op3,
                h_mod,
                h_inv2,
                h_res.data()
                );

#endif // BUILD_WITH_IGPU

            //auto d_res = result->get_host_ptr();
#endif // LIB_ON
        }

       void XeHE_mul_ckks(
           size_t max_sz,            // n*m - 1
           size_t enc1_sz, // n polys
           size_t enc2_sz,           // m polys
           const std::shared_ptr<Buffer<uint64_t>>  poly1,  // enc1
           const std::shared_ptr<Buffer<uint64_t>>  poly2, //  enc2
           const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
           std::shared_ptr<Buffer<uint64_t>> poly_res,
           bool wait)
        {
#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();           


//#if _NO_WAIT_
//            // defere deletion
//            s_XeHE_DefferedDel[poly1->get_data_ptr()] = poly1;
//            s_XeHE_DefferedDel[poly2->get_data_ptr()] = poly2;
//            s_XeHE_DefferedDel[poly_res->get_data_ptr()] = poly_res;
//#endif



            // over dest polynomials
            if (enc1_sz == 2 && enc2_sz == 2) {
                xehe::eval::ckks_multiply<uint64_t>(
                    queue,
                    enc1_sz,
                    enc2_sz,
                    max_sz,
                    xehe_mem_ctx.rns_base_size,
                    xehe_mem_ctx.coeff_count,
                    poly1->get_data_ptr(),
                    poly2->get_data_ptr(),
                    xehe_mem_ctx.xe_modulus->get_data_ptr(),
                    xehe_mem_ctx.xe_inv2->get_data_ptr(),
                    poly_res->get_data_ptr(),
                    wait
                    );
            }
            else
            {
                //static s_prev_sz = 0;
                size_t RNS_poly_len = xehe_mem_ctx.rns_base_size * xehe_mem_ctx.coeff_count;
                //static
                std::shared_ptr< xehe::ext::Buffer<uint64_t>> prod_buf = nullptr;


//#if _NO_WAIT_
//                // defere deletion
//                s_XeHE_DefferedDel[prod_buf->get_data_ptr()] = prod_buf;
//#endif
                prod_buf = xehe::ext::XeHE_malloc<uint64_t>(RNS_poly_len);
                xehe::eval::ckks_multiply<uint64_t>(
                    queue,
                    enc1_sz,
                    enc2_sz,
                    max_sz,
                    xehe_mem_ctx.rns_base_size,
                    xehe_mem_ctx.coeff_count,
                    poly1->get_data_ptr(),
                    poly2->get_data_ptr(),
                    xehe_mem_ctx.xe_modulus->get_data_ptr(),
                    xehe_mem_ctx.xe_inv2->get_data_ptr(),
                    poly_res->get_data_ptr(),
                    // temp prod
                    prod_buf->get_data_ptr(),
                    wait
                    );

            }
#endif

        }



       void XeHE_relinearize(
           size_t kswitch_keys_index,      // index in the array of keys
           const XeHE_mem_keys<uint64_t>& xehe_key_mem,
           const XeHE_mem_context<uint64_t>& xehe_key_mem_ctx,
           std::shared_ptr<Buffer<uint64_t>> input,
           std::shared_ptr<Buffer<uint64_t>> output,
           bool wait
       )
       {
#ifdef LIB_ON
           cl::sycl::queue& queue = get_queue();
            int key_component_count = xehe_key_mem.n_polys;       // number of output compunents
            int rns_modulus_size = xehe_key_mem.rns_base_size + 1;           // extended rns base size
            int decomp_modulus_size = xehe_key_mem.rns_base_size;       // rns based size
            int coeff_count = xehe_key_mem_ctx.coeff_count;               // order of poly           
            //static 
            std::shared_ptr< xehe::ext::Buffer<uint64_t>> p_prod_buf;
            //static 
            std::shared_ptr< xehe::ext::Buffer<uint64_t>> operand_pll_buf;
            size_t p_prod_buf_sz = (key_component_count * coeff_count * rns_modulus_size);
            //if (p_prod_buf == nullptr || p_prod_buf->get_size() < p_prod_buf_sz)
            {
                p_prod_buf = xehe::ext::XeHE_malloc<uint64_t>(p_prod_buf_sz);
            }
            size_t operand_pll_sz = (rns_modulus_size * decomp_modulus_size * coeff_count);
            //if (operand_pll_buf == nullptr || operand_pll_buf->get_size() < operand_pll_sz)
            {
                operand_pll_buf = xehe::ext::XeHE_malloc<uint64_t>(operand_pll_sz);
            }

           // TODO: FIX ThIS INTERFACE REMOVE xe_inv_modulus, mod_inv_size WITH INV2
           xehe::eval::ckks_relin<uint64_t>(
               queue,
               xehe_key_mem.n_polys,        // number of output compunents
               xehe_key_mem.rns_base_size + 1,           // extended rns base size
               xehe_key_mem.rns_base_size,        // rns based size
               xehe_key_mem_ctx.coeff_count,
               xehe_key_mem_ctx.xe_modulus->get_data_ptr(),            // RNS modules
               xehe_key_mem_ctx.xe_prim_roots_op->get_data_ptr(),  // powers of prime roots of unity mod each RNS prime
               xehe_key_mem_ctx.xe_prim_roots_quo->get_data_ptr(),  // powers of prime roots of unity mod each RNS prime
               xehe_key_mem_ctx.xe_inv_prim_roots_op->get_data_ptr(), // powers of invert prime roots of unity mod each RNS prime
               xehe_key_mem_ctx.xe_inv_prim_roots_quo->get_data_ptr(), // powers of invert prime roots of unity mod each RNS prime
               xehe_key_mem_ctx.xe_inv_degree_op->get_data_ptr(), // n^(-1) modulo q; scaled by in iNTT
               xehe_key_mem_ctx.xe_inv_degree_quo->get_data_ptr(), // n^(-1) modulo q; scaled by in iNTT
               xehe_key_mem.xe_keys->get_data_ptr(),
               xehe_key_mem_ctx.rns_base_size,
               xehe_key_mem_ctx.xe_modulus->get_data_ptr(),
               xehe_key_mem_ctx.xe_inv2->get_data_ptr(),
               xehe_key_mem_ctx.xe_inv_q_last_mod_q_op->get_data_ptr(),
               xehe_key_mem_ctx.xe_inv_q_last_mod_q_quo->get_data_ptr(),
               input->get_data_ptr(),
               output->get_data_ptr(),
               // temps
               p_prod_buf->get_data_ptr(),
               operand_pll_buf->get_data_ptr(),
               wait
               );


#endif
       }



        void XeHE_divide_round_q_last_ckks(
            size_t encrypted_size,       // n_polys in encr message
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> input, // not a constant, reused inside kernel
            std::shared_ptr<Buffer<uint64_t>> output,
            bool wait
        )

        {
#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();

//#if _NO_WAIT_
//            // defer deletion
//            s_XeHE_DefferedDel[input->get_data_ptr()] = input;
//            s_XeHE_DefferedDel[output->get_data_ptr()] = output;
//#endif

            xehe::eval::ckks_divide_and_round_q_last<uint64_t>(
                queue,
                encrypted_size,       // n_polys in encr message
                xehe_mem_ctx.rns_base_size,            // current RNS base size; will be decreased by 1
                xehe_mem_ctx.coeff_count,          // poly order
                xehe_mem_ctx.next_rns_base_size, // next RNS base size
                xehe_mem_ctx.xe_modulus->get_data_ptr(),            // RNS modules
                xehe_mem_ctx.xe_inv2->get_data_ptr(),
                xehe_mem_ctx.xe_prim_roots_op->get_data_ptr(),  // powers of prime roots of unity mod each RNS prime
                xehe_mem_ctx.xe_prim_roots_quo->get_data_ptr(),  // powers of prime roots of unity mod each RNS prime
                xehe_mem_ctx.xe_inv_prim_roots_op->get_data_ptr(), // powers of invert prime roots of unity mod each RNS prime
                xehe_mem_ctx.xe_inv_prim_roots_quo->get_data_ptr(), // powers of invert prime roots of unity mod each RNS prime
                xehe_mem_ctx.xe_inv_degree_op->get_data_ptr(), // n^(-1) modulo q; scaled by in iNTT
                xehe_mem_ctx.xe_inv_degree_quo->get_data_ptr(), // n^(-1) modulo q; scaled by in iNTT
                xehe_mem_ctx.xe_inv_q_last_mod_q_op->get_data_ptr(),  // 1/qi mod last qi for each qi in [0, q_base_sz-1]
                xehe_mem_ctx.xe_inv_q_last_mod_q_quo->get_data_ptr(),  // 1/qi mod last qi for each qi in [0, q_base_sz-1]
                input->get_data_ptr(),
                output->get_data_ptr(),
                wait
            );
#endif

        }



        void XeHE_square(
            int dest_size,                                   // number of output components
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,            // context
            std::shared_ptr<Buffer<uint64_t>> inp_output,     // input/output inplace
            bool wait
        )
        {
#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();

//#if _NO_WAIT_
//            // defer deletion
//            s_XeHE_DefferedDel[inp_output->get_data_ptr()] = inp_output;
//#endif

#if 1
            xehe::eval::ckks_square<uint64_t>(
                queue,
                dest_size,                                // number of output components
                xehe_mem_ctx.rns_base_size,               // rns based size
                xehe_mem_ctx.coeff_count,                  // order of poly
                xehe_mem_ctx.xe_modulus->get_data_ptr(),            // RNS modules
                xehe_mem_ctx.xe_inv2->get_data_ptr(),
                inp_output->get_data_ptr(),                  // input/output inplace
                wait
                );

#else
            size_t alloc_sz = dest_size * xehe_mem_ctx.rns_base_size * xehe_mem_ctx.coeff_count;
            //static 
            std::shared_ptr< xehe::ext::Buffer<uint64_t>> temp_buf;
            //if (temp_buf == nullptr || temp_buf->get_size() < alloc_sz)
            {
                temp_buf = xehe::ext::XeHE_malloc<uint64_t>(alloc_sz);
            }

            xehe::eval::ckks_square<uint64_t>(
                queue,
                dest_size,                                // number of output components
                xehe_mem_ctx.rns_base_size,               // rns based size
                xehe_mem_ctx.coeff_count,                  // order of poly
                xehe_mem_ctx.xe_modulus->get_data_ptr(),            // RNS modules
                xehe_mem_ctx.xe_inv2->get_data_ptr(),
                inp_output->get_data_ptr(),                  // input/output inplace
                temp_buf->get_data_ptr()                
            );
#endif
#endif
        }


        void XeHE_mod_switch_ckks(
            int dest_size,        // number of output compunents
            int dest_q_base_sz,   // destination rns base size
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx, // context
            const std::shared_ptr<Buffer<uint64_t>> input,
            std::shared_ptr<Buffer<uint64_t>> output,
            bool wait)
        {
#ifdef LIB_ON
            cl::sycl::queue& queue = get_queue();
//#if _NO_WAIT_
//            // defer deletion
//            s_XeHE_DefferedDel[input->get_data_ptr()] = input;
//            s_XeHE_DefferedDel[output->get_data_ptr()] = output;
//#endif
            xehe::eval::ckks_mod_switch_drop_to_next<uint64_t>(
                queue,
                dest_size,        // number of output compunents
                dest_q_base_sz,   // destination rns base size
                xehe_mem_ctx.rns_base_size,        // input rns based size
                xehe_mem_ctx.coeff_count,                // order of poly
                input->get_data_ptr(),
                output->get_data_ptr(),
                wait
            );
#endif
        }


        static std::mutex permute_lock_;
        static void generate_permute_table_ntt(
            cl::sycl::queue queue,
            uint32_t galois_elt,
            uint32_t index,
            size_t coeff_count,
            uint32_t* index_ptr,
            bool wait);
            
        void XeHE_permute(
            uint32_t encryp_sz,
            uint32_t q_base_sz,
            uint32_t coeff_count,
            uint32_t galois_elt,
            uint32_t index,
            const std::shared_ptr<Buffer<uint64_t>> input,
            std::shared_ptr<Buffer<uint64_t>> output,
            bool wait
        )
        {
#ifdef LIB_ON

            cl::sycl::queue& queue = get_queue();
#if 0
            auto index_buf = xehe::ext::XeHE_malloc<uint32_t>(coeff_count);
            auto perm_table = index_buf->get_data_ptr(); 
#else            
            
            static std::vector<std::shared_ptr< xehe::ext::Buffer<uint32_t>>> index_buf_vec;
            {
                std::lock_guard<std::mutex> lk(permute_lock_);
                
                if (index_buf_vec.size() <= index)
                {
                    index_buf_vec.resize(index+1);
                }
                if (index_buf_vec[index] == nullptr || index_buf_vec[index]->get_size() < coeff_count)
                {
                    index_buf_vec[index] = xehe::ext::XeHE_malloc<uint32_t>(coeff_count);
                }
            }
            // should be valid pointer at this point
            auto perm_table = index_buf_vec[index]->get_data_ptr(); // (*perm_table)[index];
#endif            
            generate_permute_table_ntt(queue, galois_elt, index, coeff_count, perm_table, false); // do not wait

            xehe::eval::permute<uint64_t>(queue,
                                          encryp_sz,
                                          q_base_sz,
                                          coeff_count,
                                          perm_table,
                                          input->get_data_ptr(),
                                          output->get_data_ptr(),
                                          wait);
#endif
        }


        static void generate_permute_table_ntt(cl::sycl::queue queue, uint32_t galois_elt, uint32_t index, size_t coeff_count, uint32_t* index_ptr, bool wait)
        {
#ifdef LIB_ON



            int coeff_count_power = int(ilogb(coeff_count));

            xehe::eval::permute_index(queue,
                                      coeff_count,
                                      coeff_count_power,
                                      galois_elt,
                                      index_ptr,
                                      wait
            );

#else
            static uint32_t fake = 0;
            static auto fake_ptr = &fake;
#endif 

        }

        template<typename T>
        void XeHE_NTT(
                int poly_num,
                const XeHE_mem_context<T>& xehe_mem_ctx,
                std::shared_ptr<Buffer<T>> modulus,
                std::shared_ptr<Buffer<T>> roots_op,
                std::shared_ptr<Buffer<T>> roots_quo,
                std::shared_ptr<Buffer<T>> values,
                std::shared_ptr<Buffer<T>> scalar_op,
                std::shared_ptr<Buffer<T>> scalar_quo,
                size_t q_idx,
                bool lazy,
                bool wait)
        {
            // std::cout << "NTT operation is on queue #" << q_idx << std::endl;
            // std::cout << "values is on queue #" << values->get_q_idx() << std::endl;
            // std::cout << "xehe_mem_ctx.xe_modulus is on queue #" << xehe_mem_ctx.xe_modulus->get_q_idx() << std::endl;
            cl::sycl::queue& queue = get_queue(q_idx);

            int coeff_count_power = int(ilogb(xehe_mem_ctx.coeff_count));
            T *s_op = (scalar_op == nullptr) ? nullptr : scalar_op->get_data_ptr();
            T *s_quo = (scalar_quo == nullptr) ? nullptr : scalar_quo->get_data_ptr();
            util::ntt_negacyclic_harvey<T>(
                    queue,
                    poly_num,  // poly num
                    xehe_mem_ctx.rns_base_size,   // q_base_size
                    coeff_count_power, // log_n
                    values->get_data_ptr(), // values
                    modulus->get_data_ptr(), // modulus
                    roots_op->get_data_ptr(), // roots
                    roots_quo->get_data_ptr(),
                    s_op,
                    s_quo,
                    lazy, wait);
        }

        template<typename T>
        void XeHE_invNTT(
                int poly_num,
                const XeHE_mem_context<T>& xehe_mem_ctx,
                std::shared_ptr<Buffer<T>> modulus,
                std::shared_ptr<Buffer<T>> roots_op,
                std::shared_ptr<Buffer<T>> roots_quo,
                std::shared_ptr<Buffer<T>> values,
                std::shared_ptr<Buffer<T>> scalar_op,
                std::shared_ptr<Buffer<T>> scalar_quo,
                size_t q_idx,
                bool lazy,
                bool wait)
        {
            // std::cout << "iNTT operation is on queue #" << q_idx << std::endl;
            // std::cout << "values is on queue #" << values->get_q_idx() << std::endl;
            // std::cout << "xehe_mem_ctx.xe_modulus is on queue #" << xehe_mem_ctx.xe_modulus->get_q_idx() << std::endl;

            cl::sycl::queue& queue = get_queue(q_idx);

            int coeff_count_power = int(ilogb(xehe_mem_ctx.coeff_count));

            T *s_op = (scalar_op == nullptr) ? nullptr : scalar_op->get_data_ptr();
            T *s_quo = (scalar_quo == nullptr) ? nullptr : scalar_quo->get_data_ptr();

            util::inverse_ntt_negacyclic_harvey<T>(
                    queue,
                    poly_num,  // poly num
                    xehe_mem_ctx.rns_base_size,   // q_base_size
                    coeff_count_power, // log_n
                    values->get_data_ptr(), // values
                    modulus->get_data_ptr(), // modulus
                    roots_op->get_data_ptr(), // roots
                    roots_quo->get_data_ptr(),
                    s_op,
                    s_quo,
                    lazy, wait);
        }

        template void XeHE_NTT<uint64_t>(int poly_num,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> modulus,
                std::shared_ptr<Buffer<uint64_t>> roots_op,
                std::shared_ptr<Buffer<uint64_t>> roots_quo,
                std::shared_ptr<Buffer<uint64_t>> values,
                std::shared_ptr<Buffer<uint64_t>> scalar_op,
                std::shared_ptr<Buffer<uint64_t>> scalar_quo,
                size_t q_idx,
                bool lazy,
                bool wait);

        template void XeHE_NTT<uint32_t>(int poly_num,
                const XeHE_mem_context<uint32_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint32_t>> modulus,
                std::shared_ptr<Buffer<uint32_t>> roots_op,
                std::shared_ptr<Buffer<uint32_t>> roots_quo,
                std::shared_ptr<Buffer<uint32_t>> values,
                std::shared_ptr<Buffer<uint32_t>> scalar_op,
                std::shared_ptr<Buffer<uint32_t>> scalar_quo,
                size_t q_idx,
                bool lazy,
                bool wait);

        template void XeHE_invNTT<uint64_t>(int poly_num,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> modulus,
                std::shared_ptr<Buffer<uint64_t>> roots_op,
                std::shared_ptr<Buffer<uint64_t>> roots_quo,
                std::shared_ptr<Buffer<uint64_t>> values,
                std::shared_ptr<Buffer<uint64_t>> scalar_op,
                std::shared_ptr<Buffer<uint64_t>> scalar_quo,
                size_t q_idx,
                bool lazy,
                bool wait);

        template void XeHE_invNTT<uint32_t>(int poly_num,
                const XeHE_mem_context<uint32_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint32_t>> modulus,
                std::shared_ptr<Buffer<uint32_t>> roots_op,
                std::shared_ptr<Buffer<uint32_t>> roots_quo,
                std::shared_ptr<Buffer<uint32_t>> values,
                std::shared_ptr<Buffer<uint32_t>> scalar_op,
                std::shared_ptr<Buffer<uint32_t>> scalar_quo,
                size_t q_idx,
                bool lazy,
                bool wait);

        template<class T>
        Buffer<T>::~Buffer(){
            if (owner_ == nullptr)
            {

                auto& xehe_lib = get_xehe_lib();
                auto data = get_data_ptr();
                xehe_lib.pool_free((void*)data, capacity(), q_idx_, (flag_&NONE_CACHED));

            }
            else
            {
                //std::cout << "Do not own" << std::endl;
            }
        }

        template<class T>
        void Buffer<T>::gpu_copy(T* dst, const T* src, size_t len, bool wait)
        {
            cl::sycl::queue& queue = get_queue(q_idx_);
            xehe::gpu_copy(queue, dst, src, len, wait);
        }

        template<class T>
        void Buffer<T>::gpu_host_get(T* dst, const T* src, size_t len, bool wait)
        {
            cl::sycl::queue& queue = get_queue(q_idx_);
            xehe::gpu_host_get(queue, dst, src, len, wait);
        }

        template<class T>
        void Buffer<T>::gpu_set(T* dst, T value, size_t len, bool wait)
        {
            cl::sycl::queue& queue = get_queue(q_idx_);
            xehe::gpu_set(queue, dst, value, len, wait);
        }


        template<class T>
        void Buffer<T>::set_data(const T* ptr, size_t ptr_size, size_t offset, bool wait){
            size_t copy_size = (ptr_size + offset > get_size()) ? get_size() - offset : ptr_size;
            gpu_copy(get_data_ptr() + offset, (T* )ptr, copy_size, wait);
        }

        template<class T>
        void Buffer<T>::set_data(const T* ptr, bool wait) {
            set_data(ptr, get_size(), 0, wait);
        }

// 64 to 32 adapters
        template<class T>
        void Buffer<T>::set_data64to32(const uint64_t* host_data, size_t size, size_t offset, bool wait)
        {
            if (size > 0)
            {
                auto q = get_queue(q_idx_);
                auto gpu_data = get_data_ptr();
                q.submit([&](cl::sycl::handler& h) {
                    h.parallel_for(size, [=](cl::sycl::id<1> i)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        gpu_data[i + offset] = T(host_data[i]);
                    });
                });

                if (wait){
                    q.wait();
                }
            }
        }

        template<class T>
        void Buffer<T>::get_data32to64(uint64_t* host_data, size_t size, size_t offset, bool wait)
        {
            if (size > 0)
            {
                auto q = get_queue(q_idx_);
                auto gpu_data = get_data_ptr();
                q.submit([&](cl::sycl::handler& h) {
                    h.parallel_for(size, [=](cl::sycl::id<1> i)
                        [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        host_data[i] = uint64_t(gpu_data[i + offset]);
                    });
                });

                if (wait){
                    q.wait();
                }
            }
        }

        template<class T>
        T* Buffer<T>::get_host_ptr(void)
        {
            host_reflect_.clear();
            host_reflect_.resize(get_size());
            get_data(host_reflect_.data());
            return(host_reflect_.data());
        }


        // get data to host memory
        template<class T>
        void Buffer<T>::get_data(T* ptr, size_t ptr_size, size_t offset, bool wait) {

            size_t copy_size = (ptr_size + offset > get_size()) ? get_size() - offset : ptr_size;
#if 1
            gpu_host_get(ptr, get_data_ptr() + offset, copy_size, wait);
#else
            gpu_copy(ptr, get_data_ptr() + offset, copy_size, wait);
#endif
        }

        template<class T>
        void Buffer<T>::get_data(T* ptr, bool wait) {
            get_data(ptr, get_size(), 0, wait);
        }


        // TODO: GPU kernel
        template<class T>
        void Buffer<T>::set_value(const T value, size_t ptr_size, size_t offset){
            size_t copy_size = (ptr_size + offset > get_size()) ? get_size() - offset : ptr_size;
            gpu_set(get_data_ptr() + offset, value, copy_size);
        }

        template<class T>
        T Buffer<T>::get_value(size_t offset)
        {
            T ret;
            get_data(&ret, 1, offset);
            return(ret);
        }

        // actual copy
        template<class T>
        void Buffer<T>::deep_copy(const std::shared_ptr<Buffer<T>> b, size_t len, size_t offset0, size_t offset1, bool wait)
        {
            auto min_sz0 = std::min(get_size(), (len + offset0));
            auto min_sz1 = std::min(b->get_size(), (len + offset1));
            auto min_sz = std::min(min_sz0, min_sz1);
            gpu_copy(get_data_ptr() + offset0, b->get_data_ptr() + offset1, min_sz, wait); // don't wait
        }

        template<class T>
        void Buffer<T>::deep_copy(const std::shared_ptr<Buffer<T>> b, bool wait)
        {
            return(deep_copy(b, get_size(), 0, 0, wait));
        }

        template<class T>
        bool Buffer<T>::is_alias(const std::shared_ptr<Buffer<T>> b)
        {
            bool aliased = (get_data_ptr() != nullptr && b->get_data_ptr() != nullptr && get_data_ptr() == b->get_data_ptr());
            return(aliased);
        }


        template<class T>
        Buffer<T>::Buffer(const Buffer<T> &b){
            *this = b;
        }


        // keep the same pointer
        template<class T>
        Buffer<T>& Buffer<T>::operator = (const Buffer<T> &b){
            flag_ = b.flag_;
            offset_ = b.offset_;
            size() = b.size();
            capacity() = b.capacity();
            // most of the time this is nullptr
            // for sub-buffers it's not
            owner_ = b.owner_;
            data_ = (T * )b.get_data_ptr();
            return *this;
        }

        template<class T>
        Buffer<T>::Buffer(){
            flag_ = 0;
            offset_ = 0;
            size() = 0;
            capacity() = 0;
            data_ = nullptr;
        }

        template<class T>
        Buffer<T>::Buffer(size_t buffer_size, size_t q_idx, size_t flag){
            this->malloc(buffer_size, q_idx, flag);
        }

        template<class T>
        void Buffer<T>::malloc(size_t buffer_size, size_t q_idx, size_t flag){
            offset_ = 0;
            flag_ = flag;
            q_idx_ = q_idx;
            size() = buffer_size;
            size_t new_capacity = buffer_size;
            capacity() = new_capacity;

            auto& xehe_lib = get_xehe_lib();
            // can update capacity
            // TODO :: has to be atomic from here!!!!

            data_ = xehe_lib.pool_alloc<T>(buffer_size, new_capacity, q_idx, (flag_&NONE_CACHED));
            capacity() = new_capacity;
        }


        template<class T>
        T* Buffer<T>::get_data_ptr(){
            return data_;
        }

        template<class T>
        const T* Buffer<T>::get_data_ptr() const{
            return data_;
        }

        template<class T>
        size_t Buffer<T>::get_size() const{
            return buffer_size_;
        }
        
        template<class T>
        size_t Buffer<T>::get_q_idx() const{
            return q_idx_;
        }

        template<class T>
        const size_t& Buffer<T>::capacity(void) const
        {
            return(buffer_capacity_);
        }
        template<class T>
        size_t& Buffer<T>::capacity(void)
        {
            return(buffer_capacity_);
        }

        template<class T>
        const size_t& Buffer<T>::size(void) const
        {
            return(buffer_size_);
        }
        template<class T>
        size_t& Buffer<T>::size(void)
        {
            return(buffer_size_);
        }

        template<class T>
        void Buffer<T>::move_owner_ptr(std::shared_ptr<Buffer<T>>& owner, size_t sz, size_t offset)
        {
            offset_ = offset;
            data_ += offset_;
            buffer_size_ = sz;
            // make sure memory won't be removed underneath
            owner_ = owner;
        }


        template class Buffer<uint64_t>;
        template class Buffer<uint32_t>;
        template class Buffer<int64_t>;
        template class Buffer<int32_t>;



        template<typename T>
        std::shared_ptr<Buffer<T>> XeHE_malloc(size_t sz, size_t q_idx, size_t flag){
            std::shared_ptr<Buffer<T>> buf(new Buffer<T>(sz, q_idx, flag));
            return buf;
        }

        void get_mem_cache_stat(size_t &alloced, size_t &freed, size_t q_idx)
        {
            auto memcache = get_xehe_lib().get_memcache(q_idx);
            memcache->get_memory_cache_stat(alloced, freed);
        }

        void activate_memory_cache(size_t q_idx)
        {
            auto memcache = get_xehe_lib().get_memcache(q_idx);
            memcache->activate_cache();
        }

        void free_memory_cache(size_t q_idx)
        {
            auto memcache = get_xehe_lib().get_memcache(q_idx);
            memcache->dealloc();
        }

        void free_free_cache(size_t q_idx)
        {
            auto memcache = get_xehe_lib().get_memcache(q_idx);
            memcache->pool_free_free();           
        }


        // wait for all activities to complete on the queue
        void wait_for_queue(int queue)
        {
            auto q = get_queue(queue);
            q.wait();

        }

        template std::shared_ptr<Buffer<uint64_t>> XeHE_malloc<uint64_t>(size_t sz, size_t q_idx, size_t flag);
        template std::shared_ptr<Buffer<uint32_t>> XeHE_malloc<uint32_t>(size_t sz, size_t q_idx, size_t flag);
        template std::shared_ptr<Buffer<int64_t>> XeHE_malloc<int64_t>(size_t sz, size_t q_idx, size_t flag);
        template std::shared_ptr<Buffer<int32_t>> XeHE_malloc<int32_t>(size_t sz, size_t q_idx, size_t flag);


    } //ext
} // xehe