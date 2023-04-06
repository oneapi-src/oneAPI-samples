/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XeHE_HPP
#define XeHE_HPP

#ifdef __JETBRAINS_IDE__
// Stuff that only clion will see goes here
#define BUILD_WITH_IGPU
#endif

#include <memory>
#include <vector>
#include <string>


namespace xehe {
    namespace ext {

        //cl::sycl::queue get_queue(size_t index = 0);
//
//        template<typename T>
//        struct MulModOperand
//        {
//            T operand;
//            T quotient;
//        };

        // forward declaration
        template<class T>
        struct XeHE_mem_context;

        template<>
        struct XeHE_mem_context<uint64_t>;

        template<class T>
        struct XeHE_mem_keys;
        template<>
        struct XeHE_mem_keys<uint64_t>;

        template<class T>
        class Buffer{
        public:

            
            /********************************************************************
            * 
            * Start of XeHE buffer external interface
            * 
            *********************************************************************/

            // upload host data to full GPU  buffer
            void set_data(const T* host_data, bool wait = true);
            // upload size number of T words from host_data buffer to GPU  startbuffering from offset
            void set_data(const T* host_data, size_t size, size_t offset = 0, bool wait = true);

            // adpter from 64 to 32 bit host to GPU transfer
            template<class S>
            void set_data_adapter(const S* host_data, size_t size, size_t offset = 0, bool wait = true)
            {
                if constexpr (std::is_same<T, S>::value)
                {
                    set_data(host_data, size, offset, wait);
                }
                else
                {
                    set_data64to32(host_data, size, offset, wait);
                }
            }

            template<class S>
            void set_data_adapter(const S* host_data, size_t size, bool wait = true)
            {
                set_data_adapter<S>(host_data, size, 0, wait);
            }

            template<class S>
            void set_data_adapter(const S* host_data, bool wait = true)
            {
                set_data_adapter<S>(host_data, get_size(), 0, wait);
            }


            // downoad  full GPU bufer into  host_data memorymemory
            void get_data(T* host_data, bool wait = true);
            // download size T words fro GPU buffer into host_data starting from offset
            void get_data(T* host_data, size_t size, size_t offset = 0, bool wait = true);


            template<class S>
            void get_data_adapter(S* host_data, size_t size, size_t offset = 0, bool wait = true)
            {
                if constexpr (std::is_same<T, S>::value)
                {
                    get_data(host_data, size, offset);
                }
                else
                {
                    get_data32to64(host_data, size, offset);
                }
            }

            template<class S>
            void get_data_adapter(S* host_data, size_t size, bool wait = true)
            {
                get_data_adapter<S>(host_data, size, 0, wait);
            }

            template<class S>
            void get_data_adapter(S* host_data, bool wait = true)
            {
                get_data_adapter<S>(host_data, get_size(), 0, wait);
            }


            // set the same value size times in GPU buffer starting from offset 
            void set_value(const T value, size_t size = 1, size_t offset = 0);
            // returna single value from GPU buffer at offset 
            T get_value(size_t offset = 0);

            // copy full buffer b into this buffer
            void deep_copy(const std::shared_ptr<Buffer<T>> b, bool wait = false);
            // src buffer, len, dst buffer offset, src buffer offset
            // copy part of the buffer b of size from offset1 into this buffer strting from offset0
            void deep_copy(const std::shared_ptr<Buffer<T>> b, size_t size, size_t offset0 = 0, size_t offset1 = 0, bool wait = false);

            // check whether b GPU buffer is an alias of this buffer
            // serves to indicate whther the previous content is going destructed by writing over.
            bool is_alias(const std::shared_ptr<Buffer<T>> b);

            // get GPU buffer allocated size
            size_t get_size() const;

            // get queue index the Buffer is on
            size_t get_q_idx() const;

            bool is_allocated(void) const
            {
                return(get_data_ptr() != nullptr);
            }

            Buffer(const Buffer &b);

            Buffer & operator = (const Buffer &b);



            /********************************************************************
            *
            * End of XeHE buffer external interface
            *
            *********************************************************************/

            ~Buffer();


            template<class U>
            friend std::shared_ptr<Buffer<U>> XeHE_malloc(size_t sz, size_t q_idx, size_t flag);

            //template<typename U>
            //friend std::shared_ptr<Buffer<U>> XeHE_subbuffer(std::shared_ptr<Buffer<U>>& ref, size_t sz, size_t offset, size_t flag);


            friend void XeHE_negate(
                    int n_polys,
                    const std::shared_ptr<Buffer<uint64_t>> values,
                    const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                    std::shared_ptr<Buffer<uint64_t>> result,
                    bool wait);



            friend void XeHE_add(
                int n_polys,
                int n_poly1,
                int n_poly2,
                const std::shared_ptr<Buffer<uint64_t>> oprnd1,
                const std::shared_ptr<Buffer<uint64_t>> oprnd2,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> result,
                bool wait
            );


            friend void XeHE_sub(
                int n_polys,
                int n_poly1,
                int n_poly2,
                const std::shared_ptr<Buffer<uint64_t>> oprnd1,
                const std::shared_ptr<Buffer<uint64_t>> oprnd2,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> result,
                bool wait
            );


            friend void XeHE_multiply_plain_ckks(
                int n_polys,
                const std::shared_ptr<Buffer<uint64_t>> oprnd1,
                const std::shared_ptr<Buffer<uint64_t>> oprnd2,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>>  result,
                bool wait);

            friend void XeHE_multiply_plain_ckks_add(
                int n_polys,
                int n_poly_add,
                int n_poly_mul,
                const std::shared_ptr<Buffer<uint64_t>> oprnd_add,
                const std::shared_ptr<Buffer<uint64_t>> oprnd_mul,
                const std::shared_ptr<Buffer<uint64_t>> oprnd_plain,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>>  result,
                bool wait);

            friend void XeHE_multiply_ckks_add(
                int n_polys,
                int n_poly_add,
                int n_poly2,
                int n_poly3,
                const std::shared_ptr<Buffer<uint64_t>> oprnd_add,
                const std::shared_ptr<Buffer<uint64_t>> oprnd2,
                const std::shared_ptr<Buffer<uint64_t>> oprnd3,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>>  result,
                bool wait);

            friend void XeHE_mul_ckks(
                size_t max_sz,            // n*m - 1
                size_t enc1_sz, // n polys
                size_t enc2_sz,           // m polys
                const std::shared_ptr<Buffer<uint64_t>>  poly1,  // enc1
                const std::shared_ptr<Buffer<uint64_t>>  poly2, //  enc2
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> poly_res,               // destination, result
                bool wait);

            friend void XeHE_divide_round_q_last_ckks(
                size_t encrypted_size,       // n_polys in encr message
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> input, // not a constant, reused inside kernel
                std::shared_ptr<Buffer<uint64_t>> output,
                bool wait
            );

            friend void XeHE_mod_switch_ckks(
                int dest_size,        // number of output compunents
                int dest_q_base_sz,   // destination rns base size
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx, // context
                const std::shared_ptr<Buffer<uint64_t>> input,
                std::shared_ptr<Buffer<uint64_t>> output,
                bool wait);


            friend void XeHE_relinearize(
                size_t kswitch_keys_index,      // index in the array of keys
                const XeHE_mem_keys<uint64_t>& xehe_key_mem,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>> input,
                std::shared_ptr<Buffer<uint64_t>> output,
                bool wait
            );
 

            friend void XeHE_square(
                int dest_size,                                   // number of output components
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,            // context
                std::shared_ptr<Buffer<uint64_t>> inp_output,     // input/output inplace
                bool wait
            );

            friend void XeHE_permute(
                uint32_t encryp_sz,
                uint32_t q_base_sz,
                uint32_t coeff_count,
                uint32_t galois_elt,
                uint32_t index,
                const std::shared_ptr<Buffer<uint64_t>> input,
                std::shared_ptr<Buffer<uint64_t>> output,
                bool wait
            );

            template<typename U>
            friend void XeHE_NTT(
                    int poly_num,
                    const XeHE_mem_context<U>& xehe_mem_ctx,
                    std::shared_ptr<Buffer<U>> modulus,
                    std::shared_ptr<Buffer<U>> roots_op,
                    std::shared_ptr<Buffer<U>> roots_quo,
                    std::shared_ptr<Buffer<U>> values,
                    std::shared_ptr<Buffer<U>> scalar_op,
                    std::shared_ptr<Buffer<U>> scalar_quo,
                    size_t q_idx,
                    bool lazy,
                    bool wait);

            template<typename U>
            friend void XeHE_invNTT(
                    int poly_num,
                    const XeHE_mem_context<U>& xehe_mem_ctx,
                    std::shared_ptr<Buffer<U>> modulus,
                    std::shared_ptr<Buffer<U>> roots_op,
                    std::shared_ptr<Buffer<U>> roots_quo,
                    std::shared_ptr<Buffer<U>> values,
                    std::shared_ptr<Buffer<U>> scalar_op,
                    std::shared_ptr<Buffer<U>> scalar_quo,
                    size_t q_idx,
                    bool lazy,
                    bool wait);
            typedef enum {
                NONE_CACHED = 1,
                REFERED = 0x8000
            } FLAGS;

        protected:


            Buffer();

            Buffer(size_t buffer_size, size_t q_idx = 0, size_t flag = 0);


            void malloc(size_t buffer_size, size_t q_idx = 0, size_t flag = 0);

            void gpu_copy(T* dst, const T* src, size_t len, bool wait = true);
            void gpu_set(T* dst, T value, size_t len, bool wait = true);
            void gpu_host_get(T* dst, const T* src, size_t len, bool wait = true);

            void set_data64to32(const uint64_t* host_data, size_t size, size_t offset, bool wait);

            void get_data32to64(uint64_t* host_data, size_t size, size_t offset, bool wait);

            T* get_host_ptr(void);

            T* get_data_ptr(void);

            const T* get_data_ptr(void) const;

            void move_owner_ptr(std::shared_ptr<Buffer<T>>& owner, size_t sz, size_t offset);

            const size_t& capacity(void) const;
            size_t& capacity(void);
            const size_t& size(void) const;
            size_t& size(void);


        private:

            size_t flag_;
            size_t q_idx_;
            size_t offset_;
            size_t buffer_size_ = 0;
            size_t buffer_capacity_ = 0;
            // to make sure memory won't be removed underneath
            // we keep pointer to owner of the memory
            std::shared_ptr<Buffer<T>> owner_;
            // GPU buffer 
            T* data_ = nullptr;
            std::vector<T> host_reflect_;
        };


        /********************************************************************
        *
        * Start of XeHE functional interface
        *
        *********************************************************************/



        // this is an entry of the array of structures
        // it's structure reflects encryption level parmeters and
        // precomputed data sets relevant to the level
        template<class T>
        struct XeHE_mem_context {
            size_t rns_base_size;                                           // RNS base size
            size_t coeff_count;                                             // coefficent count
            size_t mod_inv_size;                                            // size in T units of 2^BitCount(T)/modulus values
            size_t next_rns_base_size;                                      // nextxRNS base size
            std::shared_ptr<Buffer<T>> xe_modulus;                          // RNS modules
            std::shared_ptr<Buffer<T>> xe_inv1;                             // 2^BitCount(T)/prime values; Bitcount = BitCount(T)
            std::shared_ptr<Buffer<T>> xe_inv2;                             // 2^(2*BitCount(T))/prime values; BitCount = 2*BitCount(T)
            std::shared_ptr<Buffer<T>> xe_inv_modulus;                      // 2^128/prime values
            std::shared_ptr<Buffer<T>> xe_prim_roots_op;                    // prim roots of unity
            std::shared_ptr<Buffer<T>> xe_prim_roots_quo;                   // powers of prim roots of unity * 2^64 / mod each RNS prime
//            std::shared_ptr<Buffer<MulModOperand<T>>> xe_prim_roots;        // powers of prim roots of unity * 2^64 / mod each RNS prime
            std::shared_ptr<Buffer<T>> xe_inv_prim_roots_op;                // powers of invert prime roots of unity 
            std::shared_ptr<Buffer<T>> xe_inv_prim_roots_quo;               // powers of invert prime roots of unity *2^64 / mod each RNS prime
//            std::shared_ptr<Buffer<MulModOperand<T>>> xe_inv_prim_roots;    // powers of invert prime roots of unity *2^64 / mod each RNS prime
            std::shared_ptr<Buffer<T>> xe_inv_degree_op;                    // n^(-1) modulo m;  iNTT scaler
            std::shared_ptr<Buffer<T>> xe_inv_degree_quo;                   // n^(-1) modulo m;  iNTT scaler
//            std::shared_ptr<Buffer<MulModOperand<T>>> xe_inv_degree;        // n^(-1) modulo m;  iNTT scaler
            std::shared_ptr<Buffer<T>> xe_inv_q_last_mod_q_op;              // 1/qi mod last qi for each qi in [0, q_base_sz-1]
            std::shared_ptr<Buffer<T>> xe_inv_q_last_mod_q_quo;             // 1/qi mod last qi for each qi in [0, q_base_sz-1]
//            std::shared_ptr<Buffer<MulModOperand<T>>> xe_inv_q_last_mod_q;  // 1/qi mod last qi for each qi in [0, q_base_sz-1]
        };

        template<>
        struct XeHE_mem_context<uint64_t> {
            size_t rns_base_size;                                           // RNS base size
            size_t coeff_count;                                             // coefficent count
            size_t mod_inv_size;                                            // size in T units of 2^BitCount(T)/modulus values
            size_t next_rns_base_size;                                      // nextxRNS base size
            std::shared_ptr<Buffer<uint64_t>> xe_modulus;                          // RNS modules
            std::shared_ptr<Buffer<uint64_t>> xe_inv1;                             // 2^BitCount(T)/prime values; Bitcount = BitCount(T)
            std::shared_ptr<Buffer<uint64_t>> xe_inv2;                             // 2^(2*BitCount(T))/prime values; BitCount = 2*BitCount(T)
            std::shared_ptr<Buffer<uint64_t>> xe_inv_modulus;                      // 2^128/prime values
            std::shared_ptr<Buffer<uint64_t>> xe_prim_roots_op;                    // prim roots of unity
            std::shared_ptr<Buffer<uint64_t>> xe_prim_roots_quo;                   // powers of prim roots of unity * 2^64 / mod each RNS prime
//            std::shared_ptr<Buffer<MulModOperand<uint64_t>>> xe_prim_roots;        // powers of prim roots of unity * 2^64 / mod each RNS prime
            std::shared_ptr<Buffer<uint64_t>> xe_inv_prim_roots_op;                // powers of invert prime roots of unity 
            std::shared_ptr<Buffer<uint64_t>> xe_inv_prim_roots_quo;               // powers of invert prime roots of unity *2^64 / mod each RNS prime
//            std::shared_ptr<Buffer<MulModOperand<uint64_t>>> xe_inv_prim_roots;    // powers of invert prime roots of unity *2^64 / mod each RNS prime
            std::shared_ptr<Buffer<uint64_t>> xe_inv_degree_op;                    // n^(-1) modulo m;  iNTT scaler
            std::shared_ptr<Buffer<uint64_t>> xe_inv_degree_quo;                   // n^(-1) modulo m;  iNTT scaler
//            std::shared_ptr<Buffer<MulModOperand<uint64_t>>> xe_inv_degree;        // n^(-1) modulo m;  iNTT scaler
            std::shared_ptr<Buffer<uint64_t>> xe_inv_q_last_mod_q_op;              // 1/qi mod last qi for each qi in [0, q_base_sz-1]
            std::shared_ptr<Buffer<uint64_t>> xe_inv_q_last_mod_q_quo;             // 1/qi mod last qi for each qi in [0, q_base_sz-1]
//            std::shared_ptr<Buffer<MulModOperand<uint64_t>>> xe_inv_q_last_mod_q;  // 1/qi mod last qi for each qi in [0, q_base_sz-1]
        };


        // key reated context info
        template<class T>
        struct XeHE_mem_keys {
            size_t num_keys;                           // num of keys
            size_t rns_base_size;                      // current level RNS base size
            size_t n_polys;                            // mod switch target, number of polinomials 
            size_t key_rns_base_size;                  // originak key RNS base size 
            size_t coeff_count;                        // polymonial order
            std::shared_ptr<Buffer<T>> xe_keys; // keys GPU buffer
        };

        template<>
        struct XeHE_mem_keys<uint64_t> {
            size_t num_keys;                           // num of keys
            size_t rns_base_size;                      // current level RNS base size
            size_t n_polys;                            // mod switch target, number of polinomials 
            size_t key_rns_base_size;                  // originak key RNS base size 
            size_t coeff_count;                        // polymonial order
            std::shared_ptr<Buffer<uint64_t>> xe_keys; // keys GPU buffer
        };


        /**
        
          Allocates a GPU buffer of size, flag is not used right now

          @param[in] size
          @return shared ptr to a Buffer class
        */
        template<typename T>
        extern std::shared_ptr<Buffer<T>> XeHE_malloc(size_t size, size_t q_idx = 0, size_t flag = 0);

        extern void get_mem_cache_stat(size_t &alloced, size_t &freed, size_t q_idx = 0);
        extern void activate_memory_cache(size_t q_idx = 0);
        extern void free_memory_cache(size_t q_idx = 0);
        extern void free_free_cache(size_t q_idx = 0);


         /**
            Negates a ciphertext .

            @param[in] encrypted The ciphertext to negate
            @param[in] xehe_mem_ctx The memory context
            @param[out] destination The output ciphertext
          */

        void XeHE_negate(
            int n_polys,
            const std::shared_ptr<Buffer<uint64_t>> encrypted,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> destination,
            bool wait = true);

        /**
          Sums up 2 ciphertext messages to produce ciphertext of the sum

          @param[in] n_polys The number of polinimials in the resulting message
          @param[in] n_poly1 The number of polinimials in the input message 1
          @param[in] n_poly2 The number of polinimials in the input message 2
          @param[in] encrypted1 The input1
          @param[in] encrypted2 The input2
          @param[in] xehe_mem_ctx The memory context
          @param[out] destination The output ciphertext
        */

        void XeHE_add(
            int n_polys,
            int n_poly1,
            int n_poly2,
            const std::shared_ptr<Buffer<uint64_t>> encrypted1,
            const std::shared_ptr<Buffer<uint64_t>> encrypted2,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> destination,
            bool wait = true
        );

        /**
        Substruct 1 ciphertext message from the other to produce a ciphertext of the difference.

          @param[in] n_polys The number of polinimials in the resulting message
          @param[in] n_poly1 The number of polinimials in the input message 1
          @param[in] n_poly2 The number of polinimials in the input message 2
          @param[in] encrypted1 The input1
          @param[in] encrypted2 The input2
          @param[in] xehe_mem_ctx The xehe memory context
          @param[out] destination The output ciphertext
        */



        void XeHE_sub(
            int n_polys,
            int n_poly1,
            int n_poly2,
            const std::shared_ptr<Buffer<uint64_t>> encrypted1,
            const std::shared_ptr<Buffer<uint64_t>> encrypted2,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> destination,
            bool wait = true
        );


        /**
        Multiply a ciphertext message with a plaintext message to produce a ciphertext of the sum.

        @param[in] n_polys The number of polinimials in the encrypted message
        @param[in] encrypted The encrypted input
        @param[in] plain The plain input
        @param[in] xehe_mem_ctx The xehe memory context
        @param[out] destination The output ciphertext

        */

        void XeHE_multiply_plain_ckks(
            int n_polys,
            const std::shared_ptr<Buffer<uint64_t>> encrypted,
            const std::shared_ptr<Buffer<uint64_t>> plain,
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>>  destination,
            bool wait = true);



        /**
        Multiply up 2 ciphertext messages to produce ciphertext of the product.

        @param[in] n_polys The number of polinimials in the resulting message
        @param[in] n_poly1 The number of polinimials in the input message 1
        @param[in] n_poly2 The number of polinimials in the input message 2
        @param[in] encrypted1 The input1
        @param[in] encrypted2 The input2
        @param[in] xehe_mem_ctx The xehe memory context
        @param[out] destination The output ciphertext
        */


        void XeHE_mul_ckks(
            size_t n_poly,     // n*m - 1
            size_t n_poly1,           // n  
            size_t n_poly2,           // m 
            const std::shared_ptr<Buffer<uint64_t>>  encrypted1,  
            const std::shared_ptr<Buffer<uint64_t>>  encrypted2, 
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> poly_res,
            bool wait = true);

        /**
        Multiply a ciphertext message with a plaintext message, and accumulate to a ciphertext to produce a ciphertext of the updated sum with the product.
        result = oprnd_add + oprnd_mul * oprnd_plain
        @param[in] n_polys The number of polynomials in the destination
        @param[in] n_poly_add The number of polynomials in the encrypted message
        @param[in] n_poly_mul The number of polynomials in the encrypted message
        @param[in] oprnd_add The encrypted input for add
        @param[in] oprnd_mul The encrypted input for multiply
        @param[in] oprnd_plain The plain input
        @param[in] xehe_mem_ctx The xehe memory context
        @param[out] destination The output ciphertext

        */
        void XeHE_multiply_plain_ckks_add(
                int n_polys,
                int n_poly_add,
                int n_poly_mul,
                const std::shared_ptr<Buffer<uint64_t>> oprnd_add,
                const std::shared_ptr<Buffer<uint64_t>> oprnd_mul,
                const std::shared_ptr<Buffer<uint64_t>> oprnd_plain,
                const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
                std::shared_ptr<Buffer<uint64_t>>  result,
                bool wait = true);

        /**
        Multiply up 2 ciphertext messages to produce ciphertext of the product, and accumulate to a ciphertext to produce a ciphertext of the updated sum with the product.
        result = oprnd_add + oprnd2 * oprnd3
        @param[in] n_polys The number of polynomials in the destination
        @param[in] n_poly_add The number of polynomials in the oprnd_add
        @param[in] n_poly2 The number of polynomials in the oprnd2
        @param[in] n_poly3 The number of polynomials in the oprnd3
        @param[in] oprnd_add  The input for add
        @param[in] oprnd2 The input1 for multiply
        @param[in] oprnd3 The input2 for multiply
        @param[in] xehe_mem_ctx The xehe memory context
        @param[out] destination The output ciphertext
        */
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
            bool wait = true);

        /**
        Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1} and
        stores the result in the destination parameter.

        @param[in] n_poly                 The number of destination polynomials
        @param[in] xehe_mem_ctx           The xehe memory context
        @param[in/out] input              The input ciphertext. mutable, size = 2
        @param[in/out] destination        The cybertext the modulus switched result. mutable.
        */

        void XeHE_divide_round_q_last_ckks(
            size_t n_poly,       // n_polys in encr message
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> input, 
            std::shared_ptr<Buffer<uint64_t>> destination,
            bool wait = true
        );


        /**
        Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1} and
        scales the destination message down accordingly.

        @param[in] n_poly                 The number of destination polynomials
        @param[in] dest_rns_base_size,    The destination message rns base size
        @param[in] xehe_mem_ctx           The xehe memory context
        @param[in] encrypted              The input ciphertext
        @param[out] destination           The ciphertext with the modulus switched result.
        */

        void XeHE_mod_switch_ckks(
            int n_poly,        
            int dest_rns_base_size,   
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            const std::shared_ptr<Buffer<uint64_t>> encrypted,
            std::shared_ptr<Buffer<uint64_t>> destination,
            bool wait = true);


        /**
        Relinearize an input ciphertext messages with relianirization keys to produce a new ciphertext
        with a low number of encrypted polinomials.
        The processing is done in-place, that destroys the content of the input GPU buffers.

        @param[in] kswitch_keys_index     Not used. 
        @param[in] xehe_key_mem           The xehe keys memory context
        @param[in] xehe_mem_ctx           The xehe memory context
        @param[in/out] input              The key swith target; mutable.
        @param[in/out] destination        The destination message mutable.

        */

        void XeHE_relinearize(
            size_t kswitch_keys_index,      
            const xehe::ext::XeHE_mem_keys<uint64_t> & xehe_key_mem, 
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> input,
            std::shared_ptr<Buffer<uint64_t>> destination,
            bool wait = true
        );


        /**
       Squares a ciphertext in-place. This functions computes the square of encrypted and stores the result in the destination
       parameter.

       @param[in] n_poly                 The number of cipertext polynomials
       @param[in] xehe_mem_ctx           The xehe memory context
       @param[in/out] encrypted          The ciphertext to square
       */


        void XeHE_square(
            int n_poly,                                   
            const XeHE_mem_context<uint64_t>& xehe_mem_ctx,
            std::shared_ptr<Buffer<uint64_t>> encrypted,
            bool wait = true
        );


        void XeHE_permute(
                          uint32_t encryp_sz,
                          uint32_t q_base_sz,
                          uint32_t coeff_count,
                          uint32_t galois_elt,
                          uint32_t index,
                          const std::shared_ptr<Buffer<uint64_t>> input,
                          std::shared_ptr<Buffer<uint64_t>> output,
                          bool wait = true
        );

        template<typename T>
        extern void XeHE_NTT(
                int poly_num,
                const XeHE_mem_context<T>& xehe_mem_ctx,
                std::shared_ptr<Buffer<T>> modulus,
                std::shared_ptr<Buffer<T>> roots_op,
                std::shared_ptr<Buffer<T>> roots_quo,
                std::shared_ptr<Buffer<T>> values,
                std::shared_ptr<Buffer<T>> scalar_op = nullptr,
                std::shared_ptr<Buffer<T>> scalar_quo = nullptr,
                size_t q_idx = 0,
                bool lazy = false,
                bool wait = true);

        template<typename T>
        extern void XeHE_invNTT(
                int poly_num,
                const XeHE_mem_context<T>& xehe_mem_ctx,
                std::shared_ptr<Buffer<T>> modulus,
                std::shared_ptr<Buffer<T>> roots_op,
                std::shared_ptr<Buffer<T>> roots_quo,
                std::shared_ptr<Buffer<T>> values,
                std::shared_ptr<Buffer<T>> scalar_op = nullptr,
                std::shared_ptr<Buffer<T>> scalar_quo = nullptr,
                size_t q_idx = 0,
                bool lazy = false,
                bool wait = true);

#if _NO_WAIT_
        extern void end_computation();
#endif

        // summurizing the profilling data in EventCollector
        void process_events();
        // clearing the EventCollector data
        void clear_events();
        // process all the events in EventCollector into an operation export record, append it to export table
        void add_operation(std::string op_name, double avg_external_time, int loops);
        // export the export table to a txt file
        void export_table(std::string file_name="event_stats.txt");
        // add header for the export table, describing the parameter choice
        void add_header(int poly_order, int data_bound, int delta_bits, int rns_size);
        // clear the export table in record.
        void clear_export_table();

        // wait for all activities to complete on the queue
        extern void wait_for_queue(int queue = 0);
        /********************************************************************
        *
        * End of XeHE functional interface
        *
        *********************************************************************/



    } // ext
} // xehe

#endif //XeHE_HPP
