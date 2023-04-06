#include "include/driver.h"




template<typename T>
void params_to_gpu(seal::SEALContext& ctx, xehe::ext::XeHE_mem_context<T>& gpu_context_entry, std::vector<T>& cpu_modulus, size_t q_idx = 0) {
    // std::cout << "GPU context params are sent to queue #" << q_idx << std::endl;
    auto context_data_ptr = ctx.key_context_data();
    auto& parms0 = context_data_ptr->parms();
    auto& coeff_modulus0 = parms0.coeff_modulus();
    auto const_ratio_sz = coeff_modulus0[0].const_ratio().size();

    auto& parms = context_data_ptr->parms();
    auto& coeff_modulus = parms.coeff_modulus();
    auto coeff_count = parms.poly_modulus_degree();
    auto q_base_sz = coeff_modulus.size();
    auto NTTTables = context_data_ptr->small_ntt_tables();
    //auto rns_tool = context_data_ptr->rns_tool();
    //auto inv_q_last_mod_q = rns_tool->inv_q_last_mod_q();
    auto next_context_data_ptr = context_data_ptr->next_context_data();
    size_t next_coeff_modulus_size = 0;
    if (next_context_data_ptr)
    {
        auto& next_parms = (*next_context_data_ptr).parms();
        next_coeff_modulus_size = next_parms.coeff_modulus().size();
    }

    // gpu_context_entry.xe_inv1 = xehe::ext::XeHE_malloc<T>(q_base_sz);
    // gpu_context_entry.xe_inv2 = xehe::ext::XeHE_malloc<T>(2 * q_base_sz);
    // gpu_context_entry.xe_inv_q_last_mod_q_op = xehe::ext::XeHE_malloc<T>(q_base_sz - 1);
    // gpu_context_entry.xe_inv_q_last_mod_q_quo = xehe::ext::XeHE_malloc<T>(q_base_sz - 1);

    gpu_context_entry.xe_modulus = xehe::ext::XeHE_malloc<T>(q_base_sz, q_idx);

    gpu_context_entry.xe_prim_roots_op = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count, q_idx);
    gpu_context_entry.xe_prim_roots_quo = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count, q_idx);

    gpu_context_entry.xe_inv_prim_roots_op = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count, q_idx);
    gpu_context_entry.xe_inv_prim_roots_quo = xehe::ext::XeHE_malloc<T>(q_base_sz * coeff_count, q_idx);

    gpu_context_entry.xe_inv_degree_op = xehe::ext::XeHE_malloc<T>(q_base_sz, q_idx);
    gpu_context_entry.xe_inv_degree_quo = xehe::ext::XeHE_malloc<T>(q_base_sz, q_idx);

    std::vector<T> cpu_inv1(q_base_sz);
    std::vector<T> cpu_inv2(2 * q_base_sz);
    std::vector<T> cpu_inv_q_last_mod_q_op(q_base_sz - 1);
    std::vector<T> cpu_inv_q_last_mod_q_quo(q_base_sz - 1);

    cpu_modulus = std::vector<T>(q_base_sz);

    std::vector<T> cpu_roots_op(q_base_sz * coeff_count);
    std::vector<T> cpu_roots_quo(q_base_sz * coeff_count);

    std::vector<T> cpu_inv_roots_op(q_base_sz * coeff_count);
    std::vector<T> cpu_inv_roots_quo(q_base_sz * coeff_count);

    std::vector<T> cpu_inv_degree_modulus_op(q_base_sz);
    std::vector<T> cpu_inv_degree_modulus_quo(q_base_sz);

    for (size_t j = 0; j < q_base_sz; j++)
    {
        cpu_modulus[j] = coeff_modulus[j].value();
        //auto mod = cpu_modulus[j];
        // // inv1
        // T numer2[2]{ 0, 1 };
        // T quo2[2];
        // seal::util::divide_uint128_uint64_inplace_generic(numer2, mod, quo2);
        // cpu_inv1[j] = quo2[0];

        // // inv2
        // T numer3[3]{ 0, 0, 1 };
        // T quo3[3];
        // seal::util::divide_uint192_inplace(numer3, mod, quo3);
        // cpu_inv2[2 * j] = quo3[0];
        // cpu_inv2[2 * j + 1] = quo3[1];

        auto ntt = NTTTables[j];
        if constexpr (sizeof(T) == 8)
        {
            for (size_t i = 0; i < coeff_count; i++)
            {

                cpu_roots_op[j * coeff_count + i] = ntt.get_from_root_powers()[i].operand;
                cpu_roots_quo[j * coeff_count + i] = ntt.get_from_root_powers()[i].quotient;

                cpu_inv_roots_op[j * coeff_count + i] = ntt.get_from_inv_root_powers()[i].operand;
                cpu_inv_roots_quo[j * coeff_count + i] = ntt.get_from_inv_root_powers()[i].quotient;
            }
            cpu_inv_degree_modulus_op[j] = ntt.inv_degree_modulo().operand;
            cpu_inv_degree_modulus_quo[j] = ntt.inv_degree_modulo().quotient;
        }
        else  if constexpr (sizeof(T) == 4)
        {
            for (size_t i = 0; i < coeff_count; i++)
            {

                auto root_op = T(ntt.get_from_root_powers()[i].operand);
                cpu_roots_op[j * coeff_count + i] = root_op;
                cpu_roots_quo[j * coeff_count + i] = calc_quotient<T>(coeff_modulus[j], root_op);

                auto inv_root_op = T(ntt.get_from_inv_root_powers()[i].operand);
                cpu_inv_roots_op[j * coeff_count + i] = inv_root_op;
                cpu_inv_roots_quo[j * coeff_count + i] = calc_quotient<T>(coeff_modulus[j], inv_root_op);
            }
            auto mod_op = T(ntt.inv_degree_modulo().operand);
            cpu_inv_degree_modulus_op[j] = mod_op;
            cpu_inv_degree_modulus_quo[j] = calc_quotient<T>(coeff_modulus[j], mod_op);
        }

        // if (j < q_base_sz - 1)
        // {
        //     cpu_inv_q_last_mod_q_op[j] = inv_q_last_mod_q[j].operand;
        //     cpu_inv_q_last_mod_q_quo[j] = inv_q_last_mod_q[j].quotient;
        // }
    }

    // gpu_context_entry.xe_inv1->set_data(cpu_inv1.data(), q_base_sz);
    // gpu_context_entry.xe_inv2->set_data(cpu_inv2.data(), 2 * q_base_sz);
    // gpu_context_entry.xe_inv_q_last_mod_q_op->set_data(cpu_inv_q_last_mod_q_op.data(), q_base_sz - 1);
    // gpu_context_entry.xe_inv_q_last_mod_q_quo->set_data(cpu_inv_q_last_mod_q_quo.data(), q_base_sz - 1);

    gpu_context_entry.xe_prim_roots_op->set_data(cpu_roots_op.data(), q_base_sz * coeff_count);
    gpu_context_entry.xe_prim_roots_quo->set_data(cpu_roots_quo.data(), q_base_sz * coeff_count);

    gpu_context_entry.xe_inv_prim_roots_op->set_data(cpu_inv_roots_op.data(), q_base_sz * coeff_count);
    gpu_context_entry.xe_inv_prim_roots_quo->set_data(cpu_inv_roots_quo.data(), q_base_sz * coeff_count);

    gpu_context_entry.xe_inv_degree_op->set_data(cpu_inv_degree_modulus_op.data(), q_base_sz);
    gpu_context_entry.xe_inv_degree_quo->set_data(cpu_inv_degree_modulus_quo.data(), q_base_sz);

    gpu_context_entry.xe_modulus->set_data(cpu_modulus.data(), q_base_sz);

    gpu_context_entry.rns_base_size = q_base_sz;
    gpu_context_entry.coeff_count = coeff_count;
    gpu_context_entry.mod_inv_size = const_ratio_sz;
    gpu_context_entry.next_rns_base_size = next_coeff_modulus_size;
}


template<typename T>
std::shared_ptr<xehe::ext::Buffer<T>> generate_random_buffer(const xehe::ext::XeHE_mem_context<T> &gpu_context_entry, const std::vector<T> &modulus, size_t q_idx = 0){
    size_t rns_base_size = gpu_context_entry.rns_base_size; 
    size_t coeff_count = gpu_context_entry.coeff_count;
    size_t poly_len = rns_base_size * coeff_count;
    std::shared_ptr<xehe::ext::Buffer<T>> buf;
    buf = xehe::ext::XeHE_malloc<T>(poly_len, q_idx);

    random_device rd;
    std::vector<T> random_input(poly_len);
    for (size_t i = 0; i < rns_base_size; i++){
        for (size_t j = 0; j < coeff_count; j++){
            T temp = static_cast<T>(rd()) % modulus[i];
            random_input[i * coeff_count + j] = temp;
        }
    }
    buf->set_data(random_input.data(), poly_len);
    return buf;
}


template<typename T>
std::shared_ptr<xehe::ext::Buffer<T>> generate_random_buffer(const xehe::ext::XeHE_mem_context<T> &gpu_context_entry, const std::vector<T> &modulus, 
                                                            std::vector<T> &save_input, size_t q_idx = 0){
    size_t rns_base_size = gpu_context_entry.rns_base_size; 
    size_t coeff_count = gpu_context_entry.coeff_count;
    size_t poly_len = rns_base_size * coeff_count;
    std::shared_ptr<xehe::ext::Buffer<T>> buf;
    buf = xehe::ext::XeHE_malloc<T>(poly_len, q_idx);

    random_device rd;
    save_input = std::vector<T>(poly_len);
    for (size_t i = 0; i < rns_base_size; i++){
        for (size_t j = 0; j < coeff_count; j++){
            T temp = static_cast<T>(rd()) % modulus[i];
            save_input[i * coeff_count + j] = temp;
        }
    }
    buf->set_data(save_input.data(), poly_len);
    return buf;
}

template<typename T>
void NTT_correctness_generic(std::shared_ptr<xehe::ext::Buffer<T>> poly, std::vector<T> input, int time_loop, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
    const std::vector<T> & modulus, int ph_q_num, size_t q_idx = 0) {

    auto pq_idx = q_idx % ph_q_num;

    std::cout << "On queue logical #" << q_idx << " physical #" << pq_idx << std::endl;

    size_t rns_base_size = gpu_context_entry.rns_base_size;
    size_t coeff_count = gpu_context_entry.coeff_count;
    size_t poly_len = rns_base_size * coeff_count;
    for (int i = 0; i < time_loop; i++) {
//        std::vector<T> input;
        //std::shared_ptr<xehe::ext::Buffer<T>> poly = generate_random_buffer<T>(gpu_context_entry, modulus, input);

        xehe::ext::XeHE_NTT<T>(1, gpu_context_entry,
            gpu_context_entry.xe_modulus,
            gpu_context_entry.xe_prim_roots_op,
            gpu_context_entry.xe_prim_roots_quo,
            poly, nullptr, nullptr, pq_idx);

        xehe::ext::XeHE_invNTT<T>(1, gpu_context_entry,
            gpu_context_entry.xe_modulus,
            gpu_context_entry.xe_inv_prim_roots_op,
            gpu_context_entry.xe_inv_prim_roots_quo,
            poly,
            gpu_context_entry.xe_inv_degree_op,
            gpu_context_entry.xe_inv_degree_quo, pq_idx);

        std::vector<T> output(poly_len);
        poly->get_data(output.data(), poly_len);
        for (int j = 0; j < poly_len; j++) {
            if (input[j] != output[j]) {
                std::cout << "Difference in " << j << " position:" << std::endl;
                std::cout << "Input: " << input[j] << " Output: " << output[j] << std::endl;
                std::cout << "===========Correctness Check Failed!============" << std::endl;
                return;
            }
        }

    }

}

template<typename T>
Driver<T>::~Driver(void)
{
    size_t alloced, freed;
    xehe::ext::get_mem_cache_stat(alloced, freed);    
    //xehe::ext::free_memory_cache();
}

template<typename T>
void Driver<T>::benchmark_interface(int n, int time_loop, int prime_length, std::vector<int>* p_mods, size_t queue_num){
    // call benchmarking
    queue_num = ((queue_num + ph_q_num_ - 1)/ph_q_num_)*ph_q_num_;
    std::cout << "Current Benchmark is running on " << queue_num <<  " logical " << ph_q_num_  << " physical " <<  " queue(s)" << std::endl;
    
    // int prime_length = sizeof(T) * 8 - 4;
    std::cout << "Prime bit width is " << prime_length << std::endl;
    std::vector<int> def_mod(6, prime_length);

    std::vector<int> mods;
    static int t_n = 0;
    if (!p_mods)
    {
        for (const auto& m : def_mod)
        {
            mods.push_back(m);
        }
    }
    else
    {
        for (const auto& m : *p_mods)
        {
            mods.push_back(m);
        }
    }

    if (t_n !=n)
    {
        std::cout << "**************************************************************************" << std::endl;
        std::cout << "poly order: " << n*2
            << ", RNS base: " << mods.size()
            << ", loops: " << time_loop
            << std::endl;
        std::cout << "**************************************************************************" << std::endl;
        t_n =n;
    }
    
    seal::EncryptionParameters parms(seal::scheme_type::ckks);

    size_t slot_size = n;
    parms.set_poly_modulus_degree(slot_size * 2);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

    seal::SEALContext context(parms, false, seal::sec_level_type::none);
    
    params_to_gpu<T>(context, gpu_context_entry_, modulus_);
    std::vector<std::vector<T>> inputs;
    std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys;
    for (size_t q_idx = 0; q_idx < queue_num; q_idx++){
        auto pq_idx = q_idx % ph_q_num_;
        std::vector<T> input;
        std::shared_ptr<xehe::ext::Buffer<T>> poly = generate_random_buffer<T>(gpu_context_entry_, modulus_, input, pq_idx);
        inputs.push_back(input);
        polys.push_back(poly);
    }

    // Benchmarking
    
    std::cout << "-----  Correctness Check  -----" << std::endl;
    NTT_correctness(polys, inputs, gpu_context_entry_, modulus_);
    std::cout << "-----   Check Finished!   -----" << std::endl;

    xehe::ext::clear_events();
    NTT_benchmark(polys, time_loop, gpu_context_entry_, modulus_);
    xehe::ext::process_events();   

    xehe::ext::clear_events();
    invNTT_benchmark(polys, time_loop, gpu_context_entry_, modulus_);
    xehe::ext::process_events();
    
    std::cout << "Benchmarking Over" << std::endl;
}

template<typename T>
void Driver<T>::NTT_correctness(std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, std::vector<std::vector<T>> inputs, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
    const std::vector<T>& modulus){
    for (size_t q_idx = 0; q_idx < polys.size(); q_idx++){
        NTT_correctness_generic<T>(polys[q_idx], inputs[q_idx], 1, gpu_context_entry, modulus, ph_q_num_, q_idx);
    }
}

template<typename T>
void NTT_thread(size_t q_idx, size_t ph_q_num, std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, int time_loop, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
    const std::vector<T>& modulus){
    //
    
    auto p_idx = q_idx % ph_q_num;
    for (int i = 0; i < time_loop; i++)
    {
        //std::cout << q_idx << std::endl;
        size_t l_idx = q_idx;// * ph_q_num;
        //for(size_t p_idx = 0; p_idx < ph_q_num; ++p_idx)
        {
        xehe::ext::XeHE_NTT<T>(1, gpu_context_entry,
            gpu_context_entry.xe_modulus,
            gpu_context_entry.xe_prim_roots_op,
            gpu_context_entry.xe_prim_roots_quo,
            polys[l_idx],// + p_idx],
            nullptr,
            nullptr,
            p_idx, false, false);
        }
                     
    }

    //for(size_t p_idx = 0; p_idx < ph_q_num; ++p_idx)
    {
        xehe::ext::wait_for_queue(p_idx);
    }

    
    
}

#if _NTT_MT
template<typename T>
void Driver<T>::NTT_benchmark(std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, int time_loop, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
    const std::vector<T>& modulus){
    Timer timer;
    double accum_time = 0;
   
    std::cout << "-----  Multi-threading " <<  polys.size() << " -----" << std::endl;
    std::vector<thread> thread_pool;

    timer.start();

    //for (int k = 0; k < time_loop; k++){
    thread_pool.clear();

    for(int i = 0; i < polys.size(); ++i)
    {    
        thread_pool.push_back(std::thread(NTT_thread<T>, size_t(i), ph_q_num_, polys, time_loop, gpu_context_entry,modulus) );
    }    
    
    for(int i = 0; i < polys.size(); ++i)
    {
        thread_pool[i].join();
    }
    //}

    timer.stop();
    accum_time += timer.elapsedMicroseconds();

    std::cout << "Average NTT time " << accum_time/((double)time_loop * (polys.size()/ph_q_num_)) << " us" << std::endl;
    
}
#else
template<typename T>
void Driver<T>::NTT_benchmark(std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, int time_loop, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
    const std::vector<T>& modulus){
    Timer timer;
    double accum_time = 0;

    timer.start();
    for (int i = 0; i < time_loop; i++){
        for (size_t q_idx = 0; q_idx < polys.size(); q_idx++){
            auto pq_idx = size_t(q_idx % ph_q_num_);
            xehe::ext::XeHE_NTT<T>(1, gpu_context_entry,
                                gpu_context_entry.xe_modulus,
                                gpu_context_entry.xe_prim_roots_op,
                                gpu_context_entry.xe_prim_roots_quo,
                                polys[q_idx],
                                nullptr,
                                nullptr,
                                pq_idx, false, false);
        }
    }

    for (size_t q_idx = 0; q_idx < polys.size(); q_idx++)
    { 
        auto pq_idx = size_t(q_idx % ph_q_num_);
        xehe::ext::wait_for_queue(pq_idx);
    }
    // xehe::ext::wait_for_queue(polys.size() - 1);
    
    timer.stop();
    accum_time += timer.elapsedMicroseconds();

    std::cout << "Average NTT time " << accum_time/((double)time_loop * (polys.size()/ph_q_num_)) << " us" << std::endl;
    
}
#endif

template<typename T>
void Driver<T>::invNTT_benchmark(std::vector<std::shared_ptr<xehe::ext::Buffer<T>>> polys, int time_loop, const xehe::ext::XeHE_mem_context<T>& gpu_context_entry,
    const std::vector<T>& modulus){
    Timer timer;
    double accum_time = 0;

    timer.start();

    
    for (int i = 0; i < time_loop; i++){
        for (size_t q_idx = 0; q_idx < polys.size(); q_idx++){
            auto pq_idx = size_t(q_idx % ph_q_num_);
            xehe::ext::XeHE_invNTT<T>(1, gpu_context_entry,
                                gpu_context_entry.xe_modulus,
                                gpu_context_entry.xe_inv_prim_roots_op,
                                gpu_context_entry.xe_inv_prim_roots_quo,
                                polys[q_idx],
                                gpu_context_entry.xe_inv_degree_op,
                                gpu_context_entry.xe_inv_degree_quo,
                                pq_idx, false, false);
        }
    }

    for (size_t q_idx = 0; q_idx < polys.size(); q_idx++)
    {
        auto pq_idx = size_t(q_idx % ph_q_num_);
        xehe::ext::wait_for_queue(pq_idx);
    }
    // xehe::ext::wait_for_queue(polys.size() - 1);

    timer.stop();
    accum_time += timer.elapsedMicroseconds();
    
    std::cout << "Average inverse NTT time " << accum_time/((double)time_loop * (polys.size()/ph_q_num_)) << " us" << std::endl;
 
}

template class Driver<uint64_t>;
template class Driver<uint32_t>;