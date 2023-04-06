/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef WIN32
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch2/catch.hpp"
#endif
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

#define _MUL_ADD_ONLY_ false
#define _RELIN_ONLY_ false
#define _NO_WAIT_ false



#include <cstddef>
#include <cstdint>
#include <ctime>
#include <string>
#ifdef BUILD_WITH_SEAL
#include "seal/batchencoder.h"
#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/keygenerator.h"
#include "seal/modulus.h"
#endif //#ifdef BUILD_WITH_SEAL

using namespace std;

namespace xehetest {

    template <typename T>
    void XeTests_Evaluator(bool benchmark = false,
        int n = 1024*8,
        int data_bound_bits=10,
        int delta_bits=50,
        int outer_loop=10,
        int time_loop = 20,
        std::vector<int>* p_mods=nullptr)
    {

        std::vector<int> def_mod{ 60,60,60,60 };

        std::vector<int> mods;
        static int t_n = 0;
        static int t_data_bound_bits=0;
        static int t_delta_bits=0;
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

        if (t_n !=n || t_data_bound_bits != data_bound_bits || t_delta_bits != delta_bits)
        {
            std::cout << "**************************************************************************" << std::endl;
            std::cout << "poly order: " << n*2
                << ", data bound: " << (1 << data_bound_bits)
                << ", scale(log): " << delta_bits
                << ", RNS base: " << mods.size()
                << ", loops: " << outer_loop
                << std::endl;
            std::cout << "**************************************************************************" << std::endl;
            t_n =n;
            t_data_bound_bits = data_bound_bits;
            t_delta_bits = delta_bits;
        }
        
        xehe::ext::add_header(n*2, (1 << data_bound_bits), delta_bits, mods.size());

#if !_RELIN_ONLY_
#if !_MUL_ADD_ONLY_

        {
            std::cout << "BFV EncryptNegate " << std::endl;
            seal::EncryptionParameters parms(seal::scheme_type::bfv);
            seal::Modulus plain_modulus(1 << 6);
            parms.set_poly_modulus_degree(64);
            parms.set_plain_modulus(plain_modulus);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(64, { 40 }));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::Encryptor encryptor(context, pk);

            seal::Evaluator evaluator(context);


            seal::Decryptor decryptor(context, keygen.secret_key());

            seal::Ciphertext encrypted;
            encrypted.gpu();
            seal::Plaintext plain;

            plain = "1x^28 + 1x^25 + 1x^21 + 1x^20 + 1x^18 + 1x^14 + 1x^12 + 1x^10 + 1x^9 + 1x^6 + 1x^5 + 1x^4 + 1x^3";
            encryptor.encrypt(plain, encrypted);
            evaluator.negate_inplace(encrypted);
            decryptor.decrypt(encrypted, plain);
#ifdef WIN32
            if (plain.to_string() != "3Fx^28 + 3Fx^25 + 3Fx^21 + 3Fx^20 + 3Fx^18 + 3Fx^14 + 3Fx^12 + 3Fx^10 + 3Fx^9 + 3Fx^6 + 3Fx^5 + 3Fx^4 + 3Fx^3")
            {
                std::cout << "failed";
            }
            else
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;
#else
            REQUIRE(plain.to_string() == "3Fx^28 + 3Fx^25 + 3Fx^21 + 3Fx^20 + 3Fx^18 + 3Fx^14 + 3Fx^12 + 3Fx^10 + 3Fx^9 + 3Fx^6 + 3Fx^5 + 3Fx^4 + 3Fx^3");
#endif
        }

        {
            std::cout << "BFV EncryptAdd " << std::endl;
            seal::EncryptionParameters parms(seal::scheme_type::bfv);
            seal::Modulus plain_modulus(1 << 6);
            parms.set_poly_modulus_degree(64);
            parms.set_plain_modulus(plain_modulus);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(64, { 40 }));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::Encryptor encryptor(context, pk);

            seal::Evaluator evaluator(context);

            seal::Decryptor decryptor(context, keygen.secret_key());

            seal::Ciphertext encrypted1;
            seal::Ciphertext encrypted2;
            encrypted1.gpu();
            encrypted2.gpu();


            seal::Plaintext plain, plain1, plain2;

            plain1 = "1x^28 + 1x^25 + 1x^21 + 1x^20 + 1x^18 + 1x^14 + 1x^12 + 1x^10 + 1x^9 + 1x^6 + 1x^5 + 1x^4 + 1x^3";
            plain2 = "1x^18 + 1x^16 + 1x^14 + 1x^9 + 1x^8 + 1x^5 + 1";
            encryptor.encrypt(plain1, encrypted1);
            encryptor.encrypt(plain2, encrypted2);
            evaluator.add_inplace(encrypted1, encrypted2);
            decryptor.decrypt(encrypted1, plain);

#ifdef WIN32
            if (plain.to_string() != "1x^28 + 1x^25 + 1x^21 + 1x^20 + 2x^18 + 1x^16 + 2x^14 + 1x^12 + 1x^10 + 2x^9 + 1x^8 + "
                "1x^6 + 2x^5 + 1x^4 + 1x^3 + 1")
            {
                std::cout << "failed";
            }
            else
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;
#else
            REQUIRE(plain.to_string() == "1x^28 + 1x^25 + 1x^21 + 1x^20 + 2x^18 + 1x^16 + 2x^14 + 1x^12 + 1x^10 + 2x^9 + 1x^8 + "
                "1x^6 + 2x^5 + 1x^4 + 1x^3 + 1");
#endif
        }


        {
            std::cout << "BFV EncryptSub" << std::endl;

            seal::EncryptionParameters parms(seal::scheme_type::bfv);
            seal::Modulus plain_modulus(1 << 6);
            parms.set_poly_modulus_degree(64);
            parms.set_plain_modulus(plain_modulus);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(64, { 40 }));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::Encryptor encryptor(context, pk);

            seal::Evaluator evaluator(context);

            seal::Decryptor decryptor(context, keygen.secret_key());

            seal::Ciphertext encrypted1;
            seal::Ciphertext encrypted2;

            seal::Plaintext plain, plain1, plain2;

            plain1 = "1x^28 + 1x^25 + 1x^21 + 1x^20 + 1x^18 + 1x^14 + 1x^12 + 1x^10 + 1x^9 + 1x^6 + 1x^5 + 1x^4 + 1x^3";
            plain2 = "1x^18 + 1x^16 + 1x^14 + 1x^9 + 1x^8 + 1x^5 + 1";
            encryptor.encrypt(plain1, encrypted1.gpu());
            encryptor.encrypt(plain2, encrypted2.gpu());
            evaluator.sub_inplace(encrypted1, encrypted2);
            decryptor.decrypt(encrypted1, plain);

#ifdef WIN32
            if (plain.to_string() != "1x^28 + 1x^25 + 1x^21 + 1x^20 + 3Fx^16 + 1x^12 + 1x^10 + 3Fx^8 + 1x^6 + 1x^4 + 1x^3 + 3F")
            {
                std::cout << "failed";
            }
            else
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;
#else
            REQUIRE(plain.to_string() == "1x^28 + 1x^25 + 1x^21 + 1x^20 + 3Fx^16 + 1x^12 + 1x^10 + 3Fx^8 + 1x^6 + 1x^4 + 1x^3 + 3F");
#endif
        }


        {
            std::cout << "CKKS Encrypt0AddEncrypt0 " << std::endl;

            seal::EncryptionParameters parms(seal::scheme_type::ckks);
            // Adding two zero vectors
            size_t slot_size = 1024;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, { 30, 30, 30, 30, 30 }));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());

            seal::Evaluator evaluator(context);

            seal::Ciphertext encrypted;
            seal::Plaintext plain;
            seal::Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            const double delta = static_cast<double>(1ULL << 16);

            encoder.encode(input, context.first_parms_id(), delta, plain);

            encryptor.encrypt(plain, encrypted.gpu());
            seal::Ciphertext t_encrypted(encrypted);

            // XeHE
            evaluator.add_inplace(encrypted, encrypted);

            // Check correctness of encryption
            //ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

            decryptor.decrypt(encrypted, plainRes);
            encoder.decode(plainRes, output);
            if (!benchmark)
            {

                bool success = true;
                for (size_t i = 0; i < slot_size && success; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    if (tmp >= 0.5)
                    {
                        std::cout << "failed at slot " << i << " with err " << tmp;
                        success = false;
                    }

                }
                if (success)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;
#ifndef WIN32
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    REQUIRE(tmp < 0.5);
                }
#endif
            }

        }

#endif //#if !_MUL_ADD_ONLY_


        {
            // 2+2 add
            {
                std::cout << "CKKS EncryptAddEncrypt " << std::endl;

                // Adding two random vectors 50 times
                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods/*{ 30, 30, 30, 30, 30 }*/));

                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());

                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext t_encrypted1;
                seal::Ciphertext t_encrypted2;
                
                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plainRes;
                seal::Plaintext t_plainRes;

                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                vector<complex<double>> output(slot_size);
                vector<complex<double>> t_output(slot_size);

                //delta_bits = 30;
                //data_bound_bits = 16;

                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                srand(static_cast<unsigned>(time(NULL)));


                {
                    for (int round = 0; round < outer_loop && success; round++)
                    {
                        for (size_t i = 0; i < slot_size; i++)
                        {
                            input1[i] = static_cast<double>(rand() % data_bound);
                            input2[i] = static_cast<double>(rand() % data_bound);
                            expected[i] = input1[i] + input2[i];
                        }

                        encoder.encode(input1, context.first_parms_id(), delta, plain1);
                        encoder.encode(input2, context.first_parms_id(), delta, plain2);

                        encryptor.encrypt(plain1, encrypted1);
                        encryptor.encrypt(plain2, encrypted2);

                        t_encrypted1 = encrypted1;
                        t_encrypted2 = encrypted2;
                        t_encrypted1.gpu();
                        t_encrypted2.gpu();

                        if (benchmark)
                        {
                            break;
                        }

                        //std::cout << "inp h " << encrypted1.data(0)[0] << std::endl;
                        evaluator.add_inplace(encrypted1, encrypted2);
                        //std::cout << "out h " << encrypted1.data(0)[0] << std::endl;

                        // Check correctness of encryption
                        //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                        decryptor.decrypt(encrypted1, plainRes);
                        encoder.decode(plainRes, output);

                        //t_encrypted1.download();
                        //std::cout << "inp d " << t_encrypted1.data(0)[0] << std::endl;
                        evaluator.add_inplace(t_encrypted1, t_encrypted2);
                        //std::cout << "out d " << t_encrypted1.data(0)[0] << std::endl;
                        decryptor.decrypt(t_encrypted1, t_plainRes);
                        encoder.decode(t_plainRes, t_output);


                        for (size_t i = 0; i < slot_size && success; i++)
                        {
                            auto tmp = abs(expected[i].real() - output[i].real());
                            auto t_tmp = abs(expected[i].real() - t_output[i].real());

                            if (abs(t_tmp-tmp) > 0.00001 || tmp >= 0.5)
                            {
                                std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                                success = false;
                            }
#ifndef WIN32
                            REQUIRE((abs(t_tmp-tmp) <= 0.00001 && tmp < 0.5));
#endif
                        }

                    }

                    if (success && !benchmark)
                    {
                        std::cout << "succeed";
                    }
                    std::cout << std::endl;

                }

                if (benchmark)
                {
                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                    std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);

                    {
                        evaluator.add_inplace(tt_encrypted1[0], tt_encrypted2[0]);

                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {     
                            evaluator.add_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                        };
// #if _NO_WAIT_
//                         tt_encrypted1[time_loop-1].download();
// #endif
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "Add(noprefetch): " << to_string(duration_count/ time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }
                    {
                        evaluator.add_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            evaluator.add_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                        };
// #if _NO_WAIT_
//                         tt_encrypted1[time_loop-1].download();
// #endif
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "Add(prefetch): " << to_string(duration_count/time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("CKKS EncryptAddEncrypt", duration_count/time_loop, time_loop);
                    }
                    // stream breaker
                    // download GPU data into host
                    //t_encrypted1.download();
                }

            }
            // 2+3 add
            {
                if (!benchmark) std::cout << "CKKS EncryptAdd 2+3 Correctness Check" << std::endl;

                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));
                    
                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;
                    
                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] + input2[i] * input3[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta*delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta, plain3);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();

                    if (benchmark)
                    {
                        break;
                    }

                    // CPU
                    seal::Ciphertext encrypted_tmp;
                    evaluator.multiply_inplace(encrypted2, encrypted3);
                    evaluator.add(encrypted1, encrypted2, encrypted_tmp);
                    decryptor.decrypt(encrypted_tmp, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    seal::Ciphertext xe_encrypted_tmp;
                    evaluator.multiply_inplace(xe_encrypted2, xe_encrypted3);
                    evaluator.add(xe_encrypted1, xe_encrypted2, xe_encrypted_tmp);
                    decryptor.decrypt(xe_encrypted_tmp, xe_plainRes);
                    encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        auto t_tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || t_tmp >= 0.5 || abs(tmp - t_tmp)>=0.01)
                        {
                            std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((tmp < 0.5 && t_tmp < 0.5 && (abs(tmp - t_tmp) < 0.01)));
#endif
                    }

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;

            }
            
            // 3+2 add
            {
                if (!benchmark) std::cout << "CKKS EncryptAdd 3+2 Correctness Check" << std::endl;

                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));
                    
                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;
                    
                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] + input3[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta*delta, plain3);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();

                    if (benchmark)
                    {
                        break;
                    }

                    // CPU
                    seal::Ciphertext encrypted_tmp;
                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.add(encrypted1, encrypted3, encrypted_tmp);
                    decryptor.decrypt(encrypted_tmp, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    seal::Ciphertext xe_encrypted_tmp;
                    evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    evaluator.add(xe_encrypted1, xe_encrypted3, xe_encrypted_tmp);
                    decryptor.decrypt(xe_encrypted_tmp, xe_plainRes);
                    encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        auto t_tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || t_tmp >= 0.5 || abs(tmp - t_tmp)>=0.01)
                        {
                            std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((tmp < 0.5 && t_tmp < 0.5 && (abs(tmp - t_tmp) < 0.01)));
#endif
                    }

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;

            }
        }

        
        {
            // 2+2 add
            {
                std::cout << "CKKS EncryptSubEncrypt " << std::endl;

                // Adding two random vectors 50 times
                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods/*{ 30, 30, 30, 30, 30 }*/));

                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());

                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext t_encrypted1;
                seal::Ciphertext t_encrypted2;
                
                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plainRes;
                seal::Plaintext t_plainRes;

                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                vector<complex<double>> output(slot_size);
                vector<complex<double>> t_output(slot_size);

                //delta_bits = 30;
                //data_bound_bits = 16;

                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                srand(static_cast<unsigned>(time(NULL)));


                {
                    for (int round = 0; round < outer_loop && success; round++)
                    {
                        for (size_t i = 0; i < slot_size; i++)
                        {
                            input1[i] = static_cast<double>(rand() % data_bound);
                            input2[i] = static_cast<double>(rand() % data_bound);
                            expected[i] = input1[i] - input2[i];
                        }

                        encoder.encode(input1, context.first_parms_id(), delta, plain1);
                        encoder.encode(input2, context.first_parms_id(), delta, plain2);

                        encryptor.encrypt(plain1, encrypted1);
                        encryptor.encrypt(plain2, encrypted2);

                        t_encrypted1 = encrypted1;
                        t_encrypted2 = encrypted2;
                        t_encrypted1.gpu();
                        t_encrypted2.gpu();

                        if (benchmark)
                        {
                            break;
                        }

                        //std::cout << "inp h " << encrypted1.data(0)[0] << std::endl;
                        evaluator.sub_inplace(encrypted1, encrypted2);
                        //std::cout << "out h " << encrypted1.data(0)[0] << std::endl;

                        // Check correctness of encryption
                        //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                        decryptor.decrypt(encrypted1, plainRes);
                        encoder.decode(plainRes, output);

                        //t_encrypted1.download();
                        //std::cout << "inp d " << t_encrypted1.data(0)[0] << std::endl;
                        evaluator.sub_inplace(t_encrypted1, t_encrypted2);
                        //std::cout << "out d " << t_encrypted1.data(0)[0] << std::endl;
                        decryptor.decrypt(t_encrypted1, t_plainRes);
                        encoder.decode(t_plainRes, t_output);


                        for (size_t i = 0; i < slot_size && success; i++)
                        {
                            auto tmp = abs(expected[i].real() - output[i].real());
                            auto t_tmp = abs(expected[i].real() - t_output[i].real());

                            if (abs(t_tmp-tmp) > 0.00001 || tmp >= 0.5)
                            {
                                std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                                success = false;
                            }
#ifndef WIN32
                            REQUIRE((abs(t_tmp-tmp) <= 0.00001 && tmp < 0.5));
#endif
                        }

                    }

                    if (success && !benchmark)
                    {
                        std::cout << "succeed";
                    }
                    std::cout << std::endl;

                }

                if (benchmark)
                {
                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                    std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);

                    {
                        evaluator.sub_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {                  
                            evaluator.sub_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                        };
// #if _NO_WAIT_
//                         tt_encrypted1[time_loop-1].download();
// #endif
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "Sub(noprefetch): " << to_string(duration_count/ time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }
                    {
                        evaluator.sub_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            evaluator.sub_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                        };
// #if _NO_WAIT_
//                         tt_encrypted1[time_loop-1].download();
// #endif
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "Sub(prefetch): " << to_string(duration_count/time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("CKKS EncryptSubEncrypt", duration_count/time_loop, time_loop);
                    }
                    // stream breaker
                    // download GPU data into host
                    //t_encrypted1.download();
                }

            }
            // 2-3 sub
            {
                if (!benchmark) std::cout << "CKKS EncryptSub 2-3 Correctness Check" << std::endl;

                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));
                    
                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;
                    
                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] - input2[i] * input3[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta*delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta, plain3);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();

                    if (benchmark)
                    {
                        break;
                    }

                    // CPU
                    seal::Ciphertext encrypted_tmp;
                    evaluator.multiply_inplace(encrypted2, encrypted3);
                    evaluator.sub(encrypted1, encrypted2, encrypted_tmp);
                    decryptor.decrypt(encrypted_tmp, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    seal::Ciphertext xe_encrypted_tmp;
                    evaluator.multiply_inplace(xe_encrypted2, xe_encrypted3);
                    evaluator.sub(xe_encrypted1, xe_encrypted2, xe_encrypted_tmp);
                    decryptor.decrypt(xe_encrypted_tmp, xe_plainRes);
                    encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        auto t_tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || t_tmp >= 0.5 || abs(tmp - t_tmp)>=0.01)
                        {
                            std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((tmp < 0.5 && t_tmp < 0.5 && (abs(tmp - t_tmp) < 0.01)));
#endif
                    }

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;

            }
            
            // 3-2sub
            {
                if (!benchmark) std::cout << "CKKS EncryptSub 3-2 Correctness Check" << std::endl;

                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));
                    
                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;
                    
                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] - input3[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta*delta, plain3);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();

                    if (benchmark)
                    {
                        break;
                    }

                    // CPU
                    seal::Ciphertext encrypted_tmp;
                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.sub(encrypted1, encrypted3, encrypted_tmp);
                    decryptor.decrypt(encrypted_tmp, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    seal::Ciphertext xe_encrypted_tmp;
                    evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    evaluator.sub(xe_encrypted1, xe_encrypted3, xe_encrypted_tmp);
                    decryptor.decrypt(xe_encrypted_tmp, xe_plainRes);
                    encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        auto t_tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || t_tmp >= 0.5 || abs(tmp - t_tmp)>=0.01)
                        {
                            std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((tmp < 0.5 && t_tmp < 0.5 && (abs(tmp - t_tmp) < 0.01)));
#endif
                    }

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;

            }
        }

#if !_MUL_ADD_ONLY_

        {
            std::cout << "CKKS EncryptAddPlain" << std::endl;
            // Adding two random vectors 50 times
            seal::EncryptionParameters parms(seal::scheme_type::ckks);

            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());

            seal::Evaluator evaluator(context);

            seal::Ciphertext encrypted1;
            seal::Ciphertext t_encrypted1;
            encrypted1.gpu();
            seal::Plaintext plain1;
            seal::Plaintext plain2;
            seal::Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << data_bound_bits);
            const double delta = static_cast<double>(1ULL << delta_bits);

            srand(static_cast<unsigned>(time(NULL)));
            // warm up 
            bool success = true;

            
                for (int expCount = 0; expCount < outer_loop
                    && success
                    ; expCount++)
                {
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] + input2[i];
                    }

                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    // TO DO: this code has to be inside encoder
                    // the bit indicates the host data update
                    // it is used if plain buffer participates in any ops on GPU
                    encoder.set_dirty_bit(plain2);
                    encryptor.encrypt(plain1, encrypted1);
                    encrypted1.gpu();

                    if (benchmark)
                    {
                        t_encrypted1 = encrypted1;
                        break;
                    }

                    evaluator.add_plain_inplace(encrypted1, plain2);

                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                    decryptor.decrypt(encrypted1, plainRes);
                    encoder.decode(plainRes, output);

                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());

                        if (tmp >= 0.5)
                        {
                            std::cout << "failed at slot " << i << "with err " << tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE(tmp < 0.5);
#endif

                    }
                }
                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;
            

            if (benchmark)
            {

                std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                std::vector<seal::Plaintext> tt_plain2(time_loop + 1, plain2);
                {
                    evaluator.add_plain_inplace(tt_encrypted1[0], tt_plain2[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        //std:: cout << i << std::endl;                    
                        evaluator.add_plain_inplace(tt_encrypted1[i], tt_plain2[i]);

                    };
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Add plain(noprefetch): " << to_string(duration_count/ time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                {
                    evaluator.add_plain_inplace(tt_encrypted1[0], tt_plain2[0]);

                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        evaluator.add_plain_inplace(tt_encrypted1[i], tt_plain2[i]);
                    };
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Add plain(prefetch): " << to_string(duration_count/time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKS EncryptAddPlain", duration_count/time_loop, time_loop);
                }


            }


        }


        {
            std::cout << "CKKS EncryptSubPlain" << std::endl;
            // Adding two random vectors 50 times
            seal::EncryptionParameters parms(seal::scheme_type::ckks);

            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());

            seal::Evaluator evaluator(context);


            seal::Ciphertext encrypted1;
            encrypted1.gpu();
            seal::Ciphertext t_encrypted1;


            seal::Plaintext plain1;
            seal::Plaintext plain2;
            seal::Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);


            int data_bound = (1 << data_bound_bits);
            const double delta = static_cast<double>(1ULL << delta_bits);

            srand(static_cast<unsigned>(time(NULL)));
            // warm up 
            bool success = true;

            for (int expCount = 0; expCount < outer_loop
                && success
                ; expCount++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] - input2[i];
                }

                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);

                if (benchmark)
                {
                    t_encrypted1 = encrypted1;
                    break;
                }


                evaluator.sub_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size && success; i++)
                {

                    auto tmp = abs(expected[i].real() - output[i].real());

                    if (tmp >= 0.5)
                    {
                        std::cout << "failed at slot " << i << "with err " << tmp;
                        success = false;
                    }
#ifndef WIN32

                    REQUIRE(tmp < 0.5);
#endif

                }
            }
            if (success)
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;


            if (benchmark)
            {

                std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                std::vector<seal::Plaintext> tt_plain2(time_loop + 1, plain2);
                {
                    evaluator.sub_plain_inplace(tt_encrypted1[0], tt_plain2[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {                                     
                        evaluator.sub_plain_inplace(tt_encrypted1[i], tt_plain2[i]);
                        //std:: cout << i << std::endl;  
                    };
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Sub plain(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();

                }
                {
                    evaluator.sub_plain_inplace(tt_encrypted1[0], tt_plain2[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        evaluator.sub_plain_inplace(tt_encrypted1[i], tt_plain2[i]);
                    };
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Sub plain(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKS EncryptSubPlain", duration_count/time_loop, time_loop);
                }
                // stream breaker
                // download GPU data into host
                //t_encrypted1.download();
            }


        }


        {
            std::cout << "CKKS EncryptMultiplyByInt64Decrypt" << std::endl;

            seal::EncryptionParameters parms(seal::scheme_type::ckks);
            

            // Multiplying two random vectors by an integer
            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, { 60, 60, 40, 40, 40 }));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());
            seal::Evaluator evaluator(context);


            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);

            seal::Ciphertext encrypted1;
            seal::Plaintext plain1;
            seal::Plaintext plain2;
            seal::Plaintext plainRes;

            seal::Plaintext xe_plain1;
            seal::Plaintext xe_plain2;
            seal::Plaintext xe_plainRes;
            seal::Ciphertext t_encrypted1;
            seal::Plaintext t_plain2;


            vector<complex<double>> input1(slot_size, 0.0);
            int64_t input2;
            vector<complex<double>> expected(slot_size, 0.0);

            int data_bound = (1 << 10);
            srand(static_cast<unsigned>(time(NULL)));

            bool success = true;

        
                for (int iExp = 0; iExp < outer_loop && success; iExp++)
                {
                    auto r = rand() % data_bound;
                    input2 = (r > 1) ? r : 1;
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * static_cast<double>(input2);
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);

                    const double delta = static_cast<double>(1ULL << delta_bits);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), plain2);

                    encryptor.encrypt(plain1, encrypted1);


                    // make a copy of the input
                    // the be able to compare the outcome
                    seal::Ciphertext xe_encrypted1(encrypted1);
                    xe_encrypted1.gpu();



                    evaluator.multiply_plain_inplace(encrypted1, plain2);

                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                    decryptor.decrypt(encrypted1, plainRes);
                    encoder.decode(plainRes, output);

                    xehe_encoder.encode(input2, context.first_parms_id(), delta, xe_plain2);

                    if (benchmark)
                    {
                        t_encrypted1 = xe_encrypted1;
                        t_plain2 = xe_plain2;
                        break;
                    }
                    // TO DO: this code has to be inside encoder
                    // the bit indicates the host data update
                    // it is used if the plain buffer participates in any ops on GPU
                    //xehe_encoder.set_dirty_bit(xe_plain2);


                    xehe_evaluator.multiply_plain_inplace(xe_encrypted1, xe_plain2);

                    xehe_decryptor.decrypt(xe_encrypted1, xe_plainRes);
                    xehe_encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto xe_tmp = abs(xe_output[i].real() - output[i].real());
                        auto tmp = abs(expected[i].real() - output[i].real());

                        if (xe_tmp > 0.000001 || tmp >= 0.5)
                        {
                            std::cout << "failed at slot " << i << "with err " << tmp << " resid " << xe_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE(xe_tmp < 0.000001);
                        REQUIRE(tmp < 0.5);
#endif                       
                    }
                }
                

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;

            


                if (benchmark)
                {

                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop  + 1, t_encrypted1);
                    std::vector<seal::Plaintext> tt_plain2(time_loop + 1, t_plain2);
                    {
                        evaluator.multiply_plain_inplace(tt_encrypted1[0], tt_plain2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            evaluator.multiply_plain_inplace(tt_encrypted1[i], tt_plain2[i]);
                        };
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "Mul plain uint(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }
                    {
                        evaluator.multiply_plain_inplace(tt_encrypted1[0], tt_plain2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            evaluator.multiply_plain_inplace(tt_encrypted1[i], tt_plain2[i]);
                        };
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "Mul plain uint(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("CKKS EncryptMultiplyByInt64Decrypt", duration_count/time_loop, time_loop);
                    }

 

                }

        }


        {
            std::cout << "CKKS EncryptMultiplyByDoubleDecrypt" << std::endl;

            seal::EncryptionParameters parms(seal::scheme_type::ckks);


            // Multiplying two random vectors by a double
            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());
            seal::Evaluator evaluator(context);

            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);


            seal::Ciphertext encrypted1;
            seal::Plaintext plain1;
            seal::Plaintext plain2;
            seal::Plaintext plainRes;

            seal::Plaintext xe_plain1;
            seal::Plaintext xe_plain2;
            seal::Plaintext xe_plainRes;
            seal::Ciphertext t_encrypted1;
            seal::Plaintext t_plain2;


            vector<complex<double>> input1(slot_size, 0.0);
            double input2;
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> xe_output(slot_size);

            int data_bound = (1 << data_bound_bits);
            srand(static_cast<unsigned>(time(NULL)));


            bool success = true;

            for (int iExp = 0; iExp < outer_loop && success; iExp++)
            {
                input2 = static_cast<double>(uint64_t(rand()) % (uint64_t(data_bound) * uint64_t(data_bound))) / static_cast<double>(data_bound);
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2;
                }

                const double delta = static_cast<double>(1ULL << delta_bits);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);

                seal::Ciphertext xe_encrypted1(encrypted1);
                xe_encrypted1.gpu();

                evaluator.multiply_plain_inplace(encrypted1, plain2);



                // Check correctness of encryption
                //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);

                xehe_encoder.encode(input2, context.first_parms_id(), delta, xe_plain2);

                if (benchmark)
                {
                    t_encrypted1 = xe_encrypted1;
                    t_plain2 = xe_plain2;
                    break;
                }


                xehe_evaluator.multiply_plain_inplace(xe_encrypted1, xe_plain2);
                xehe_decryptor.decrypt(xe_encrypted1, xe_plainRes);
                xehe_encoder.decode(xe_plainRes, xe_output);


                for (size_t i = 0; i < slot_size && success; i++)
                {
                    auto xe_tmp = abs(xe_output[i].real() - output[i].real());

                    auto tmp = abs(expected[i].real() - output[i].real());

                    if (tmp >= 0.5)
                    {
                        std::cout << "failed at slot " << i << "with err " << tmp << " resid " << xe_tmp;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(xe_tmp < 0.000001);
                    REQUIRE(tmp < 0.5);
#endif                       

                }

            }


            if (success && !benchmark)
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;


            if (benchmark)
            {

                std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                std::vector<seal::Plaintext> tt_plain2(time_loop + 1, t_plain2);
                {
                    evaluator.multiply_plain_inplace(tt_encrypted1[0], tt_plain2[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {    
                        evaluator.multiply_plain_inplace(tt_encrypted1[i], tt_plain2[i]);
                    };
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Mul plain flt(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                {
                    evaluator.multiply_plain_inplace(tt_encrypted1[0], tt_plain2[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        evaluator.multiply_plain_inplace(tt_encrypted1[i], tt_plain2[i]);
                    };
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Mul plain flt(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKS EncryptMultiplyByDoubleDecrypt", duration_count/time_loop, time_loop);
                }



            }


        }

#endif //#if !_MUL_ADD_ONLY_

        {
            std::cout << "CKKS EncryptNaiveMultiplyDecrypt" << std::endl;

            // Multiplying two random vectors
            seal::EncryptionParameters parms(seal::scheme_type::ckks);

            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);
            seal::RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());

            seal::Evaluator evaluator(context);

            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);


            seal::Ciphertext encrypted1;
            seal::Ciphertext encrypted2;
            seal::Plaintext plain1;
            seal::Plaintext plain2;
            seal::Plaintext plainRes;

            seal::Plaintext xe_plain1;
            seal::Plaintext xe_plain2;
            seal::Plaintext xe_plainRes;
            seal::Ciphertext t_encrypted1;
            seal::Ciphertext t_encrypted2;



            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            vector<complex<double>> xe_output(slot_size);

            const double delta = static_cast<double>(1ULL << delta_bits);

            int data_bound = (1 << data_bound_bits);
            srand(static_cast<unsigned>(time(NULL)));
            bool success = true;


            for (int round = 0; round < outer_loop && success; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(i % data_bound); // rand() % data_bound);
                    input2[i] = static_cast<double>(i % data_bound); // rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);


                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);

                seal::Ciphertext xe_encrypted1(encrypted1);
                seal::Ciphertext xe_encrypted2(encrypted2);

                xe_encrypted1.gpu();
                if (benchmark)
                {
                    // gpu bit passed to assignee
                    t_encrypted1 = xe_encrypted1;
                    t_encrypted2 = xe_encrypted2.gpu();
                    break;
                }


                evaluator.multiply_inplace(encrypted1, encrypted2);

                // Check correctness of encryption
                //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);


                encoder.decode(plainRes, output);


                xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);

                // Check correctness of encryption
                //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                xehe_decryptor.decrypt(xe_encrypted1, xe_plainRes);


                xehe_encoder.decode(xe_plainRes, xe_output);

                for (size_t i = 0; i < slot_size && success; i++)
                {
                    auto xe_tmp = abs(xe_output[i].real() - output[i].real());
                    auto tmp = abs(expected[i].real() - xe_output[i].real());

                    if (xe_tmp >= 0.000001 || tmp >= 0.5)
                    {
                        std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(xe_tmp < 0.000001);
                    REQUIRE(tmp < 0.5);
#endif
                }
            }

            if (success && !benchmark)
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;



            if (benchmark) {
                std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);

                {
                    evaluator.multiply_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {                   
                        evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                    };
// #if _NO_WAIT_
//                     tt_encrypted1[time_loop-1].download();
// #endif
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Mul(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                    vector<complex<double>> inp_zero(slot_size, 0.0);
                    seal::Plaintext plain_zero;

                    xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                    seal::Ciphertext tt_zero;

                    xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                    std::vector<seal::Ciphertext> tt_encrypted11(time_loop + 1, t_encrypted1);

                    // move to gpu mem by adding zero
                    for (int i = 0; i < tt_encrypted11.size(); ++i)
                    {
                        xehe_evaluator.add_inplace(tt_encrypted11[i].gpu(), tt_zero);
                    }
// #if _NO_WAIT_
//                     tt_encrypted11[time_loop-1].download();
// #endif

                {
                    evaluator.multiply_inplace(tt_encrypted11[0], tt_encrypted2[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        evaluator.multiply_inplace(tt_encrypted11[i], tt_encrypted2[i]);

                    };
// #if _NO_WAIT_
//                     tt_encrypted11[time_loop-1].download();
// #endif
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Mul(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKS EncryptNaiveMultiplyDecrypt", duration_count/time_loop, time_loop);
                }
            }


        }
    

        {
            {
                std::cout << "CKKS EncryptMADDecrypt" << std::endl;
                // Multiplying two random vectors 10 times
                seal::EncryptionParameters parms(seal::scheme_type::ckks);
                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

                seal::SEALContext context(parms, true, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::CKKSEncoder xehe_encoder(context);
                seal::Encryptor xehe_encryptor(context, pk);
                seal::Decryptor xehe_decryptor(context, keygen.secret_key());
                seal::Evaluator xehe_evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;

                seal::Ciphertext t_encrypted1;
                seal::Ciphertext t_encrypted2;
                seal::Ciphertext t_encrypted3;


                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] + input3[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta*delta, plain3);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();

                    if (benchmark)
                    {
                        // gpu bit passed to assignee
                        t_encrypted1 = xe_encrypted1;
                        t_encrypted2 = xe_encrypted2;
                        t_encrypted3 = xe_encrypted3;
                        break;
                    }

                    // CPU
                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.add_inplace(encrypted3, encrypted1);

                    decryptor.decrypt(encrypted3, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    xehe_evaluator.multiply_add_inplace(xe_encrypted3, xe_encrypted1, xe_encrypted2);

                    xehe_decryptor.decrypt(xe_encrypted3, xe_plainRes);
                    xehe_encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto xe_tmp = abs(xe_output[i].real() - output[i].real());
                        auto tmp = abs(expected[i].real() - output[i].real());

                        if (tmp >= 0.5 || xe_tmp >=0.001)
                        {
                            std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch: " << xe_tmp << std::endl;
                            success = false;
                        }
    #ifndef WIN32
                        REQUIRE(tmp < 0.5);
                        REQUIRE(xe_tmp < 0.001);
    #endif
                    }
                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;

                if (benchmark) {
                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                    std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);
                    std::vector<seal::Ciphertext> tt_encrypted3(time_loop + 1, t_encrypted3);

                    {
                        xehe_evaluator.multiply_add_inplace(tt_encrypted3[0], tt_encrypted1[0], tt_encrypted2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            xehe_evaluator.multiply_add_inplace(tt_encrypted3[i], tt_encrypted1[i], tt_encrypted2[i]);
                        };
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MAD(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }
                        vector<complex<double>> inp_zero(slot_size, 0.0);
                        seal::Plaintext plain_zero;

                        xehe_encoder.encode(inp_zero, context.first_parms_id(), delta*delta, plain_zero);
                        seal::Ciphertext tt_zero;

                        xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                        std::vector<seal::Ciphertext> tt_encrypted33(time_loop + 1, t_encrypted3);

                        // move to gpu mem by adding zero
                        for (int i = 0; i < tt_encrypted33.size(); ++i)
                        {
                            xehe_evaluator.add_inplace(tt_encrypted33[i].gpu(), tt_zero);
                        }
                    {
                        xehe_evaluator.multiply_add_inplace(tt_encrypted33[0], tt_encrypted1[0], tt_encrypted2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            xehe_evaluator.multiply_add_inplace(tt_encrypted33[i], tt_encrypted1[i], tt_encrypted2[i]);
                        };
                        //tt_encrypted11[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MAD(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("CKKS EncryptMADDecrypt", duration_count/time_loop, time_loop);
                    }

                }
            }
            // 3+3 MAD
            {
                if (!benchmark) std::cout << "CKKS Encrypt MAD 3+2*2 Correctness Check" << std::endl;

                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));
                    
                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;
                seal::Ciphertext encrypted4;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plain4;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;
                    
                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> input4(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        input4[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] + input3[i] * input4[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta, plain3);
                    encoder.encode(input4, context.first_parms_id(), delta, plain4);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);
                    encryptor.encrypt(plain4, encrypted4);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);
                    seal::Ciphertext xe_encrypted4(encrypted4);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();
                    xe_encrypted4.gpu();

                    if (benchmark)
                    {
                        break;
                    }

                    // CPU
                    seal::Ciphertext encrypted_tmp;
                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.multiply_inplace(encrypted3, encrypted4);
                    evaluator.add(encrypted1, encrypted3, encrypted_tmp);
                    decryptor.decrypt(encrypted_tmp, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    seal::Ciphertext xe_encrypted_tmp;
                    evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    evaluator.multiply_add(xe_encrypted1, xe_encrypted3, xe_encrypted4, xe_encrypted_tmp);
                    decryptor.decrypt(xe_encrypted_tmp, xe_plainRes);
                    encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        auto t_tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || t_tmp >= 0.5 || abs(tmp - t_tmp)>=0.01)
                        {
                            std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((tmp < 0.5 && t_tmp < 0.5 && (abs(tmp - t_tmp) < 0.01)));
#endif
                    }

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;
            }
        }


        {
            {
                std::cout << "CKKS EncryptMADPlainDecrypt" << std::endl;
                // Multiplying two random vectors 10 times
                seal::EncryptionParameters parms(seal::scheme_type::ckks);
                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

                seal::SEALContext context(parms, true, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::CKKSEncoder xehe_encoder(context);
                seal::Encryptor xehe_encryptor(context, pk);
                seal::Decryptor xehe_decryptor(context, keygen.secret_key());
                seal::Evaluator xehe_evaluator(context);


                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;

                seal::Ciphertext t_encrypted1;
                seal::Ciphertext t_encrypted3;


                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = 1 << data_bound_bits;
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] + input3[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta * delta, plain3);

                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted3.gpu();
                    if (benchmark)
                    {
                        // gpu bit passed to assignee
                        t_encrypted1 = xe_encrypted1;
                        t_encrypted3 = xe_encrypted3;
                        break;
                    }

                    // CPU
                    evaluator.multiply_plain_inplace(encrypted1, plain2);
                    evaluator.add_inplace(encrypted3, encrypted1);

                    decryptor.decrypt(encrypted3, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    xehe_evaluator.multiply_plain_add_inplace(xe_encrypted3, xe_encrypted1, plain2);

                    xehe_decryptor.decrypt(xe_encrypted3, xe_plainRes);
                    xehe_encoder.decode(xe_plainRes, xe_output);

                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto xe_tmp = abs(xe_output[i].real() - output[i].real());
                        auto tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || xe_tmp >=0.001)
                        {
                            std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch: " << xe_tmp << std::endl;
                            success = false;
                        }
    #ifndef WIN32
                        REQUIRE(tmp < 0.5);
                        REQUIRE(xe_tmp < 0.001);
    #endif
                    }
                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;


                if (benchmark) {
                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                    std::vector<seal::Plaintext> tt_plain2(time_loop + 1, plain2);
                    std::vector<seal::Ciphertext> tt_encrypted3(time_loop + 1, t_encrypted3);

                    {
                        xehe_evaluator.multiply_plain_add_inplace(tt_encrypted3[0], tt_encrypted1[0], tt_plain2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            xehe_evaluator.multiply_plain_add_inplace(tt_encrypted3[i], tt_encrypted1[i], tt_plain2[i]);
                        };
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MADPlain(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }
                        vector<complex<double>> inp_zero(slot_size, 0.0);
                        seal::Plaintext plain_zero;

                        xehe_encoder.encode(inp_zero, context.first_parms_id(), delta*delta, plain_zero);
                        seal::Ciphertext tt_zero;

                        xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                        std::vector<seal::Ciphertext> tt_encrypted33(time_loop + 1, t_encrypted3);

                        // move to gpu mem by adding zero
                        for (int i = 0; i < tt_encrypted33.size(); ++i)
                        {
                            xehe_evaluator.add_inplace(tt_encrypted33[i].gpu(), tt_zero);
                        }
                    {
                        xehe_evaluator.multiply_plain_add_inplace(tt_encrypted33[0], tt_encrypted1[0], tt_plain2[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            xehe_evaluator.multiply_plain_add_inplace(tt_encrypted33[i], tt_encrypted1[i], tt_plain2[i]);
                        };
                        //tt_encrypted11[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MADPlain(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("CKKSEncryptMADPlainDecrypt", duration_count/time_loop, time_loop);

                    }

                }

            }
            // 3+2 MAD plain
            {
                if (!benchmark) std::cout << "CKKS Encrypt MAD plain 3+2*p Correctness Check" << std::endl;

                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));
                    
                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plain4;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;
                    
                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> input4(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        input4[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] + input3[i] * input4[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta, plain3);
                    encoder.encode(input4, context.first_parms_id(), delta, plain4);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();

                    if (benchmark)
                    {
                        break;
                    }

                    // CPU
                    seal::Ciphertext encrypted_tmp;
                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.multiply_plain_inplace(encrypted3, plain4);
                    evaluator.add(encrypted1, encrypted3, encrypted_tmp);
                    decryptor.decrypt(encrypted_tmp, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    seal::Ciphertext xe_encrypted_tmp;
                    evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    evaluator.multiply_plain_add(xe_encrypted1, xe_encrypted3, plain4, xe_encrypted_tmp);
                    decryptor.decrypt(xe_encrypted_tmp, xe_plainRes);
                    encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        auto t_tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || t_tmp >= 0.5 || abs(tmp - t_tmp)>=0.01)
                        {
                            std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((tmp < 0.5 && t_tmp < 0.5 && (abs(tmp - t_tmp) < 0.01)));
#endif
                    }

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;
            }
            
            // 2+3 MAD plain
            {
                if (!benchmark) std::cout << "CKKS Encrypt MAD plain 2+3*p Correctness Check" << std::endl;

                seal::EncryptionParameters parms(seal::scheme_type::ckks);

                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));
                    
                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encrypted3;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plain3;
                seal::Plaintext plain4;
                seal::Plaintext plainRes;
                seal::Plaintext xe_plainRes;
                    
                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> input3(slot_size, 0.0);
                vector<complex<double>> input4(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        input3[i] = static_cast<double>(rand() % data_bound);
                        input4[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] + input2[i] * input3[i] * input4[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta * delta * delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);
                    encoder.encode(input3, context.first_parms_id(), delta, plain3);
                    encoder.encode(input4, context.first_parms_id(), delta, plain4);
                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);
                    encryptor.encrypt(plain3, encrypted3);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    seal::Ciphertext xe_encrypted3(encrypted3);

                    xe_encrypted1.gpu();
                    xe_encrypted2.gpu();
                    xe_encrypted3.gpu();

                    if (benchmark)
                    {
                        break;
                    }

                    // CPU
                    seal::Ciphertext encrypted_tmp;
                    evaluator.multiply_inplace(encrypted2, encrypted3);
                    evaluator.multiply_plain_inplace(encrypted2, plain4);
                    evaluator.add(encrypted1, encrypted2, encrypted_tmp);
                    decryptor.decrypt(encrypted_tmp, plainRes);
                    encoder.decode(plainRes, output);

                    // GPU
                    seal::Ciphertext xe_encrypted_tmp;
                    evaluator.multiply_inplace(xe_encrypted2, xe_encrypted3);
                    evaluator.multiply_plain_add(xe_encrypted1, xe_encrypted2, plain4, xe_encrypted_tmp);
                    decryptor.decrypt(xe_encrypted_tmp, xe_plainRes);
                    encoder.decode(xe_plainRes, xe_output);


                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        auto t_tmp = abs(expected[i].real() - xe_output[i].real());

                        if (tmp >= 0.5 || t_tmp >= 0.5 || abs(tmp - t_tmp)>=0.01)
                        {
                            std::cout << "failed at round " << round << " slot " << i << "  with h err " << tmp << " d err " << t_tmp;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((tmp < 0.5 && t_tmp < 0.5 && (abs(tmp - t_tmp) < 0.01)));
#endif
                    }

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;
            }
        }

#endif // #if !_RELIN_ONLY_

#if !_MUL_ADD_ONLY_

        {
            std::cout << "CKKS EncryptMulLinDecrypt" << std::endl;
            // Multiplying two random vectors 10 times
            seal::EncryptionParameters parms(seal::scheme_type::ckks);

            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods/*{ 60, 30, 30, 30 }*/));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);
            seal::RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());
            seal::Evaluator evaluator(context);

            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);


            seal::Ciphertext encrypted1;
            seal::Ciphertext encrypted2;


            seal::Plaintext plain1;
            seal::Plaintext plain2;
            seal::Plaintext plainRes;
            seal::Plaintext xe_plainRes;
            seal::Ciphertext t_encrypted1;
            seal::Ciphertext t_encrypted2;


            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            int data_bound = 1 << data_bound_bits;
            const double delta = static_cast<double>(1ULL << delta_bits);

            bool success = true;

            for (int round = 0; round < outer_loop && success; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }

                vector<complex<double>> output(slot_size);
                vector<complex<double>> xe_output(slot_size);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                //encryptor.encrypt(plain1, xe_encrypted1);
                encryptor.encrypt(plain2, encrypted2);

                // Check correctness of encryption
                //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                //ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());



                //evaluator.multiply_inplace(encrypted1, encrypted2);
                //std::cout << "Scale fresh: " << log2(encrypted1.scale()) << " bits" << std::endl;
                seal::Ciphertext xe_encrypted1(encrypted1);
                seal::Ciphertext xe_encrypted2(encrypted2);
                xe_encrypted1.gpu();
                if (benchmark)
                {
                    // gpu bit passed to assignee
                    t_encrypted1 = xe_encrypted1;
                    t_encrypted2 = xe_encrypted2.gpu();
                    break;
                }

                evaluator.multiply_inplace(encrypted1, encrypted2);

                evaluator.relinearize_inplace(encrypted1, rlk);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);

                xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);

                //std::cout << "Scale after mul: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;
                xehe_evaluator.relinearize_inplace(xe_encrypted1, rlk);

                //std::cout << "Scale after relin: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;


                xehe_decryptor.decrypt(xe_encrypted1, xe_plainRes);
                xehe_encoder.decode(xe_plainRes, xe_output);

                for (size_t i = 0; i < slot_size && success; i++)
                {

                    auto xe_tmp = abs(xe_output[i].real() - output[i].real());
                    auto tmp = abs(expected[i].real() - xe_output[i].real());

                    if (xe_tmp >= 0.00001 || tmp >= 0.5)
                    {
                        std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(xe_tmp < 0.000001);
                    REQUIRE(tmp < 0.5);
#endif
                }
            }

            if (success && !benchmark)
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;




            if (benchmark) {
                std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);

                {
                    xehe_evaluator.multiply_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                    xehe_evaluator.relinearize_inplace(tt_encrypted1[0], rlk);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted1[i], rlk);
                    };
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "MulRelin(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                    vector<complex<double>> inp_zero(slot_size, 0.0);
                    seal::Plaintext plain_zero;

                    xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                    seal::Ciphertext tt_zero;

                    xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                    std::vector<seal::Ciphertext> tt_encrypted11(time_loop + 1, t_encrypted1);

                    // move to gpu mem by adding zero
                    for (int i = 0; i < tt_encrypted11.size(); ++i)
                    {
                        xehe_evaluator.add_inplace(tt_encrypted11[i].gpu(), tt_zero);
                    }
                    //tt_encrypted11[time_loop-1].download();
                {
                    xehe_evaluator.multiply_inplace(tt_encrypted11[0], tt_encrypted2[0]);
                    xehe_evaluator.relinearize_inplace(tt_encrypted11[0], rlk);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted11[i], tt_encrypted2[i]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted11[i], rlk);
                    };
                    //tt_encrypted11[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "MulRelin(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKSEncryptMulLinDecrypt", duration_count/time_loop, time_loop);
                }

            }

        }


#if !_RELIN_ONLY_


        {
            std::cout << "CKKS EncryptMulLinRescaleDecrypt" << std::endl;

            {
                // Multiplying two random vectors 10 times
                seal::EncryptionParameters parms(seal::scheme_type::ckks);
                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

                seal::SEALContext context(parms, true, seal::sec_level_type::none);
                //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::CKKSEncoder xehe_encoder(context);
                seal::Encryptor xehe_encryptor(context, pk);
                seal::Decryptor xehe_decryptor(context, keygen.secret_key());
                seal::Evaluator xehe_evaluator(context);



                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encryptedRes;

                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plainRes;

                seal::Ciphertext xe_encryptedRes;
                seal::Plaintext xe_plainRes;
                seal::Ciphertext t_encrypted1;
                seal::Ciphertext t_encrypted2;

                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                double delta = static_cast<double>(1ULL << delta_bits);
                int data_bound = 1 << data_bound_bits;

                bool success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);

                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);

                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);

                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    xe_encrypted1.gpu();

                    if (benchmark)
                    {
                        // gpu bit passed to assignee
                        t_encrypted1 = xe_encrypted1;
                        t_encrypted2 = xe_encrypted2.gpu();
                        break;
                    }

                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.relinearize_inplace(encrypted1, rlk);
                    evaluator.rescale_to_next_inplace(encrypted1);
                    decryptor.decrypt(encrypted1, plainRes);
                    encoder.decode(plainRes, output);

                    //std::cout << "Scale fresh: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;
                    xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    //std::cout << "Scale after mul: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;
                    xehe_evaluator.relinearize_inplace(xe_encrypted1, rlk);
                    //std::cout << "Scale after relin: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;
                    xehe_evaluator.rescale_to_next(xe_encrypted1, xe_encryptedRes);
                    //std::cout << "Scale after rescale: " << log2(xe_encryptedRes.scale()) << " bits" << std::endl;

                    // Check correctness of modulus switching
                    //ASSERT_TRUE(encrypted1.parms_id() == next_parms_id);



                    xehe_decryptor.decrypt(xe_encryptedRes, xe_plainRes);
                    xehe_encoder.decode(xe_plainRes, xe_output);

                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto xe_tmp = abs(output[i].real() - xe_output[i].real());
                        auto tmp = abs(expected[i].real() - xe_output[i].real());

                        if (xe_tmp >= 0.000001 || tmp >= 0.5)
                        {
                            std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp << std::endl;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE(xe_tmp < 0.000001);
                        REQUIRE(tmp < 0.5);
#endif
                    }
                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;



                if (benchmark) {
                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                    std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);

                    seal::Ciphertext tt_encryptedRes;

                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted1[0], rlk);
                        xehe_evaluator.rescale_to_next(tt_encrypted1[0], tt_encryptedRes);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            xehe_evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                            xehe_evaluator.relinearize_inplace(tt_encrypted1[i], rlk);
                            xehe_evaluator.rescale_to_next(tt_encrypted1[i], tt_encryptedRes);
                        };
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MulRelinRescale(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }

                    vector<complex<double>> inp_zero(slot_size, 0.0);
                    seal::Plaintext plain_zero;

                    xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                    seal::Ciphertext tt_zero;

                    xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                    std::vector<seal::Ciphertext> tt_encrypted11(time_loop + 1, t_encrypted1);

                    // move to gpu mem by adding zero
                    for (int i = 0; i < tt_encrypted11.size(); ++i)
                    {
                        xehe_evaluator.add_inplace(tt_encrypted11[i].gpu(), tt_zero);
                    }
                    //tt_encrypted11[time_loop-1].download();
                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted11[0], tt_encrypted2[0]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted11[0], rlk);
                        xehe_evaluator.rescale_to_next(tt_encrypted11[0], tt_encryptedRes);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            evaluator.multiply_inplace(tt_encrypted11[i], tt_encrypted2[i]);
                            xehe_evaluator.relinearize_inplace(tt_encrypted11[i], rlk);
                            xehe_evaluator.rescale_to_next(tt_encrypted11[i], tt_encryptedRes);
                        };
                        //tt_encrypted11[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MulRelinRescale(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("CKKS EncryptMulLinRescaleDecrypt", duration_count/time_loop, time_loop);
                    }
                }


            }


            {
                // Multiplying two random vectors 10 times
                seal::EncryptionParameters parms(seal::scheme_type::ckks);
                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create((slot_size * 2), mods));

                seal::SEALContext context(parms, true, seal::sec_level_type::none);
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);
                seal::RelinKeys rlk;
                keygen.create_relin_keys(rlk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::CKKSEncoder xehe_encoder(context);
                seal::Encryptor xehe_encryptor(context, pk);
                seal::Decryptor xehe_decryptor(context, keygen.secret_key());
                seal::Evaluator xehe_evaluator(context);


                seal::Ciphertext encrypted1;
                seal::Ciphertext encrypted2;
                seal::Ciphertext encryptedRes;
                seal::Plaintext plain1;
                seal::Plaintext plain2;
                seal::Plaintext plainRes;

                seal::Plaintext xe_plain1;
                seal::Plaintext xe_plain2;
                seal::Plaintext xe_plainRes;
                seal::Ciphertext t_encrypted1;
                seal::Ciphertext t_encrypted2;


                vector<complex<double>> input1(slot_size, 0.0);
                vector<complex<double>> input2(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                int data_bound = 1 << data_bound_bits;
                double delta = static_cast<double>(1ULL << delta_bits);


                bool success = true;
                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] * input2[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);
                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);

                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);

                    xe_encrypted1.gpu();
                    if (benchmark)
                    {
                        // gpu bit passed to assignee
                        t_encrypted1 = xe_encrypted1;
                        t_encrypted2 = xe_encrypted2.gpu();
                        break;
                    }


                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.relinearize_inplace(encrypted1, rlk);
                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.relinearize_inplace(encrypted1, rlk);

                    xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    xehe_evaluator.relinearize_inplace(xe_encrypted1, rlk);
                    xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    xehe_evaluator.relinearize_inplace(xe_encrypted1, rlk);

                    // Scale down by two levels
                    auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                    evaluator.rescale_to_inplace(encrypted1, target_parms);
                    xehe_evaluator.rescale_to_inplace(xe_encrypted1, target_parms);

                    // Check correctness of modulus switching
                    //ASSERT_TRUE(encrypted1.parms_id() == target_parms);

                    decryptor.decrypt(encrypted1, plainRes);
                    encoder.decode(plainRes, output);

                    xehe_decryptor.decrypt(xe_encrypted1, xe_plainRes);
                    xehe_encoder.decode(xe_plainRes, xe_output);

                    // Check correctness of modulus switching
                    //ASSERT_TRUE(encrypted1.parms_id() == target_parms);

                    decryptor.decrypt(encrypted1, plainRes);
                    encoder.decode(plainRes, output);



                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto xe_tmp = abs(output[i].real() - xe_output[i].real());
                        auto tmp = abs(expected[i].real() - xe_output[i].real());

                        if (xe_tmp >= 0.000001 || tmp >= 0.5)
                        {
                            std::cout << "failed1 at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp << std::endl;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE(xe_tmp < 0.000001);
                        REQUIRE(tmp < 0.5);
#endif
                    }


                }


                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;



                if (benchmark) {

                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                    std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);
                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted1[0], rlk);
                        xehe_evaluator.multiply_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted1[0], rlk);
                        // Scale down by two levels
                        auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                        xehe_evaluator.rescale_to_inplace(tt_encrypted1[0], target_parms);
                        
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            xehe_evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                            xehe_evaluator.relinearize_inplace(tt_encrypted1[i], rlk);
                            xehe_evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                            xehe_evaluator.relinearize_inplace(tt_encrypted1[i], rlk);
                            // Scale down by two levels
                            // auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                            xehe_evaluator.rescale_to_inplace(tt_encrypted1[i], target_parms);
                        }
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MulRelinMulRelinRescale(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }

                    vector<complex<double>> inp_zero(slot_size, 0.0);
                    seal::Plaintext plain_zero;

                    xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                    seal::Ciphertext tt_zero;

                    xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                    std::vector<seal::Ciphertext> tt_encrypted11(time_loop + 1, t_encrypted1);

                    // move to gpu mem by adding zero
                    for (int i = 0; i < tt_encrypted11.size(); ++i)
                    {
                        xehe_evaluator.add_inplace(tt_encrypted11[i].gpu(), tt_zero);
                    }
                    //tt_encrypted11[time_loop-1].download();
                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted11[0], tt_encrypted2[0]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted11[0], rlk);
                        xehe_evaluator.multiply_inplace(tt_encrypted11[0], tt_encrypted2[0]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted11[0], rlk);
                        auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                        xehe_evaluator.rescale_to_inplace(tt_encrypted11[0], target_parms);

                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();
                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            xehe_evaluator.multiply_inplace(tt_encrypted11[i], tt_encrypted2[i]);
                            xehe_evaluator.relinearize_inplace(tt_encrypted11[i], rlk);
                            xehe_evaluator.multiply_inplace(tt_encrypted11[i], tt_encrypted2[i]);
                            xehe_evaluator.relinearize_inplace(tt_encrypted11[i], rlk);
                            // Scale down by two levels
                            // auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                            xehe_evaluator.rescale_to_inplace(tt_encrypted11[i], target_parms);
                        };
                        //tt_encrypted11[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MulRelinMulRelinRescale(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }

                }
            

            

                // Test with inverted order: rescale then relin
                success = true;

                for (int round = 0; round < outer_loop && success; round++)
                {
                    srand(static_cast<unsigned>(time(NULL)));
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input1[i] = static_cast<double>(rand() % data_bound);
                        input2[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input1[i] * input2[i] * input2[i];
                    }

                    vector<complex<double>> output(slot_size);
                    vector<complex<double>> xe_output(slot_size);

                    encoder.encode(input1, context.first_parms_id(), delta, plain1);
                    encoder.encode(input2, context.first_parms_id(), delta, plain2);

                    encryptor.encrypt(plain1, encrypted1);
                    encryptor.encrypt(plain2, encrypted2);

                    seal::Ciphertext xe_encrypted1(encrypted1);
                    seal::Ciphertext xe_encrypted2(encrypted2);
                    xe_encrypted1.gpu();
                    if (benchmark)
                    {
                        // gpu bit passed to assignee
                        t_encrypted1 = xe_encrypted1;
                        t_encrypted2 = xe_encrypted2.gpu();
                        break;
                    }

                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                    // Check correctness of encryption
                    //ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                    evaluator.multiply_inplace(encrypted1, encrypted2);
                    evaluator.relinearize_inplace(encrypted1, rlk);
                    evaluator.multiply_inplace(encrypted1, encrypted2);

                    xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                    xehe_evaluator.relinearize_inplace(xe_encrypted1, rlk);
                    xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);


                    // Scale down by two levels
                    auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                    evaluator.rescale_to_inplace(encrypted1, target_parms);
                    xehe_evaluator.rescale_to_inplace(xe_encrypted1, target_parms);

                    // Relinearize now
                    evaluator.relinearize_inplace(encrypted1, rlk);
                    xehe_evaluator.relinearize_inplace(xe_encrypted1, rlk);

                    // Check correctness of modulus switching
                    //ASSERT_TRUE(encrypted1.parms_id() == target_parms);

                    decryptor.decrypt(encrypted1, plainRes);
                    encoder.decode(plainRes, output);

                    xehe_decryptor.decrypt(xe_encrypted1, xe_plainRes);
                    xehe_encoder.decode(xe_plainRes, xe_output);

                    for (size_t i = 1; i < slot_size && success; i++)
                    {
                        auto xe_tmp = abs(output[i].real() - xe_output[i].real());
                        auto tmp = abs(expected[i].real() - xe_output[i].real());

                        if (xe_tmp >= 0.000001 || tmp >= 0.5)
                        {
                            std::cout << "failed2 at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp << std::endl;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE((xe_tmp < 0.000001 && tmp < 0.5));
#endif
                    }
                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;





                if (benchmark) {
                    std::vector<seal::Ciphertext> tt_encrypted1(time_loop, t_encrypted1);
                    std::vector<seal::Ciphertext> tt_encrypted2(time_loop, t_encrypted2);
                    {
                        auto start = std::chrono::high_resolution_clock::now();

                        for (int i = 0; i < time_loop; ++i)
                        {
                            xehe_evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                            xehe_evaluator.relinearize_inplace(tt_encrypted1[i], rlk);
                            xehe_evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);


                            // Scale down by two levels
                            auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                            xehe_evaluator.rescale_to_inplace(tt_encrypted1[i], target_parms);

                            // Relinearize now
                            xehe_evaluator.relinearize_inplace(tt_encrypted1[i], rlk);
                        }
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "MulLinMulRescaleRelin(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;


                    }

                }


            }

        }


        {
            std::cout << "CKKS EncryptSquareRelinRescaleDecrypt" << std::endl;
            seal::EncryptionParameters parms(seal::scheme_type::ckks);

            // Squaring two random vectors 10 times
            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, true, seal::sec_level_type::none);
            //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);
            seal::RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());
            seal::Evaluator evaluator(context);


            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);

            seal::Ciphertext encrypted;

            seal::Ciphertext xe_encryptedRes;
            seal::Plaintext plain;
            seal::Plaintext plainRes;
            seal::Plaintext xe_plainRes;
            seal::Ciphertext t_encrypted;


            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> xe_output(slot_size);
            vector<complex<double>> expected(slot_size, 0.0);
            int data_bound = 1 << data_bound_bits;
            double delta = static_cast<double>(1ULL << delta_bits);

            bool success = true;


            for (int round = 0; round < outer_loop && success; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input[i] * input[i];
                }

                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);

                // copy of input encryption
                seal::Ciphertext xe_encrypted(encrypted);
                xe_encrypted.gpu();

                if (benchmark)
                {
                    // gpu bit passed to assignee
                    t_encrypted = xe_encrypted;
                    break;
                }

                // Check correctness of encryption
                //ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                evaluator.square_inplace(encrypted);
                evaluator.relinearize_inplace(encrypted, rlk);
                evaluator.rescale_to_next_inplace(encrypted);

                // Check correctness of modulus switching
               // ASSERT_TRUE(encrypted.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);



                //std::cout << "Scale fresh: " << log2(xe_encrypted.scale()) << " bits" << std::endl;

                xehe_evaluator.square_inplace(xe_encrypted);
                //std::cout << "Scale after square: " << log2(xe_encrypted.scale()) << " bits" << std::endl;
                xehe_evaluator.relinearize_inplace(xe_encrypted, rlk);
                //std::cout << "Scale after relin: " << log2(xe_encrypted.scale()) << " bits" << std::endl;
                xehe_evaluator.rescale_to_next(xe_encrypted, xe_encryptedRes);
                //std::cout << "Scale after rescale: " << log2(xe_encryptedRes.scale()) << " bits" << std::endl;



                xehe_decryptor.decrypt(xe_encryptedRes, xe_plainRes);
                xehe_encoder.decode(xe_plainRes, xe_output);

                for (size_t i = 0; i < slot_size && success; i++)
                {
                    auto xe_tmp = abs(output[i].real() - xe_output[i].real());

                    auto tmp = abs(expected[i].real() - output[i].real());

                    if (xe_tmp >= 0.000001 || tmp >= 0.5)
                    {
                        std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp << std::endl;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(xe_tmp < 0.000001);
                    REQUIRE(tmp < 0.5);
#endif
                }
            }
            if (success && !benchmark)
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;

            if (benchmark) {
                std::vector<seal::Ciphertext> tt_encrypted(time_loop + 1, t_encrypted);
                seal::Ciphertext tt_encryptedRes;
                {
                    xehe_evaluator.square_inplace(tt_encrypted[0]);
                    xehe_evaluator.relinearize_inplace(tt_encrypted[0], rlk);
                    xehe_evaluator.rescale_to_next(tt_encrypted[0], tt_encryptedRes);

                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.square_inplace(tt_encrypted[i]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted[i], rlk);
                        xehe_evaluator.rescale_to_next(tt_encrypted[i], tt_encryptedRes);
                    }
                    //tt_encrypted[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "SqrRelinRescale(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                vector<complex<double>> inp_zero(slot_size, 0.0);
                seal::Plaintext plain_zero;

                xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                seal::Ciphertext tt_zero;

                xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                std::vector<seal::Ciphertext> tt_encrypted11(time_loop + 1, t_encrypted);

                // move to gpu mem by adding zero
                for (int i = 0; i < tt_encrypted11.size(); ++i)
                {
                    xehe_evaluator.add_inplace(tt_encrypted11[i].gpu(), tt_zero);
                }
                //tt_encrypted11[time_loop-1].download();
                {
                    xehe_evaluator.square_inplace(tt_encrypted11[0]);
                    xehe_evaluator.relinearize_inplace(tt_encrypted11[0], rlk);
                    xehe_evaluator.rescale_to_next(tt_encrypted11[0], tt_encryptedRes);
                    
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.square_inplace(tt_encrypted11[i]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted11[i], rlk);
                        xehe_evaluator.rescale_to_next(tt_encrypted11[i], tt_encryptedRes);
                    }
                    //tt_encrypted11[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "SqrRelinRescale(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKS EncryptSquareRelinRescaleDecrypt", duration_count/time_loop, time_loop);
                }
            }
        }
    

        {
            std::cout << "CKKS EncryptModSwitchDecrypt" << std::endl;

            seal::EncryptionParameters parms(seal::scheme_type::ckks);

            // Modulus switching without rescaling for random vectors
            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, true, seal::sec_level_type::none);
            //auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());
            seal::Evaluator evaluator(context);

            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);


            int data_bound = 1 << data_bound_bits;
            double delta = static_cast<double>(1ULL << delta_bits);
            srand(static_cast<unsigned>(time(NULL)));

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> xe_output(slot_size);

            seal::Ciphertext encrypted;
            seal::Plaintext plain;
            seal::Plaintext plainRes;
            seal::Plaintext xe_plainRes;
            seal::Ciphertext t_encrypted;

            bool success = true;
            for (int round = 0; round < outer_loop && success; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                }


                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);


                seal::Ciphertext xe_encrypted(encrypted);

                xe_encrypted.gpu();
                if (benchmark)
                {
                    // gpu bit passed to assignee
                    t_encrypted = xe_encrypted;
                    break;
                }

                seal::Ciphertext xe_destination;

                // Check correctness of encryption
                //ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                // Not inplace
                seal::Ciphertext destination;
                evaluator.mod_switch_to_next(encrypted, destination);

                // Check correctness of modulus switching
                //ASSERT_TRUE(destination.parms_id() == next_parms_id);

                decryptor.decrypt(destination, plainRes);
                encoder.decode(plainRes, output);

                //std::cout << "Scale fresh: " << log2(xe_encrypted.scale()) << " bits" << std::endl;
                xehe_evaluator.mod_switch_to_next(xe_encrypted, xe_destination);
                //std::cout << "Scale after mod switch: " << log2(xe_encrypted.scale()) << " bits" << std::endl;
                xehe_decryptor.decrypt(xe_destination, xe_plainRes);
                xehe_encoder.decode(xe_plainRes, xe_output);


                for (size_t i = 0; i < slot_size && success; i++)
                {
                    auto xe_tmp = abs(xe_output[i].real() - output[i].real());
                    auto tmp = abs(input[i].real() - output[i].real());

                    if (xe_tmp >= 0.000001 || tmp >= 0.5)
                    {
                        std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp << std::endl;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(xe_tmp < 0.000001);
                    REQUIRE(tmp < 0.5);
#endif

                }

                // Inplace
                evaluator.mod_switch_to_next_inplace(encrypted);

                // Check correctness of modulus switching
                //ASSERT_TRUE(encrypted.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);


                xehe_evaluator.mod_switch_to_next_inplace(xe_encrypted);
                //std::cout << "Scale after mod switch: " << log2(xe_encrypted.scale()) << " bits" << std::endl;

                decryptor.decrypt(xe_encrypted, xe_plainRes);
                encoder.decode(xe_plainRes, xe_output);

                for (size_t i = 0; i < slot_size; i++)
                {
                    auto xe = xe_output[i].real();
                    auto host = output[i].real();
                    auto xe_tmp = abs(xe - host);
                    auto tmp = abs(input[i].real() - host);

                    if (xe_tmp >= 0.000001 || tmp >= 0.5)
                    {
                        std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp << std::endl;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(xe_tmp < 0.000001);
                    REQUIRE(tmp < 0.5);
#endif
                }
            }

            if (success && !benchmark)
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;


            if (benchmark) {
                std::vector<seal::Ciphertext> tt_encrypted(time_loop + 1, t_encrypted);
                std::vector<seal::Ciphertext> tt_destination(time_loop + 1);

                {
                    xehe_evaluator.mod_switch_to_next(tt_encrypted[0], tt_destination[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.mod_switch_to_next(tt_encrypted[i], tt_destination[i]);
                    }
                    //tt_encrypted[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "SwitchNext(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                vector<complex<double>> inp_zero(slot_size, 0.0);
                seal::Plaintext plain_zero;

                xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                seal::Ciphertext tt_zero;

                xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                std::vector<seal::Ciphertext> tt_encrypted11(time_loop + 1, t_encrypted);
                std::vector<seal::Ciphertext> tt_destination11(time_loop + 1);

                // move to gpu mem by adding zero
                for (int i = 0; i < tt_encrypted11.size(); ++i)
                {
                    xehe_evaluator.add_inplace(tt_encrypted11[i].gpu(), tt_zero);
                }
                //tt_encrypted11[time_loop-1].download();
                {
                    xehe_evaluator.mod_switch_to_next(tt_encrypted11[0], tt_destination11[0]);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.mod_switch_to_next(tt_encrypted11[i], tt_destination11[i]);
                    }
                    //tt_encrypted11[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "SwitchNext(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKS EncryptModSwitchDecrypt", duration_count/time_loop, time_loop);
                }

            }
        }


        {
            std::cout << "CKKS EncryptMultiplyRelinRescaleModSwitchAddDecrypt" << std::endl;

            seal::EncryptionParameters parms(seal::scheme_type::ckks);

            // Multiplication and addition without rescaling for random vectors
            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, true, seal::sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);
            seal::RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            seal::CKKSEncoder encoder(context);
            seal::Encryptor encryptor(context, pk);
            seal::Decryptor decryptor(context, keygen.secret_key());
            seal::Evaluator evaluator(context);

            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);



            seal::Ciphertext encrypted1;
            seal::Ciphertext encrypted2;
            seal::Ciphertext encrypted3;
            seal::Plaintext plain1;
            seal::Plaintext plain2;
            seal::Plaintext plain3;
            seal::Plaintext plainRes;

            seal::Plaintext xe_plainRes;
            seal::Ciphertext t_encrypted1;
            seal::Ciphertext t_encrypted2;
            seal::Ciphertext t_encrypted3;


            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> input3(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            int data_bound = 1 << data_bound_bits;
            double delta = static_cast<double>(1ULL << delta_bits);


            bool success = true;

            for (int round = 0; round < outer_loop && success; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i] + input3[i];
                }

                vector<complex<double>> output(slot_size);

                vector<complex<double>> xe_output(slot_size);

                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);
                encoder.encode(input3, context.first_parms_id(), delta * delta, plain3);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encryptor.encrypt(plain3, encrypted3);

                seal::Ciphertext xe_encrypted1(encrypted1);
                seal::Ciphertext xe_encrypted2(encrypted2);
                seal::Ciphertext xe_encrypted3(encrypted3);
                xe_encrypted1.gpu();
                xe_encrypted3.gpu();

                if (benchmark)
                {
                    // gpu bit passed to assignee
                    t_encrypted1 = xe_encrypted1;
                    t_encrypted2 = xe_encrypted2.gpu();
                    t_encrypted3 = xe_encrypted3;
                    break;
                }


                // Check correctness of encryption
                //ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
               // ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                //ASSERT_TRUE(encrypted3.parms_id() == context.first_parms_id());

                // Enc1*enc2
                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);
                evaluator.rescale_to_next_inplace(encrypted1);

                // Check correctness of modulus switching with rescaling
                //ASSERT_TRUE(encrypted1.parms_id() == next_parms_id);

                // Move enc3 to the level of enc1 * enc2
                evaluator.rescale_to_inplace(encrypted3, next_parms_id);

                // Enc1*enc2 + enc3
                evaluator.add_inplace(encrypted1, encrypted3);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);

                //std::cout << "Scale fresh1: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;

                xehe_evaluator.multiply_inplace(xe_encrypted1, xe_encrypted2);
                //std::cout << "Scale mul1x2: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;

                xehe_evaluator.relinearize_inplace(xe_encrypted1, rlk);
                //std::cout << "Scale relin1: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;
                xehe_evaluator.rescale_to_next_inplace(xe_encrypted1);
                //std::cout << "Scale rescale1: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;

                // Check correctness of modulus switching with rescaling
                //ASSERT_TRUE(encrypted1.parms_id() == next_parms_id);

                // Move enc3 to the level of enc1 * enc2
                //std::cout << "Scale fresh3: " << log2(xe_encrypted3.scale()) << " bits" << std::endl;
                xehe_evaluator.rescale_to_inplace(xe_encrypted3, next_parms_id);
                //std::cout << "Scale after rescale3: " << log2(xe_encrypted3.scale()) << " bits" << std::endl;

                // Enc1*enc2 + enc3
                xehe_evaluator.add_inplace(xe_encrypted1, xe_encrypted3);
                //std::cout << "Scale after add1+3: " << log2(xe_encrypted1.scale()) << " bits" << std::endl;

                //xe_encrypted1.download();

                xehe_decryptor.decrypt(xe_encrypted1, xe_plainRes);
                xehe_encoder.decode(xe_plainRes, xe_output);



                for (size_t i = 0; i < slot_size && success; i++)
                {
                    auto xe_tmp = abs(xe_output[i].real() - output[i].real());
                    auto tmp = abs(expected[i].real() - xe_output[i].real());

                    if (xe_tmp >= 0.000001 || tmp >= 0.5)
                    {
                        std::cout << "failed at round " << round << " slot " << i << " with err " << tmp << " mismatch " << xe_tmp << std::endl;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(xe_tmp < 0.000001);
                    REQUIRE(tmp < 0.5);
#endif
                }
            }

            
            if (success && !benchmark)
            {
                std::cout << "succeed";
            }
            std::cout << std::endl;


            if (benchmark) {
                std::vector<seal::Ciphertext> tt_encrypted1(time_loop + 1, t_encrypted1);
                std::vector<seal::Ciphertext> tt_encrypted2(time_loop + 1, t_encrypted2);
                std::vector<seal::Ciphertext> tt_encrypted3(time_loop + 1, t_encrypted3);

                {
                    xehe_evaluator.multiply_inplace(tt_encrypted1[0], tt_encrypted2[0]);
                    xehe_evaluator.relinearize_inplace(tt_encrypted1[0], rlk);
                    xehe_evaluator.rescale_to_next_inplace(tt_encrypted1[0]);
                    xehe_evaluator.rescale_to_inplace(tt_encrypted3[0], next_parms_id);
                    xehe_evaluator.add_inplace(tt_encrypted1[0], tt_encrypted3[0]);

                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted1[i], tt_encrypted2[i]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted1[i], rlk);
                        xehe_evaluator.rescale_to_next_inplace(tt_encrypted1[i]);
                        xehe_evaluator.rescale_to_inplace(tt_encrypted3[i], next_parms_id);
                        // Enc1*enc2 + enc3
                        xehe_evaluator.add_inplace(tt_encrypted1[i], tt_encrypted3[i]);
                    }
                    //tt_encrypted1[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "MulRelinRescaleModSwitchAdd(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                vector<complex<double>> inp_zero(slot_size, 0.0);
                seal::Plaintext plain_zero;
                seal::Ciphertext tt_zero;
                xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                std::vector<seal::Ciphertext> tt_encrypted11(time_loop + 1, t_encrypted1);
                // zero with double scale
                xehe_encoder.encode(inp_zero, context.first_parms_id(), delta*delta, plain_zero);
                seal::Ciphertext tt_zero2;
                xehe_encryptor.encrypt(plain_zero, tt_zero2.gpu());
                std::vector<seal::Ciphertext> tt_encrypted33(time_loop + 1, t_encrypted3);

                // move to gpu mem by adding zero
                for (int i = 0; i < tt_encrypted11.size(); ++i)
                {
                    xehe_evaluator.add_inplace(tt_encrypted11[i].gpu(), tt_zero);
                    xehe_evaluator.add_inplace(tt_encrypted33[i].gpu(), tt_zero2);
                }
                //tt_encrypted33[time_loop-1].download();

                {
                    xehe_evaluator.multiply_inplace(tt_encrypted11[0], tt_encrypted2[0]);
                    xehe_evaluator.relinearize_inplace(tt_encrypted11[0], rlk);
                    xehe_evaluator.rescale_to_next_inplace(tt_encrypted11[0]);
                    xehe_evaluator.rescale_to_inplace(tt_encrypted33[0], next_parms_id);
                    xehe_evaluator.add_inplace(tt_encrypted11[0], tt_encrypted33[0]);

                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.multiply_inplace(tt_encrypted11[i], tt_encrypted2[i]);
                        xehe_evaluator.relinearize_inplace(tt_encrypted11[i], rlk);
                        xehe_evaluator.rescale_to_next_inplace(tt_encrypted11[i]);
                        xehe_evaluator.rescale_to_inplace(tt_encrypted33[i], next_parms_id);
                        // Enc1*enc2 + enc3
                        xehe_evaluator.add_inplace(tt_encrypted11[i], tt_encrypted33[i]);
                    }
                    //tt_encrypted11[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "MulRelinRescaleModSwitchAdd(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKS EncryptMultiplyRelinRescaleModSwitchAddDecrypt", duration_count/time_loop, time_loop);
                }


            }
        }


        {
            std::cout << "CKKS EncryptRotateDecrypt" << std::endl;


            seal::EncryptionParameters parms(seal::scheme_type::ckks);
            size_t slot_size = n;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::KeyGenerator keygen(context);
            seal::PublicKey pk;
            keygen.create_public_key(pk);
            seal::GaloisKeys glk;
            keygen.create_galois_keys(glk);

            seal::Encryptor encryptor(context, pk);
            seal::Evaluator evaluator(context);
            seal::Decryptor decryptor(context, keygen.secret_key());
            seal::CKKSEncoder encoder(context);


            seal::CKKSEncoder xehe_encoder(context);
            seal::Encryptor xehe_encryptor(context, pk);
            seal::Decryptor xehe_decryptor(context, keygen.secret_key());
            seal::Evaluator xehe_evaluator(context);


            const double delta = static_cast<double>(1ULL << delta_bits);

            seal::Ciphertext encrypted;
            seal::Plaintext plain;

            seal::Plaintext xe_plain;

            seal::Ciphertext t_encrypted;

            vector<complex<double>> input{ complex<double>(1, 1), complex<double>(2, 2), complex<double>(3, 3),
                                           complex<double>(4, 4) };
            input.resize(slot_size);

            vector<complex<double>> output(slot_size, 0);
            vector<complex<double>> xe_output(slot_size, 0);

            encoder.encode(input, context.first_parms_id(), delta, plain);
            int shift = 1;
            encryptor.encrypt(plain, encrypted);

            seal::Ciphertext xe_encrypted1(encrypted);

            xe_encrypted1.gpu();

            seal::Ciphertext t_encrypted1(xe_encrypted1);

            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);

            xehe_evaluator.rotate_vector_inplace(xe_encrypted1, shift, glk);
            xehe_decryptor.decrypt(xe_encrypted1, xe_plain);
            xehe_encoder.decode(xe_plain, xe_output);

            bool success = true;
            if (!benchmark)
            {

                for (size_t i = 0; i < input.size() && success; i++)
                {

                    if ((round(xe_output[i].real()) != round(output[i].real()))
                        ||
                        (round(xe_output[i].imag()) != round(output[i].imag())))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }

                    if ((round(input[(i + static_cast<size_t>(shift)) % slot_size].real()) != round(output[i].real()))
                        ||
                        (round(input[(i + static_cast<size_t>(shift)) % slot_size].imag())) != round(output[i].imag()))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }

#ifndef WIN32
                    REQUIRE(round(xe_output[i].real()) == round(output[i].real()));
                    REQUIRE(round(xe_output[i].imag()) == round(output[i].imag()));
                    REQUIRE(round((input[(i + static_cast<size_t>(shift)) % slot_size].real()) == round(output[i].real())));
                    REQUIRE(round((input[(i + static_cast<size_t>(shift)) % slot_size].imag()) == round(output[i].imag())));
#endif
                }
            }


            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 2;
            encryptor.encrypt(plain, encrypted);

            seal::Ciphertext xe_encrypted2(encrypted);

            xe_encrypted2.gpu();
            seal::Ciphertext t_encrypted2(xe_encrypted2);

            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);

            xehe_evaluator.rotate_vector_inplace(xe_encrypted2, shift, glk);
            xehe_decryptor.decrypt(xe_encrypted2, xe_plain);
            xehe_encoder.decode(xe_plain, xe_output);


            if (!benchmark)
            {
                for (size_t i = 0; i < slot_size && success; i++)
                {

                    if ((round(xe_output[i].real()) != round(output[i].real()))
                        ||
                        (round(xe_output[i].imag()) != round(output[i].imag())))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }

                    if ((round(input[(i + static_cast<size_t>(shift)) % slot_size].real()) != round(output[i].real()))
                        ||
                        (round(input[(i + static_cast<size_t>(shift)) % slot_size].imag())) != round(output[i].imag()))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(round(xe_output[i].real()) == round(output[i].real()));
                    REQUIRE(round(xe_output[i].imag()) == round(output[i].imag()));

                    REQUIRE(round((input[(i + static_cast<size_t>(shift)) % slot_size].real()) == round(output[i].real())));
                    REQUIRE(round((input[(i + static_cast<size_t>(shift)) % slot_size].imag()) == round(output[i].imag())));
#endif
                }
            }


            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 3;
            encryptor.encrypt(plain, encrypted);

            seal::Ciphertext xe_encrypted3(encrypted);

            xe_encrypted3.gpu();
            seal::Ciphertext t_encrypted3(xe_encrypted3);


            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);

            xehe_evaluator.rotate_vector_inplace(xe_encrypted3, shift, glk);
            xehe_decryptor.decrypt(xe_encrypted3, xe_plain);
            xehe_encoder.decode(xe_plain, xe_output);

            if (!benchmark)
            {
                for (size_t i = 0; i < slot_size && success; i++)
                {


                    if ((round(xe_output[i].real()) != round(output[i].real()))
                        ||
                        (round(xe_output[i].imag()) != round(output[i].imag())))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }

                    if ((round(input[(i + static_cast<size_t>(shift)) % slot_size].real()) != round(output[i].real()))
                        ||
                        (round(input[(i + static_cast<size_t>(shift)) % slot_size].imag())) != round(output[i].imag()))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }
#ifndef WIN32
                    REQUIRE(round(xe_output[i].real()) == round(output[i].real()));
                    REQUIRE(round(xe_output[i].imag()) == round(output[i].imag()));

                    REQUIRE(round((input[(i + static_cast<size_t>(shift)) % slot_size].real()) == round(output[i].real())));
                    REQUIRE(round((input[(i + static_cast<size_t>(shift)) % slot_size].imag()) == round(output[i].imag())));
#endif
                }
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            encryptor.encrypt(plain, encrypted);

            seal::Ciphertext xe_encrypted4(encrypted);

            xe_encrypted4.gpu();
            seal::Ciphertext t_encrypted4(xe_encrypted4);


            evaluator.complex_conjugate_inplace(encrypted, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);

            xehe_evaluator.complex_conjugate_inplace(xe_encrypted4, glk);
            xehe_decryptor.decrypt(xe_encrypted4, xe_plain);
            xehe_encoder.decode(xe_plain, xe_output);

            if (!benchmark)
            {
                for (size_t i = 0; i < slot_size && success; i++)
                {

                    if ((round(xe_output[i].real()) != round(output[i].real()))
                        ||
                        (round(xe_output[i].imag()) != round(output[i].imag())))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }

                    if ((round(input[i].real()) != round(output[i].real()))
                        ||
                        (round(-input[i].imag()) != round(output[i].imag())))
                    {
                        std::cout << "failed at slot " << i << std::endl;
                        success = false;
                    }

#ifndef WIN32

                    REQUIRE(round(xe_output[i].real()) == round(output[i].real()));
                    REQUIRE(round(xe_output[i].imag()) == round(output[i].imag()));

                    REQUIRE((round(input[i].real()) == round(output[i].real())));
                    REQUIRE((round(-input[i].imag()) == round(output[i].imag())));
#endif

                }

                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;
            }

            if (benchmark)
            {
                std::vector<seal::Ciphertext> tt_encrypted4(time_loop + 1, t_encrypted4);
                int shift = 4;
                {
                    xehe_evaluator.rotate_vector_inplace(tt_encrypted4[0].gpu(), shift, glk);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.rotate_vector_inplace(tt_encrypted4[i].gpu(), shift, glk);
                    }
                    // tt_encrypted4[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Rotate(noprefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                }
                vector<complex<double>> inp_zero(slot_size, 0.0);
                seal::Plaintext plain_zero;
                seal::Ciphertext tt_zero;
                xehe_encoder.encode(inp_zero, context.first_parms_id(), delta, plain_zero);
                xehe_encryptor.encrypt(plain_zero, tt_zero.gpu());
                std::vector<seal::Ciphertext> tt_encrypted44(time_loop + 1, t_encrypted4);
                // move to gpu mem by adding zero
                for (int i = 0; i < tt_encrypted44.size(); ++i)
                {
                    xehe_evaluator.add_inplace(tt_encrypted44[i].gpu(), tt_zero);
                }
                //tt_encrypted44[time_loop-1].download();
                {
                    xehe_evaluator.rotate_vector_inplace(tt_encrypted44[0], shift, glk);
                    xehe::ext::clear_events();
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 1; i < time_loop + 1; ++i)
                    {
                        xehe_evaluator.rotate_vector_inplace(tt_encrypted44[i], shift, glk);
                    }
                    //tt_encrypted44[time_loop-1].download();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto duration_count = duration.count();
                    std::cout << "Rotate(prefetch): " << to_string(duration_count / time_loop) << "us" << std::endl;
                    xehe::ext::process_events();
                    xehe::ext::add_operation("CKKSEncryptRotateDecrypt", duration_count/time_loop, time_loop);
                }
            }
        }


#endif //#if !_RELIN_ONLY_

#endif //#if !_MUL_ADD_ONLY_


        {
            std::cout << "NTT & inverse NTT correctness check" << std::endl;
            seal::EncryptionParameters parms(seal::scheme_type::ckks);
            {
                // Adding two random vectors 100 times
                size_t slot_size = n;
                parms.set_poly_modulus_degree(slot_size * 2);
                parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, {28, 28, 28, 28, 28, 28}));

                seal::SEALContext context(parms, false, seal::sec_level_type::none);
                seal::KeyGenerator keygen(context);
                seal::PublicKey pk;
                keygen.create_public_key(pk);

                seal::CKKSEncoder encoder(context);
                seal::Encryptor encryptor(context, pk);
                seal::Decryptor decryptor(context, keygen.secret_key());
                seal::Evaluator evaluator(context);

                seal::Ciphertext encrypted;
                seal::Plaintext plain;
                seal::Plaintext plainRes;

                vector<complex<double>> input(slot_size, 0.0);
                vector<complex<double>> expected(slot_size, 0.0);
                vector<complex<double>> output(slot_size);

                int data_bound = (1 << data_bound_bits);
                const double delta = static_cast<double>(1ULL << delta_bits);

                srand(static_cast<unsigned>(time(NULL)));

                bool success = true;
                for (int expCount = 0; expCount < outer_loop && success; expCount++)
                {
                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input[i];
                    }

                    encoder.encode(input, context.first_parms_id(), delta, plain);

                    encryptor.encrypt(plain, encrypted);
                    encrypted.gpu();

                    evaluator.invntt_inplace(encrypted);
                    evaluator.ntt_inplace(encrypted);

                    decryptor.decrypt(encrypted, plainRes);
                    encoder.decode(plainRes, output);

                    for (size_t i = 0; i < slot_size && success; i++)
                    {
                        auto tmp = abs(expected[i].real() - output[i].real());
                        if (tmp >= 0.5)
                        {
                            std::cout << "failed at slot " << i << std::endl;
                            success = false;
                        }
#ifndef WIN32
                        REQUIRE(tmp < 0.5);
#endif
                    }
                }
                if (success && !benchmark)
                {
                    std::cout << "succeed";
                }
                std::cout << std::endl;
            }
        }

        if (benchmark){
            {
                std::cout << "NTT benchmarking" << std::endl;
                seal::EncryptionParameters parms(seal::scheme_type::ckks);
                {
                    // Adding two random vectors 100 times
                    size_t slot_size = n;
                    parms.set_poly_modulus_degree(slot_size * 2);
                    parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

                    seal::SEALContext context(parms, false, seal::sec_level_type::none);
                    seal::KeyGenerator keygen(context);
                    seal::PublicKey pk;
                    keygen.create_public_key(pk);

                    seal::CKKSEncoder encoder(context);
                    seal::Encryptor encryptor(context, pk);
                    seal::Decryptor decryptor(context, keygen.secret_key());
                    seal::Evaluator evaluator(context);

                    seal::Ciphertext encrypted;
                    seal::Ciphertext t_encrypted;
                    seal::Plaintext plain;
                    seal::Plaintext plainRes;

                    vector<complex<double>> input(slot_size, 0.0);
                    vector<complex<double>> expected(slot_size, 0.0);
                    vector<complex<double>> output(slot_size);

                    int data_bound = (1 << data_bound_bits);
                    const double delta = static_cast<double>(1ULL << delta_bits);

                    srand(static_cast<unsigned>(time(NULL)));

                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input[i];
                    }

                    encoder.encode(input, context.first_parms_id(), delta, plain);

                    encryptor.encrypt(plain, encrypted);
                    encrypted.gpu();
                    t_encrypted = encrypted;

                    std::vector<seal::Ciphertext> tt_encrypted(time_loop + 1, t_encrypted);
                    {
                        evaluator.ntt_inplace(tt_encrypted[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();
                        for (int i = 1; i < time_loop + 1; ++i){
                            evaluator.ntt_inplace(tt_encrypted[i]);
                        }
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "forward NTT(noprefetch): " << to_string(duration_count/ time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }
                    {                      
                        evaluator.ntt_inplace(tt_encrypted[0]);                    
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();
                        for (int i = 1; i < time_loop + 1; ++i){
                        
                            evaluator.ntt_inplace(tt_encrypted[i]);
                        };
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "forward NTT(prefetch): " << to_string(duration_count/time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("Forward NTT", duration_count/time_loop, time_loop);
                    }

                }
            }

            {
                std::cout << "Inverse NTT benchmarking" << std::endl;
                seal::EncryptionParameters parms(seal::scheme_type::ckks);
                {
                    // Adding two random vectors 100 times
                    size_t slot_size = n;
                    parms.set_poly_modulus_degree(slot_size * 2);
                    parms.set_coeff_modulus(seal::CoeffModulus::Create(slot_size * 2, mods));

                    seal::SEALContext context(parms, false, seal::sec_level_type::none);
                    seal::KeyGenerator keygen(context);
                    seal::PublicKey pk;
                    keygen.create_public_key(pk);

                    seal::CKKSEncoder encoder(context);
                    seal::Encryptor encryptor(context, pk);
                    seal::Decryptor decryptor(context, keygen.secret_key());
                    seal::Evaluator evaluator(context);

                    seal::Ciphertext encrypted;
                    seal::Ciphertext t_encrypted;
                    seal::Plaintext plain;
                    seal::Plaintext plainRes;

                    vector<complex<double>> input(slot_size, 0.0);
                    vector<complex<double>> expected(slot_size, 0.0);
                    vector<complex<double>> output(slot_size);

                    int data_bound = (1 << data_bound_bits);
                    const double delta = static_cast<double>(1ULL << delta_bits);

                    srand(static_cast<unsigned>(time(NULL)));

                    for (size_t i = 0; i < slot_size; i++)
                    {
                        input[i] = static_cast<double>(rand() % data_bound);
                        expected[i] = input[i];
                    }

                    encoder.encode(input, context.first_parms_id(), delta, plain);

                    encryptor.encrypt(plain, encrypted);
                    encrypted.gpu();
                    t_encrypted = encrypted;

                    std::vector<seal::Ciphertext> tt_encrypted(time_loop + 1, t_encrypted);
                    {
                        evaluator.invntt_inplace(tt_encrypted[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();
                        for (int i = 1; i < time_loop + 1; ++i){
                            evaluator.invntt_inplace(tt_encrypted[i]);
                        }
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "inverse NTT(noprefetch): " << to_string(duration_count/ time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                    }
                    {
                        evaluator.invntt_inplace(tt_encrypted[0]);
                        xehe::ext::clear_events();
                        auto start = std::chrono::high_resolution_clock::now();
                        for (int i = 1; i < time_loop + 1; ++i)
                        {
                            evaluator.invntt_inplace(tt_encrypted[i]);
                        };
                        //tt_encrypted1[time_loop-1].download();
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                        auto duration_count = duration.count();
                        std::cout << "inverse NTT(prefetch): " << to_string(duration_count/time_loop) << "us" << std::endl;
                        xehe::ext::process_events();
                        xehe::ext::add_operation("Inverse NTT", duration_count/time_loop, time_loop);
                    }

                }
            }

        }

    }


#ifndef WIN32

    size_t rns_base_sz64 = 8;
    int prime_sz64 = sizeof(uint64_t)*8 -4;
    int prime_sz32 = sizeof(uint32_t)*8 -4;  
    auto rns_base_sz32 =  size_t(ceil(float(prime_sz64*rns_base_sz64)/prime_sz32));              
    std::vector<int> primes64(rns_base_sz64, prime_sz64);    
    std::vector<int> primes32(rns_base_sz32, prime_sz32);    
    TEST_CASE("Evaluator8K Test", "[gpu][Evaluator]")
    {
        XeTests_Evaluator<uint64_t>(false, 1024*8, 10, 50, 10, 1, &primes64);
    }

    TEST_CASE("Evaluator8K32bit Test", "[gpu][Evaluator]")
    {
        XeTests_Evaluator<uint64_t>(false, 1024*8, 10, 50, 10, 1, &primes32);
    }

    TEST_CASE("Evaluator16K Test", "[gpu][Evaluator]")
    {
        XeTests_Evaluator<uint64_t>(false, 1024*16, 10, 50, 10, 1, &primes64);
    }


    TEST_CASE("Evaluator8K-64 Test", "[gpu][Evaluator]")
    {
        size_t rns_base_sz64 = 64;
        int prime_sz64 = sizeof(uint64_t)*8 -4;
        std::vector<int> primes64(rns_base_sz64, prime_sz64);    
        XeTests_Evaluator<uint64_t>(false, 1024*8, 10, 50, 10, 1, &primes64);
    }    

#if 0
    TEST_CASE("Evaluator32K Test", "[gpu][Evaluator]")
    {
        XeTests_Evaluator<uint64_t>(false, 1024*32, 10, 50, 10);
    }

    TEST_CASE("Evaluator64K Test", "[gpu][Evaluator]")
    {
        XeTests_Evaluator<uint64_t>(false, 1024*64, 10, 50, 10);
    }
 #endif

    TEST_CASE("Evaluator Benchmark8K Test", "[gpu][EvaluatorBenchmark]")
    {
        xehe::ext::clear_export_table();
        XeTests_Evaluator<uint64_t>(true, 1024*8, 10, 50, 10, 20, &primes64);
        xehe::ext::export_table();
    }

    TEST_CASE("Evaluator Benchmark8K-64 Test", "[gpu][Evaluator]")
    {
        size_t rns_base_sz64 = 64;
        int prime_sz64 = sizeof(uint64_t)*8 -4;
        std::vector<int> primes64(rns_base_sz64, prime_sz64);    
        XeTests_Evaluator<uint64_t>(true, 1024*8, 10, 50, 10, 20, &primes64);
    }    

    TEST_CASE("Evaluator Benchmark16K Test", "[gpu][EvaluatorBenchmark]")
    {
        xehe::ext::clear_export_table();
        XeTests_Evaluator<uint64_t>(true, 1024*16, 10, 50, 10, 20, &primes64);
        xehe::ext::export_table();
    }
#if 0
    TEST_CASE("Evaluator Benchmark32K Test", "[gpu][EvaluatorBenchmark]")
    {
        XeTests_Evaluator<uint64_t>(true, 1024*32, 10, 50);
    }

    TEST_CASE("Evaluator Benchmark64K Test", "[gpu][EvaluatorBenchmark]")
    {
        XeTests_Evaluator<uint64_t>(true, 1024*64, 10, 50);
    }

    TEST_CASE("Evaluator Benchmark8K32bit Test", "[gpu][Evaluator]")
    {

        XeTests_Evaluator<uint64_t>(true, 1024*8, 10, 50, 10, 10, &primes32);
    }
#endif

#endif // #ifndef WIN32
} // namespace xehetest


