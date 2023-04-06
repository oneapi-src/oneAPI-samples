// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#ifdef SEAL_USE_INTEL_XEHE
#include "seal/batchencoder.h"
#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/keygenerator.h"
#include "seal/modulus.h"
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <string>
#include "gtest/gtest.h"

using namespace seal;
using namespace std;

namespace sealtest
{
    TEST(GPUEvaluatorTest, CKKSEncryptAddDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Adding two zero vectors
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 30, 30, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted(false);
            Plaintext plain;
            Plaintext plainRes;
            Plaintext plainRes_gpu;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> output_gpu(slot_size);
            const double delta = static_cast<double>(1 << 16);
            encoder.encode(input, context.first_parms_id(), delta, plain);

            encryptor.encrypt(plain, encrypted);
            Ciphertext encrypted_gpu(encrypted);

            // CPU
            evaluator.add_inplace(encrypted, encrypted);

            // Check correctness of encryption
            ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

            decryptor.decrypt(encrypted, plainRes);
            encoder.decode(plainRes, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                auto tmp = abs(input[i].real() - output[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }

            // GPU
            encrypted_gpu.gpu();
            evaluator.add_inplace(encrypted_gpu, encrypted_gpu);

            // Check correctness of encryption
            ASSERT_TRUE(encrypted_gpu.parms_id() == context.first_parms_id());

            decryptor.decrypt(encrypted_gpu, plainRes_gpu);
            encoder.decode(plainRes_gpu, output_gpu);
            for (size_t i = 0; i < slot_size; i++)
            {
                auto tmp = abs(input[i].real() - output_gpu[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }

            for (size_t i = 0; i < slot_size; i++)
            {
                auto tmp = abs(output[i].real() - output_gpu[i].real());
                ASSERT_TRUE(tmp < 0.0001);
            }
        }

        {
            // Adding two random vectors 100 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1(false);
            Ciphertext encrypted2(false);
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;
            Plaintext plainRes_gpu;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> output_gpu(slot_size);

            int data_bound = (1 << 30);
            const double delta = static_cast<double>(1 << 16);

            srand(static_cast<unsigned>(time(NULL)));

           for (int expCount = 0; expCount < 100; expCount++)
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
                Ciphertext encrypted1_gpu(encrypted1);
                Ciphertext encrypted2_gpu(encrypted2);

                //CPU
                evaluator.add_inplace(encrypted1, encrypted2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }

                //GPU
                encrypted1_gpu.gpu();
                evaluator.add_inplace(encrypted1_gpu, encrypted2_gpu);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1_gpu.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1_gpu, plainRes_gpu);
                encoder.decode(plainRes_gpu, output_gpu);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output_gpu[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }


                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(output[i].real() - output_gpu[i].real());
                    ASSERT_TRUE(tmp < 0.0001);
                }
            }
        }

        {
            // Adding two random vectors 100 times
            size_t slot_size = 8;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1(false);
            Ciphertext encrypted2(false);
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;
            Plaintext plainRes_gpu;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> output_gpu(slot_size);

            int data_bound = (1 << 30);
            const double delta = static_cast<double>(1 << 16);

            srand(static_cast<unsigned>(time(NULL)));

            for (int expCount = 0; expCount < 100; expCount++)
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
                Ciphertext encrypted1_gpu(encrypted1);
                Ciphertext encrypted2_gpu(encrypted2);

                //CPU
                evaluator.add_inplace(encrypted1, encrypted2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }

                //GPU
                encrypted1_gpu.gpu();
                evaluator.add_inplace(encrypted1_gpu, encrypted2_gpu);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1_gpu.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1_gpu, plainRes_gpu);
                encoder.decode(plainRes_gpu, output_gpu);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output_gpu[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }

                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(output[i].real() - output_gpu[i].real());
                    ASSERT_TRUE(tmp < 0.0001);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, XeHE_NTTs)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Adding two random vectors 100 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 30);
            const double delta = static_cast<double>(1 << 16);

            srand(static_cast<unsigned>(time(NULL)));

           for (int expCount = 0; expCount < 100; expCount++)
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
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }

            }
        }

    }


#if 0
    TEST(GPUEvaluatorTest, CKKSEncryptAddPlainDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        
        {
            // Adding two zero vectors
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 30, 30, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            const double delta = static_cast<double>(1 << 16);
            encoder.encode(input, context.first_parms_id(), delta, plain);

            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.add_plain_inplace(encrypted, plain);

            // Check correctness of encryption
            ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

            decryptor.decrypt(encrypted, plainRes);
            encoder.decode(plainRes, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                auto tmp = abs(input[i].real() - output[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            // Adding two random vectors 50 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 8);
            const double delta = static_cast<double>(1ULL << 16);

            srand(static_cast<unsigned>(time(NULL)));

           for (int expCount = 0; expCount < 50; expCount++)
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
                encrypted1.gpu();
                evaluator.add_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Adding two random vectors 50 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            double input2;
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 8);
            const double delta = static_cast<double>(1ULL << 16);

            srand(static_cast<unsigned>(time(NULL)));

            for (int expCount = 0; expCount < 50; expCount++)
            {
                input2 = static_cast<double>(rand() % (data_bound * data_bound)) / data_bound;
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] + input2;
                }

                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encrypted1.gpu();
                evaluator.add_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        {
            // Adding two random vectors 50 times
            size_t slot_size = 8;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            double input2;
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 8);
            const double delta = static_cast<double>(1ULL << 16);

            srand(static_cast<unsigned>(time(NULL)));

            for (int expCount = 0; expCount < 50; expCount++)
            {
                input2 = static_cast<double>(rand() % (data_bound * data_bound)) / data_bound;
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] + input2;
                }

                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encrypted1.gpu();
                evaluator.add_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptSubPlainDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        
        {
            // Subtracting two zero vectors
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 30, 30, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            const double delta = static_cast<double>(1 << 16);
            encoder.encode(input, context.first_parms_id(), delta, plain);

            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.add_plain_inplace(encrypted, plain);

            // Check correctness of encryption
            ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

            decryptor.decrypt(encrypted, plainRes);
            encoder.decode(plainRes, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                auto tmp = abs(input[i].real() - output[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            // Subtracting two random vectors 100 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 8);
            const double delta = static_cast<double>(1ULL << 16);

            srand(static_cast<unsigned>(time(NULL)));

            for (int expCount = 0; expCount < 100; expCount++)
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
                encrypted1.gpu();
                evaluator.sub_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Subtracting two random vectors 100 times
            size_t slot_size = 8;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 8);
            const double delta = static_cast<double>(1ULL << 16);

            srand(static_cast<unsigned>(time(NULL)));

            for (int expCount = 0; expCount < 100; expCount++)
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
                encrypted1.gpu();
                evaluator.sub_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

#include "seal/randomgen.h"
    TEST(GPUEvaluatorTest, CKKSEncryptNaiveMultiplyDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        
        {
            // Multiplying two zero vectors
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 30, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            const double delta = static_cast<double>(1 << 30);
            encoder.encode(input, context.first_parms_id(), delta, plain);

            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.multiply_inplace(encrypted, encrypted);

            // Check correctness of encryption
            ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

            decryptor.decrypt(encrypted, plainRes);
            encoder.decode(plainRes, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                auto tmp = abs(input[i].real() - output[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            // Multiplying two random vectors
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            const double delta = static_cast<double>(1ULL << 40);

            int data_bound = (1 << 10);
            srand(static_cast<unsigned>(time(NULL)));

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();
                evaluator.multiply_inplace(encrypted1, encrypted2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Multiplying two random vectors
            size_t slot_size = 16;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            const double delta = static_cast<double>(1ULL << 40);

            int data_bound = (1 << 10);
            srand(static_cast<unsigned>(time(NULL)));

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encrypted1.gpu();
                encryptor.encrypt(plain2, encrypted2);
                evaluator.multiply_inplace(encrypted1, encrypted2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptMultiplyByNumberDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Multiplying two random vectors by an integer
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 40 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            int64_t input2;
            vector<complex<double>> expected(slot_size, 0.0);

            int data_bound = (1 << 10);
            srand(static_cast<unsigned>(time(NULL)));

            for (int iExp = 0; iExp < 50; iExp++)
            {
                input2 = max(rand() % data_bound, 1);
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * static_cast<double>(input2);
                }

                vector<complex<double>> output(slot_size);
                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), plain2);

                encryptor.encrypt(plain1, encrypted1);
                encrypted1.gpu();
                evaluator.multiply_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Multiplying two random vectors by an integer
            size_t slot_size = 8;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            int64_t input2;
            vector<complex<double>> expected(slot_size, 0.0);

            int data_bound = (1 << 10);
            srand(static_cast<unsigned>(time(NULL)));

            for (int iExp = 0; iExp < 50; iExp++)
            {
                input2 = max(rand() % data_bound, 1);
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * static_cast<double>(input2);
                }

                vector<complex<double>> output(slot_size);
                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), plain2);

                encryptor.encrypt(plain1, encrypted1);
                encrypted1.gpu();
                evaluator.multiply_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        {
            // Multiplying two random vectors by a double
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            double input2;
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 10);
            srand(static_cast<unsigned>(time(NULL)));

            for (int iExp = 0; iExp < 50; iExp++)
            {
                input2 = static_cast<double>(rand() % (data_bound * data_bound)) / static_cast<double>(data_bound);
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2;
                }

                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encrypted1.gpu();
                evaluator.multiply_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Multiplying two random vectors by a double
            size_t slot_size = 16;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 2.1);
            double input2;
            vector<complex<double>> expected(slot_size, 2.1);
            vector<complex<double>> output(slot_size);

            int data_bound = (1 << 10);
            srand(static_cast<unsigned>(time(NULL)));

            for (int iExp = 0; iExp < 50; iExp++)
            {
                input2 = static_cast<double>(rand() % (data_bound * data_bound)) / static_cast<double>(data_bound);
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2;
                }

                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encrypted1.gpu();
                evaluator.multiply_plain_inplace(encrypted1, plain2);

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptMultiplyRelinDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Multiplying two random vectors 50 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encryptedRes;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            int data_bound = 1 << 10;

            for (int round = 0; round < 50; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }

                vector<complex<double>> output(slot_size);
                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Multiplying two random vectors 50 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encryptedRes;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            int data_bound = 1 << 10;

            for (int round = 0; round < 50; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }

                vector<complex<double>> output(slot_size);
                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        {
            // Multiplying two random vectors 50 times
            size_t slot_size = 2;
            parms.set_poly_modulus_degree(8);
            parms.set_coeff_modulus(CoeffModulus::Create(8, { 60, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encryptedRes;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            int data_bound = 1 << 10;
            const double delta = static_cast<double>(1ULL << 40);

            for (int round = 0; round < 50; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }

                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                evaluator.multiply_inplace(encrypted1, encrypted2);
                // Evaluator.relinearize_inplace(encrypted1, rlk);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptSquareRelinDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Squaring two random vectors 100 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);

            int data_bound = 1 << 7;
            srand(static_cast<unsigned>(time(NULL)));

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input[i] * input[i];
                }

                vector<complex<double>> output(slot_size);
                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                // Evaluator.square_inplace(encrypted);
                evaluator.multiply_inplace(encrypted, encrypted);
                evaluator.relinearize_inplace(encrypted, rlk);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Squaring two random vectors 100 times
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);

            int data_bound = 1 << 7;
            srand(static_cast<unsigned>(time(NULL)));

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input[i] * input[i];
                }

                vector<complex<double>> output(slot_size);
                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                // Evaluator.square_inplace(encrypted);
                evaluator.multiply_inplace(encrypted, encrypted);
                evaluator.relinearize_inplace(encrypted, rlk);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        {
            // Squaring two random vectors 100 times
            size_t slot_size = 16;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 60, 30, 30, 30 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);

            int data_bound = 1 << 7;
            srand(static_cast<unsigned>(time(NULL)));

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input[i] * input[i];
                }

                vector<complex<double>> output(slot_size);
                const double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                // Evaluator.square_inplace(encrypted);
                evaluator.multiply_inplace(encrypted, encrypted);
                evaluator.relinearize_inplace(encrypted, rlk);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptMultiplyRelinRescaleDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Multiplying two random vectors 100 times
            size_t slot_size = 64;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 30, 30, 30, 30, 30, 30 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encryptedRes;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);

            for (int round = 0; round < 100; round++)
            {
                int data_bound = 1 << 7;
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }

                vector<complex<double>> output(slot_size);
                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);
                evaluator.rescale_to_next_inplace(encrypted1);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted1.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Multiplying two random vectors 100 times
            size_t slot_size = 16;
            parms.set_poly_modulus_degree(128);
            parms.set_coeff_modulus(CoeffModulus::Create(128, { 30, 30, 30, 30, 30 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encryptedRes;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);

            for (int round = 0; round < 100; round++)
            {
                int data_bound = 1 << 7;
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i];
                }

                vector<complex<double>> output(slot_size);
                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);
                evaluator.rescale_to_next_inplace(encrypted1);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted1.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        {
            // Multiplying two random vectors 100 times
            size_t slot_size = 16;
            parms.set_poly_modulus_degree(128);
            parms.set_coeff_modulus(CoeffModulus::Create(128, { 60, 60, 60, 60, 60 }));

            SEALContext context(parms, true, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encryptedRes;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);

            for (int round = 0; round < 100; round++)
            {
                int data_bound = 1 << 7;
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i] * input2[i];
                }

                vector<complex<double>> output(slot_size);
                double delta = static_cast<double>(1ULL << 60);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);
                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);

                // Scale down by two levels
                auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                evaluator.rescale_to_inplace(encrypted1, target_parms);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted1.parms_id() == target_parms);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }

            // Test with inverted order: rescale then relin
            for (int round = 0; round < 100; round++)
            {
                int data_bound = 1 << 7;
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i] * input2[i];
                }

                vector<complex<double>> output(slot_size);
                double delta = static_cast<double>(1ULL << 50);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());

                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);
                evaluator.multiply_inplace(encrypted1, encrypted2);

                // Scale down by two levels
                auto target_parms = context.first_context_data()->next_context_data()->next_context_data()->parms_id();
                evaluator.rescale_to_inplace(encrypted1, target_parms);

                // Relinearize now
                evaluator.relinearize_inplace(encrypted1, rlk);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted1.parms_id() == target_parms);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptSquareRelinRescaleDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Squaring two random vectors 100 times
            size_t slot_size = 64;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 50, 50, 50 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> expected(slot_size, 0.0);
            int data_bound = 1 << 8;

            for (int round = 0; round < 100; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input[i] * input[i];
                }

                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                evaluator.square_inplace(encrypted);
                evaluator.relinearize_inplace(encrypted, rlk);
                evaluator.rescale_to_next_inplace(encrypted);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Squaring two random vectors 100 times
            size_t slot_size = 16;
            parms.set_poly_modulus_degree(128);
            parms.set_coeff_modulus(CoeffModulus::Create(128, { 50, 50, 50 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);
            vector<complex<double>> expected(slot_size, 0.0);
            int data_bound = 1 << 8;

            for (int round = 0; round < 100; round++)
            {
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input[i] * input[i];
                }

                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                evaluator.square_inplace(encrypted);
                evaluator.relinearize_inplace(encrypted, rlk);
                evaluator.rescale_to_next_inplace(encrypted);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptModSwitchDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Modulus switching without rescaling for random vectors
            size_t slot_size = 64;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 60, 60, 60, 60, 60 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            int data_bound = 1 << 30;
            srand(static_cast<unsigned>(time(NULL)));

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                }

                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                // Not inplace
                Ciphertext destination;
                evaluator.mod_switch_to_next(encrypted, destination);

                // Check correctness of modulus switching
                ASSERT_TRUE(destination.parms_id() == next_parms_id);

                decryptor.decrypt(destination, plainRes);
                encoder.decode(plainRes, output);

                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }

                // Inplace
                evaluator.mod_switch_to_next_inplace(encrypted);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Modulus switching without rescaling for random vectors
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 40, 40, 40, 40, 40 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            int data_bound = 1 << 30;
            srand(static_cast<unsigned>(time(NULL)));

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                }

                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                // Not inplace
                Ciphertext destination;
                evaluator.mod_switch_to_next(encrypted, destination);

                // Check correctness of modulus switching
                ASSERT_TRUE(destination.parms_id() == next_parms_id);

                decryptor.decrypt(destination, plainRes);
                encoder.decode(plainRes, output);

                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }

                // Inplace
                evaluator.mod_switch_to_next_inplace(encrypted);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        {
            // Modulus switching without rescaling for random vectors
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(128);
            parms.set_coeff_modulus(CoeffModulus::Create(128, { 40, 40, 40, 40, 40 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            int data_bound = 1 << 30;
            srand(static_cast<unsigned>(time(NULL)));

            vector<complex<double>> input(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            Ciphertext encrypted;
            Plaintext plain;
            Plaintext plainRes;

            for (int round = 0; round < 100; round++)
            {
                for (size_t i = 0; i < slot_size; i++)
                {
                    input[i] = static_cast<double>(rand() % data_bound);
                }

                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input, context.first_parms_id(), delta, plain);

                encryptor.encrypt(plain, encrypted);
                encrypted.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted.parms_id() == context.first_parms_id());

                // Not inplace
                Ciphertext destination;
                evaluator.mod_switch_to_next(encrypted, destination);

                // Check correctness of modulus switching
                ASSERT_TRUE(destination.parms_id() == next_parms_id);

                decryptor.decrypt(destination, plainRes);
                encoder.decode(plainRes, output);

                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }

                // Inplace
                evaluator.mod_switch_to_next_inplace(encrypted);

                // Check correctness of modulus switching
                ASSERT_TRUE(encrypted.parms_id() == next_parms_id);

                decryptor.decrypt(encrypted, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(input[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptMultiplyRelinRescaleModSwitchAddDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Multiplication and addition without rescaling for random vectors
            size_t slot_size = 64;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 50, 50, 50 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encrypted3;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plain3;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> input3(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);

            for (int round = 0; round < 100; round++)
            {
                int data_bound = 1 << 8;
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i] + input3[i];
                }

                vector<complex<double>> output(slot_size);
                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);
                encoder.encode(input3, context.first_parms_id(), delta * delta, plain3);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encryptor.encrypt(plain3, encrypted3);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted3.parms_id() == context.first_parms_id());

                // Enc1*enc2
                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);
                evaluator.rescale_to_next_inplace(encrypted1);

                // Check correctness of modulus switching with rescaling
                ASSERT_TRUE(encrypted1.parms_id() == next_parms_id);

                // Move enc3 to the level of enc1 * enc2
                evaluator.rescale_to_inplace(encrypted3, next_parms_id);

                // Enc1*enc2 + enc3
                evaluator.add_inplace(encrypted1, encrypted3);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        
        {
            // Multiplication and addition without rescaling for random vectors
            size_t slot_size = 16;
            parms.set_poly_modulus_degree(128);
            parms.set_coeff_modulus(CoeffModulus::Create(128, { 50, 50, 50 }));

            SEALContext context(parms, true, sec_level_type::none);
            auto next_parms_id = context.first_context_data()->next_context_data()->parms_id();
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            RelinKeys rlk;
            keygen.create_relin_keys(rlk);

            CKKSEncoder encoder(context);
            Encryptor encryptor(context, pk);
            Decryptor decryptor(context, keygen.secret_key());
            Evaluator evaluator(context);

            Ciphertext encrypted1;
            Ciphertext encrypted2;
            Ciphertext encrypted3;
            Plaintext plain1;
            Plaintext plain2;
            Plaintext plain3;
            Plaintext plainRes;

            vector<complex<double>> input1(slot_size, 0.0);
            vector<complex<double>> input2(slot_size, 0.0);
            vector<complex<double>> input3(slot_size, 0.0);
            vector<complex<double>> expected(slot_size, 0.0);
            vector<complex<double>> output(slot_size);

            for (int round = 0; round < 100; round++)
            {
                int data_bound = 1 << 8;
                srand(static_cast<unsigned>(time(NULL)));
                for (size_t i = 0; i < slot_size; i++)
                {
                    input1[i] = static_cast<double>(rand() % data_bound);
                    input2[i] = static_cast<double>(rand() % data_bound);
                    expected[i] = input1[i] * input2[i] + input3[i];
                }

                double delta = static_cast<double>(1ULL << 40);
                encoder.encode(input1, context.first_parms_id(), delta, plain1);
                encoder.encode(input2, context.first_parms_id(), delta, plain2);
                encoder.encode(input3, context.first_parms_id(), delta * delta, plain3);

                encryptor.encrypt(plain1, encrypted1);
                encryptor.encrypt(plain2, encrypted2);
                encryptor.encrypt(plain3, encrypted3);
                encrypted1.gpu();

                // Check correctness of encryption
                ASSERT_TRUE(encrypted1.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted2.parms_id() == context.first_parms_id());
                // Check correctness of encryption
                ASSERT_TRUE(encrypted3.parms_id() == context.first_parms_id());

                // Enc1*enc2
                evaluator.multiply_inplace(encrypted1, encrypted2);
                evaluator.relinearize_inplace(encrypted1, rlk);
                evaluator.rescale_to_next_inplace(encrypted1);

                // Check correctness of modulus switching with rescaling
                ASSERT_TRUE(encrypted1.parms_id() == next_parms_id);

                // Move enc3 to the level of enc1 * enc2
                evaluator.rescale_to_inplace(encrypted3, next_parms_id);

                // Enc1*enc2 + enc3
                evaluator.add_inplace(encrypted1, encrypted3);

                decryptor.decrypt(encrypted1, plainRes);
                encoder.decode(plainRes, output);
                for (size_t i = 0; i < slot_size; i++)
                {
                    auto tmp = abs(expected[i].real() - output[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptRotateDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Maximal number of slots
            size_t slot_size = 4;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 40, 40, 40, 40 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            GaloisKeys glk;
            keygen.create_galois_keys(glk);

            Encryptor encryptor(context, pk);
            Evaluator evaluator(context);
            Decryptor decryptor(context, keygen.secret_key());
            CKKSEncoder encoder(context);
            const double delta = static_cast<double>(1ULL << 30);

            Ciphertext encrypted;
            Plaintext plain;

            vector<complex<double>> input{ complex<double>(1, 1), complex<double>(2, 2), complex<double>(3, 3),
                                           complex<double>(4, 4) };
            input.resize(slot_size);

            vector<complex<double>> output(slot_size, 0);

            encoder.encode(input, context.first_parms_id(), delta, plain);
            int shift = 1;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].real(), round(output[i].real()));
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].imag(), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 2;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].real(), round(output[i].real()));
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].imag(), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 3;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].real(), round(output[i].real()));
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].imag(), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.complex_conjugate_inplace(encrypted, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[i].real(), round(output[i].real()));
                ASSERT_EQ(-input[i].imag(), round(output[i].imag()));
            }
        }
        
        {
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 40, 40, 40, 40 }));

            SEALContext context(parms, false, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            GaloisKeys glk;
            keygen.create_galois_keys(glk);

            Encryptor encryptor(context, pk);
            Evaluator evaluator(context);
            Decryptor decryptor(context, keygen.secret_key());
            CKKSEncoder encoder(context);
            const double delta = static_cast<double>(1ULL << 30);

            Ciphertext encrypted;
            Plaintext plain;

            vector<complex<double>> input{ complex<double>(1, 1), complex<double>(2, 2), complex<double>(3, 3),
                                           complex<double>(4, 4) };
            input.resize(slot_size);

            vector<complex<double>> output(slot_size, 0);

            encoder.encode(input, context.first_parms_id(), delta, plain);
            int shift = 1;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < input.size(); i++)
            {
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].real()), round(output[i].real()));
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].imag()), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 2;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].real()), round(output[i].real()));
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].imag()), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 3;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].real()), round(output[i].real()));
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].imag()), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.complex_conjugate_inplace(encrypted, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(round(input[i].real()), round(output[i].real()));
                ASSERT_EQ(round(-input[i].imag()), round(output[i].imag()));
            }
        }
    }

    TEST(GPUEvaluatorTest, CKKSEncryptRescaleRotateDecrypt)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            // Maximal number of slots
            size_t slot_size = 4;
            parms.set_poly_modulus_degree(slot_size * 2);
            parms.set_coeff_modulus(CoeffModulus::Create(slot_size * 2, { 40, 40, 40, 40 }));

            SEALContext context(parms, true, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            GaloisKeys glk;
            keygen.create_galois_keys(glk);

            Encryptor encryptor(context, pk);
            Evaluator evaluator(context);
            Decryptor decryptor(context, keygen.secret_key());
            CKKSEncoder encoder(context);
            const double delta = pow(2.0, 70);

            Ciphertext encrypted;
            Plaintext plain;

            vector<complex<double>> input{ complex<double>(1, 1), complex<double>(2, 2), complex<double>(3, 3),
                                           complex<double>(4, 4) };
            input.resize(slot_size);

            vector<complex<double>> output(slot_size, 0);

            encoder.encode(input, context.first_parms_id(), delta, plain);
            int shift = 1;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].real(), round(output[i].real()));
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].imag(), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 2;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].real(), round(output[i].real()));
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].imag(), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 3;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].real(), round(output[i].real()));
                ASSERT_EQ(input[(i + static_cast<size_t>(shift)) % slot_size].imag(), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.complex_conjugate_inplace(encrypted, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(input[i].real(), round(output[i].real()));
                ASSERT_EQ(-input[i].imag(), round(output[i].imag()));
            }
        }
        
        {
            size_t slot_size = 32;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 40, 40, 40, 40 }));

            SEALContext context(parms, true, sec_level_type::none);
            KeyGenerator keygen(context);
            PublicKey pk;
            keygen.create_public_key(pk);
            GaloisKeys glk;
            keygen.create_galois_keys(glk);

            Encryptor encryptor(context, pk);
            Evaluator evaluator(context);
            Decryptor decryptor(context, keygen.secret_key());
            CKKSEncoder encoder(context);
            const double delta = pow(2, 70);

            Ciphertext encrypted;
            Plaintext plain;

            vector<complex<double>> input{ complex<double>(1, 1), complex<double>(2, 2), complex<double>(3, 3),
                                           complex<double>(4, 4) };
            input.resize(slot_size);

            vector<complex<double>> output(slot_size, 0);

            encoder.encode(input, context.first_parms_id(), delta, plain);
            int shift = 1;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].real()), round(output[i].real()));
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].imag()), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 2;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].real()), round(output[i].real()));
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].imag()), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            shift = 3;
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.rotate_vector_inplace(encrypted, shift, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].real()), round(output[i].real()));
                ASSERT_EQ(round(input[(i + static_cast<size_t>(shift)) % slot_size].imag()), round(output[i].imag()));
            }

            encoder.encode(input, context.first_parms_id(), delta, plain);
            encryptor.encrypt(plain, encrypted);
            encrypted.gpu();
            evaluator.rescale_to_next_inplace(encrypted);
            evaluator.complex_conjugate_inplace(encrypted, glk);
            decryptor.decrypt(encrypted, plain);
            encoder.decode(plain, output);
            for (size_t i = 0; i < slot_size; i++)
            {
                ASSERT_EQ(round(input[i].real()), round(output[i].real()));
                ASSERT_EQ(round(-input[i].imag()), round(output[i].imag()));
            }
        }
    }
#endif

} // namespace sealtest
#endif