// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <seal/seal.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "seal/seal_ckks_context.h"
#include "seal/seal_ckks_kernel_executor.h"
#include "seal/util/test_util.h"
#include "seal/util/timer.h"

namespace intel {
namespace he {
namespace heseal {

std::vector<double> testDotCipherBatchAxis(const std::vector<double>& inputA,
                                           const std::vector<double>& inputB,
                                           size_t dim1, size_t dim2,
                                           size_t dim3) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  size_t batch_size = 1;

  std::vector<seal::Ciphertext> ciphersA =
      context.encryptVector(inputA, batch_size);

  std::vector<seal::Ciphertext> ciphersB =
      context.encryptVector(inputB, batch_size);

    for(auto &v : ciphersA) {
        v.gpu();
    }

  std::vector<seal::Ciphertext> cipher_dot =
      kernel_executor.dotCipherBatchAxis(ciphersA, ciphersB, dim1, dim2, dim3);

  std::vector<seal::Plaintext> decrypted = context.decryptVector(cipher_dot);
  return context.decodeVector(decrypted, batch_size);
}

std::vector<double> testDotPlainBatchAxis(const std::vector<double>& inputA,
                                          const std::vector<double>& inputB,
                                          size_t dim1, size_t dim2,
                                          size_t dim3) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  size_t batch_size = 1;

  std::vector<seal::Ciphertext> ciphersA =
      context.encryptVector(inputA, batch_size);

  std::vector<seal::Plaintext> plainsB =
      context.encodeVector(inputB, batch_size);

    for(auto &v : ciphersA) {
        v.gpu();
    }

  std::vector<seal::Ciphertext> cipher_dot =
      kernel_executor.dotPlainBatchAxis(ciphersA, plainsB, dim1, dim2, dim3);

  std::vector<seal::Plaintext> decrypted = context.decryptVector(cipher_dot);
  return context.decodeVector(decrypted, batch_size);
}

void prepareLinearRegression4x3(double& bias, std::vector<double>& weights,
                                std::vector<std::vector<double>>& inputs,
                                std::vector<double>& ground_truth) {
  // values hand-picked to ensure that sigmoid input remains between -1 and 1
  // to ensure valid domain range for polynomial approximation
  bias = -0.463;
  weights.assign({0.438, -0.18, 0.0, 0.3141592654});
  inputs.resize(3);
  inputs[0].assign({0.7232, -0.3469, 0.7383, 0.6038});
  inputs[1].assign({-0.8509, -0.8242, -0.1463, -0.3124});
  inputs[2].assign({0.6432, 0.0438, 0.9413, 0.2812});
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    ground_truth.emplace_back(std::inner_product(
        inputs[i].begin(), inputs[i].end(), weights.begin(), bias));
  }
}

void prepareLogisticRegression4x3(double& bias, std::vector<double>& weights,
                                  std::vector<std::vector<double>>& inputs,
                                  std::vector<double>& ground_truth) {
  prepareLinearRegression4x3(bias, weights, inputs, ground_truth);
  std::transform(ground_truth.begin(), ground_truth.end(), ground_truth.begin(),
                 approxSigmoid<3>);
}

std::vector<double> testLogisticRegression4x3(
    double bias, const std::vector<double>& weights,
    const std::vector<std::vector<double>>& inputs,
    std::size_t sigmoid_degree) {
  SealCKKSContext context(1UL << 14, {60, 45, 45, 45, 45, 45}, 1UL << 45, true,
                          true);
  SealCKKSKernelExecutor kernel_executor(context);
  seal::Plaintext plain_bias;
  seal::Ciphertext cipher_bias;
  std::vector<seal::Ciphertext> cipher_bias0(1);
  seal::Plaintext zero_plain;
  std::vector<seal::Ciphertext> zero_cipher(1);
  context.encoder().encode(bias, context.scale(), plain_bias);
  context.encryptor().encrypt(plain_bias, cipher_bias0[0]);
  context.encoder().encode(0., context.scale(), zero_plain);
  context.encryptor().encrypt(zero_plain, zero_cipher[0]);
  cipher_bias0 = kernel_executor.add(cipher_bias0, zero_cipher);
  cipher_bias = cipher_bias0[0];
  std::vector<double> vZ(weights.size(), 0);
  std::vector<seal::Ciphertext> cipher_Z =  context.encryptVector(vZ);
  std::vector<seal::Ciphertext> cipher_weights0 = context.encryptVector(weights);
   for(auto &v : cipher_weights0) {
      v.gpu();
    }  
  std::vector<seal::Ciphertext> cipher_weights = kernel_executor.add(cipher_weights0, cipher_Z);
  std::vector<std::vector<seal::Ciphertext>> cipher_inputs(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i)
  {
    auto cipher_input = context.encryptVector(inputs[i]);
    for(auto &v : cipher_input) {
      v.gpu();
    }
    cipher_inputs[i] = kernel_executor.add(cipher_input,cipher_Z);

  }  

  util::Timer t;
  t.start();

  std::vector<seal::Ciphertext> cipher_retval =
      kernel_executor.evaluateLogisticRegression(cipher_weights, cipher_inputs,
                                                 cipher_bias, weights.size(),
                                                 uint32_t(sigmoid_degree));
  std::cout << "LogReg compute time: " << double(t.elapsedMicroseconds())/1000 << " ms\n";

  std::vector<double> retval =
      context.decodeVector(context.decryptVector(cipher_retval));
  retval.resize(inputs.size());
  return retval;
}
void LogisticRegression_CKKS(size_t n_inputs,
                                     size_t n_weights, int n, const std::vector<int> coef_mods) {
  SealCKKSContext context(size_t(n), coef_mods,
                          1UL << 40, true, true);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> weights(n_weights);
  std::vector<std::vector<double>> inputs(n_inputs);

  std::default_random_engine rand_gen;
  std::uniform_real_distribution<double> uniform_rnd(-1.5, 1.5);
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = uniform_rnd(rand_gen);
  for (std::size_t j = 0; j < inputs.size(); ++j) {
    inputs[j].resize(n_weights);
    for (std::size_t i = 0; i < inputs[j].size(); ++i)
      inputs[j][i] = uniform_rnd(rand_gen);
  }
  double bias = uniform_rnd(rand_gen);

  std::vector<seal::Ciphertext> cipher_weights0 =
      context.encryptVector(gsl::span(weights.data(), weights.size()),
                            context.encoder().slot_count());

  seal::Ciphertext cipher_bias;
  seal::Plaintext plain_bias;
  context.encoder().encode(bias, context.scale(), plain_bias);
  context.encryptor().encrypt(plain_bias, cipher_bias);
  seal::Ciphertext zero_cipher;
  seal::Plaintext zero_bias;
  context.encoder().encode(0., context.scale(), zero_bias);
  context.encryptor().encrypt(zero_bias, zero_cipher);
  std::vector<seal::Ciphertext> z_v(1,zero_cipher);
  z_v[0].gpu();
  std::vector<seal::Ciphertext> b_v(1,cipher_bias);
  auto z_b_v = kernel_executor.add(z_v, b_v);
  cipher_bias = z_b_v[0];
  std::vector<double> vZ(weights.size(), 0);
  std::vector<seal::Ciphertext> cipher_Z =  context.encryptVector(vZ);
  for(auto &v : cipher_Z) {
      v.gpu();
  }  
  std::vector<seal::Ciphertext> cipher_weights = kernel_executor.add(cipher_weights0, cipher_Z);

  std::vector<std::vector<seal::Ciphertext>> cipher_inputs(inputs.size());

  for (std::size_t inputs_r = 0; inputs_r < inputs.size(); ++inputs_r)
  {
    auto cipher_input = context.encryptVector(
        gsl::span(inputs[inputs_r].data(), inputs[inputs_r].size()),
        context.encoder().slot_count());
    cipher_inputs[inputs_r] = kernel_executor.add(cipher_input,cipher_Z);
  }
  std::vector<seal::Ciphertext> cipher_retval;
  util::Timer t;
  t.start();  
  cipher_retval = kernel_executor.evaluateLogisticRegression(
        cipher_weights, cipher_inputs, cipher_bias, weights.size(), 3);
  std::cout << "LogReg compute time: " << double(t.elapsedMicroseconds())/1000 << " ms\n";  
}



TEST(seal_ckks_kernel_executor, LogReg_20x16)
{
    std::vector<int> coef_mods{60, 45, 45, 45, 45, 45};
    int n = (1024 * 16);
    LogisticRegression_CKKS(20, 16, n, coef_mods);
}

TEST(seal_ckks_kernel_executor, LogReg_5x4)
{
    std::vector<int> coef_mods{60, 45, 45, 45, 45, 45};
    int n = (1024 * 16);
    LogisticRegression_CKKS(5, 4, n, coef_mods);
}

TEST(seal_ckks_kernel_executor, encode_vector) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40, false, false);

  size_t num_plaintexts = 7;
  size_t batch_size = context.encoder().slot_count();

  std::vector<double> input(num_plaintexts * batch_size);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<double>(i);
  }

  std::vector<seal::Plaintext> encoded =
      context.encodeVector(gsl::span(input.data(), input.size()), batch_size);

  ASSERT_EQ(encoded.size(), num_plaintexts);

  std::vector<double> output;
  for (size_t i = 0; i < num_plaintexts; ++i) {
    std::vector<double> decoded;
    context.encoder().decode(encoded[i], decoded);
    output.insert(output.end(), decoded.begin(), decoded.end());
  }

  checkEqual(output, input);
}

TEST(seal_ckks_kernel_executor, encode_vector_batch_size_1) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40, false, false);

  size_t num_plaintexts = 7;
  size_t batch_size = 1;

  std::vector<double> input(num_plaintexts * batch_size);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<double>(i);
  }

  std::vector<seal::Plaintext> encoded =
      context.encodeVector(gsl::span(input.data(), input.size()), batch_size);

  ASSERT_EQ(encoded.size(), num_plaintexts);

  std::vector<double> output;
  for (size_t i = 0; i < num_plaintexts; ++i) {
    std::vector<double> decoded;
    context.encoder().decode(encoded[i], decoded);
    decoded.resize(batch_size);
    output.insert(output.end(), decoded.begin(), decoded.end());
  }
  checkEqual(output, input);
}

TEST(seal_ckks_kernel_executor, encode_vector_batch_size_3) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40, false, false);

  size_t num_plaintexts = 4;
  size_t batch_size = 3;

  std::vector<double> input(10);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<double>(i);
  }

  std::vector<seal::Plaintext> encoded =
      context.encodeVector(gsl::span(input.data(), input.size()), batch_size);

  ASSERT_EQ(encoded.size(), num_plaintexts);

  std::vector<double> output;
  for (size_t i = 0; i < num_plaintexts; ++i) {
    std::vector<double> decoded;
    context.encoder().decode(encoded[i], decoded);
    if (i == num_plaintexts - 1) {
      //size_t last_batch_size = input.size() % batch_size;
      decoded.resize(1);
    } else {
      decoded.resize(batch_size);
    }
    output.insert(output.end(), decoded.begin(), decoded.end());
  }
  checkEqual(output, input);
}

TEST(seal_ckks_kernel_executor, decode_vector) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40, false, false);

  size_t num_plaintexts = 7;
  size_t slot_count = context.encoder().slot_count();

  std::vector<double> input(num_plaintexts * slot_count);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<double>(i);
  }

  std::vector<seal::Plaintext> encoded =
      context.encodeVector(gsl::span(input.data(), input.size()), slot_count);
  std::vector<double> decoded = context.decodeVector(encoded, slot_count);

  checkEqual(input, decoded);
}

TEST(seal_ckks_kernel_executor, encrypt_vector) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40, false, false);

  size_t num_plaintexts = 7;
  size_t slot_count = context.encoder().slot_count();

  std::vector<double> input(num_plaintexts * slot_count);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<double>(i);
  }

  std::vector<seal::Plaintext> encoded =
      context.encodeVector(gsl::span(input.data(), input.size()), slot_count);
  std::vector<seal::Ciphertext> encrypted = context.encryptVector(encoded);

  ASSERT_EQ(encrypted.size(), encoded.size());

  for (size_t i = 0; i < encrypted.size(); ++i) {
    seal::Plaintext plain;
    context.decryptor().decrypt(encrypted[i], plain);

    std::vector<double> decrypted_decoded;
    context.encoder().decode(plain, decrypted_decoded);

    std::vector<double> decoded;
    context.encoder().decode(encoded[i], decoded);

    checkEqual(decoded, decrypted_decoded);
  }
}

TEST(seal_ckks_kernel_executor, decrypt_vector) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40, false, false);

  size_t num_plaintexts = 7;
  size_t slot_count = context.encoder().slot_count();

  std::vector<double> input(num_plaintexts * slot_count);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<double>(i);
  }

  std::vector<seal::Ciphertext> encrypted =
      context.encryptVector(gsl::span(input.data(), input.size()), slot_count);

  std::vector<seal::Plaintext> decrypted = context.decryptVector(encrypted);
  std::vector<double> decoded = context.decodeVector(decrypted, slot_count);

  checkEqual(input, decoded);
}

TEST(seal_ckks_kernel_executor, level) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40, false, false);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> input{0.1, 0.2, 0.3, 0.4};
  seal::Plaintext plain;

  context.encoder().encode(input, context.scale(), plain);

  seal::Ciphertext cipher1;
  context.encryptor().encrypt(plain, cipher1);
  seal::Ciphertext cipher2;
  context.encryptor().encrypt(plain, cipher2);

  EXPECT_EQ(kernel_executor.getLevel(plain), 2);
  EXPECT_EQ(kernel_executor.getLevel(cipher1), 2);

  kernel_executor.getEvaluator()->rescale_to_next_inplace(cipher1);
  EXPECT_EQ(kernel_executor.getLevel(cipher1), 1);

  kernel_executor.matchLevel(&cipher1, &cipher2);
  EXPECT_EQ(kernel_executor.getLevel(cipher1), 1);
  EXPECT_EQ(kernel_executor.getLevel(cipher2), 1);

  kernel_executor.matchLevel(&cipher1, &cipher2);
  EXPECT_EQ(kernel_executor.getLevel(cipher1), 1);
  EXPECT_EQ(kernel_executor.getLevel(cipher2), 1);

  kernel_executor.getEvaluator()->mod_switch_to_next_inplace(cipher2);
  EXPECT_EQ(kernel_executor.getLevel(cipher2), 0);

  kernel_executor.matchLevel(&cipher1, &cipher2);
  EXPECT_EQ(kernel_executor.getLevel(cipher1), 0);
  EXPECT_EQ(kernel_executor.getLevel(cipher2), 0);
}

TEST(seal_ckks_kernel_executor, add)
{
    SealCKKSContext context(8192, { 60, 40, 40 }, 1UL << 40, false, false);
    SealCKKSKernelExecutor kernel_executor(context);

    size_t num_plaintexts = 7;
    size_t slot_count = context.encoder().slot_count();

    std::vector<double> inputA(num_plaintexts * slot_count);
    std::vector<double> inputB(num_plaintexts * slot_count);
    std::vector<double> inputZ(num_plaintexts * slot_count, 0);
    std::vector<double> expected_out(num_plaintexts * slot_count);
    for (size_t i = 0; i < inputA.size(); ++i)
    {
        inputA[i] = static_cast<double>(i);
        inputB[i] = static_cast<double>(2 * i + 1);
        expected_out[i] = inputA[i] + inputB[i];
    }
    util::Timer t;
    t.start();
    std::vector<seal::Ciphertext> cipherA = context.encryptVector(gsl::span(inputA.data(), inputA.size()), slot_count);
    std::cout << "Cipher A time: " << t.elapsedMilliseconds() << " ms\n";
    t.start();
    std::vector<seal::Ciphertext> cipherB = context.encryptVector(gsl::span(inputB.data(), inputB.size()), slot_count);
    std::cout << "Cipher B time: " << t.elapsedMilliseconds() << " ms\n";
    std::vector<seal::Ciphertext> cipherZ = context.encryptVector(gsl::span(inputZ.data(), inputZ.size()), slot_count);
    t.start();
    std::vector<seal::Ciphertext> cipher_Aplus0 = kernel_executor.add(cipherA, cipherZ);
    //std::cout << "Overall A + 0 time: " << t.elapsedMilliseconds() << " ms\n";
    t.start();
    std::vector<seal::Ciphertext> cipher_Bplus0 = kernel_executor.add(cipherB, cipherZ);
    //std::cout << "Overall B + 0 time: " << t.elapsedMilliseconds() << " ms\n";

    t.start();
    std::vector<seal::Ciphertext> cipher_sum = kernel_executor.add(cipherA, cipherB);
    std::cout << "A + B compute time: " << double(t.elapsedMicroseconds())/1000 << " ms\n";

    std::vector<seal::Plaintext> plain_sum = context.decryptVector(cipher_sum);

    std::vector<double> output = context.decodeVector(plain_sum, slot_count);
    checkEqual(output, expected_out);
}

TEST(seal_ckks_kernel_executor, accumulate) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> inputZ(input.size(), 0);

  std::vector<seal::Ciphertext> ciphers =
      context.encryptVector(gsl::span(input.data(), input.size()));
  std::vector<seal::Ciphertext> ciphersZ =
      context.encryptVector(gsl::span(inputZ.data(), inputZ.size()));    
  for(auto &v : ciphers) {
        v.gpu();
  }


  std::vector<seal::Ciphertext> ciphers_0 = kernel_executor.add(ciphers, ciphersZ);
  util::Timer t;
  t.start();
  seal::Ciphertext cipher_sum =
      kernel_executor.accumulate(ciphers_0, input.size());
  std::cout << "accumulate compute time: " << double(t.elapsedMicroseconds())/1000 << " ms\n";

  seal::Plaintext plain_sum;
  context.decryptor().decrypt(cipher_sum, plain_sum);
  std::vector<double> output;
  context.encoder().decode(plain_sum, output);

  ASSERT_NEAR(output[0], 78, 0.01);
}

TEST(seal_ckks_kernel_executor, dot) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> inputA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> inputB{12, 11, 10, 9, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> inputZ(std::max(inputA.size(), inputB.size()), 0);


  std::vector<seal::Ciphertext> ciphersA =
      context.encryptVector(gsl::span(inputA.data(), inputA.size()));

  std::vector<seal::Ciphertext> ciphersB =
      context.encryptVector(gsl::span(inputB.data(), inputB.size()));

  std::vector<seal::Ciphertext> ciphersZ =
      context.encryptVector(gsl::span(inputZ.data(), inputZ.size()));   

    for(auto &v : ciphersA) {
        v.gpu();
    }
    for(auto &v : ciphersB) {
        v.gpu();
    }

  std::vector<seal::Ciphertext> ciphersA_0 = kernel_executor.add(ciphersA, ciphersZ);
  std::vector<seal::Ciphertext> ciphersB_0 = kernel_executor.add(ciphersB, ciphersZ);

  util::Timer t;
  t.start();
  seal::Ciphertext cipher_dot =
      kernel_executor.dot(ciphersA_0, ciphersB_0, inputA.size());
  std::cout << "dot compute time: " << double(t.elapsedMicroseconds())/1000 << " ms\n";

  seal::Plaintext plain_dot;
  context.decryptor().decrypt(cipher_dot, plain_dot);
  std::vector<double> output;
  context.encoder().decode(plain_dot, output);

  ASSERT_NEAR(output[0], 720, 0.01);
}

void MatMulVal_CKKS(size_t dim1, size_t dim2, size_t dim3, size_t n, const std::vector<int> & coeff_modulus)
{
    SealCKKSContext context(n, coeff_modulus, 1UL << 40, true, true);
    SealCKKSKernelExecutor kernel_executor(context);

    std::vector<std::vector<double>> mat_A(dim1, std::vector<double>(dim2));
    std::vector<std::vector<double>> mat_B_T(dim3, std::vector<double>(dim2));
    std::vector<double> inputAZ(mat_A[0].size(), 0);
    std::vector<double> inputBZ(mat_B_T[0].size(), 0);
    std::default_random_engine rand_gen;
    std::uniform_real_distribution<double> uniform_rnd(-1.5, 1.5);

  std::vector<seal::Ciphertext> ciphersAZ =
      context.encryptVector(gsl::span(inputAZ.data(), inputAZ.size()));     
  std::vector<seal::Ciphertext> ciphersBZ =
      context.encryptVector(gsl::span(inputBZ.data(), inputBZ.size()));   
  for(auto &v : ciphersAZ) {
        v.gpu();
  }
  for(auto &v : ciphersBZ) {
        v.gpu();
  }              

  
   
  for (std::size_t r = 0; r < mat_A.size(); ++r)
      for (std::size_t c = 0; c < mat_A[r].size(); ++c)
        mat_A[r][c] = uniform_rnd(rand_gen);
  for (std::size_t r = 0; r < mat_B_T.size(); ++r)
    for (std::size_t c = 0; c < mat_B_T[r].size(); ++c)
        mat_B_T[r][c] = uniform_rnd(rand_gen);

  std::vector<std::vector<seal::Ciphertext>> cipher_A(mat_A.size());
  std::vector<std::vector<seal::Ciphertext>> cipher_B_T(mat_B_T.size());


  for (std::size_t r = 0; r < mat_A.size(); ++r)
  {
    auto cipher_Arow = context.encryptVector(mat_A[r]);
    cipher_A[r] = kernel_executor.add(cipher_Arow, ciphersAZ);
  }    
  for (std::size_t r = 0; r < mat_B_T.size(); ++r)
  {
    auto cipher_Bcol = context.encryptVector(mat_B_T[r]);
    cipher_B_T[r] = kernel_executor.add(cipher_Bcol, ciphersBZ);
    //cipher_B_T[r] = context.encryptVector(mat_B_T[r]);
  }
  util::Timer t;
  t.start();     
  auto cypherAxB = kernel_executor.matMul(cipher_A, cipher_B_T, dim2);
  std::cout << "MatMul compute time: " << double(t.elapsedMicroseconds())/1000 << " ms\n";
   
}
TEST(seal_ckks_kernel_executor, matMul_100x10x1)
{
    std::vector<int> coef_mods{ 60, 40, 40 };
    auto n = size_t(1024 * 8);
    MatMulVal_CKKS(100, 10, 1, n, coef_mods);
}

TEST(seal_ckks_kernel_executor, matMul_10x9x8)
{
    std::vector<int> coef_mods{ 60, 40, 40 };
    auto n = size_t(1024 * 8);
    MatMulVal_CKKS(10, 9, 8, n, coef_mods);
}

TEST(seal_ckks_kernel_executor, matMul) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<std::vector<double>> mat_A{
      {1, -2, 3, -4}, {-5, 6, -7, 8}, {9, -10, 11, -12}};
  std::vector<std::vector<double>> mat_B_T{{21, -23, 25, 27},
                                           {-22, 24, -26, -28}};
  std::vector<double> exp_out{34, -36, -202, 212, 370, -388};
  std::vector<double> output;
  std::vector<double> inputAZ(mat_A[0].size(), 0);
  std::vector<double> inputBZ(mat_B_T[0].size(), 0);

  std::vector<std::vector<seal::Ciphertext>> cipher_A(mat_A.size());
  std::vector<std::vector<seal::Ciphertext>> cipher_B_T(mat_B_T.size());
  std::vector<seal::Ciphertext> ciphersAZ =
      context.encryptVector(gsl::span(inputAZ.data(), inputAZ.size()));     
  std::vector<seal::Ciphertext> ciphersBZ =
      context.encryptVector(gsl::span(inputBZ.data(), inputBZ.size()));           
  for (std::size_t r = 0; r < mat_A.size(); ++r)
    {
      auto cipher_Arow = context.encryptVector(mat_A[r]);
      for(auto &v : cipher_Arow) {
        v.gpu();
      }
      cipher_A[r] = kernel_executor.add(cipher_Arow, ciphersAZ);
    }
  for (std::size_t r = 0; r < mat_B_T.size(); ++r)
  {
      auto cipher_Bcol = context.encryptVector(mat_B_T[r]);
      for(auto &v : cipher_Bcol) {
        v.gpu();
      }
      cipher_B_T[r] = kernel_executor.add(cipher_Bcol, ciphersBZ);
  }

  util::Timer t;
  t.start();  
  auto cipher_AB =
      kernel_executor.matMul(cipher_A, cipher_B_T, mat_A.front().size());
  std::cout << "MatMul compute time: " << double(t.elapsedMicroseconds())/1000 << " ms\n";

  output.resize(cipher_AB.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    std::vector<double> tmp;
    seal::Plaintext plain;
    context.decryptor().decrypt(cipher_AB[i], plain);
    context.encoder().decode(plain, tmp);
    output[i] = tmp.front();
  }

  checkEqual(output, exp_out);
}

TEST(seal_ckks_kernel_executor, logisticRegression4x3_SigDeg3) {
  constexpr unsigned int sigmoid_deg = 3;
  double bias;
  std::vector<double> weights;
  std::vector<std::vector<double>> inputs;
  std::vector<double> ground_truth;
  prepareLogisticRegression4x3(bias, weights, inputs, ground_truth);
  std::vector<double> output =
      testLogisticRegression4x3(bias, weights, inputs, sigmoid_deg);

  checkEqual(output, ground_truth);
}

TEST(seal_ckks_kernel_executor, dotCipherBatchAxis4x3x2) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> inputA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> inputB{1, 2, 3, 4, 5, 6};
  std::vector<double> exp_out{38, 44, 50, 56, 83, 98, 113, 128};
  std::vector<double> output = testDotCipherBatchAxis(inputA, inputB, 4, 3, 2);

  checkEqual(output, exp_out);
}

TEST(seal_ckks_kernel_executor, DotCipherBatchAxis2x2x2) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> input1{1, 2, 3, 4};
  std::vector<double> input2{1, 2, 3, 4};
  std::vector<double> exp_out{7, 10, 15, 22};
  std::vector<double> output = testDotCipherBatchAxis(input1, input2, 2, 2, 2);

  checkEqual(output, exp_out);
}

TEST(seal_ckks_kernel_executor, DotPlainBatchAxis4x3x2) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> inputA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> inputB{1, 2, 3, 4, 5, 6};
  std::vector<double> exp_out{38, 44, 50, 56, 83, 98, 113, 128};
  std::vector<double> output = testDotCipherBatchAxis(inputA, inputB, 4, 3, 2);

  checkEqual(output, exp_out);
}

TEST(seal_ckks_kernel_executor, DotPlainBatchAxis2x2x2) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  std::vector<double> input1{1, 2, 3, 4};
  std::vector<double> input2{1, 2, 3, 4};
  std::vector<double> exp_out{7, 10, 15, 22};
  std::vector<double> output = testDotPlainBatchAxis(input1, input2, 2, 2, 2);

  checkEqual(output, exp_out);
}
TEST(seal_ckks_kernel_bench, DotPlainBatchAxis100x10x1) {
  SealCKKSContext context(8192, {60, 40, 40}, 1UL << 40);
  SealCKKSKernelExecutor kernel_executor(context);

  size_t batch_size = 1;
  size_t dim1 = 100;
  size_t dim2 = 10;
  size_t dim3 = 1;

  std::vector<double> input1(dim1 * dim2, 7);
  std::vector<double> input2(dim2 * dim3, 8);

  std::vector<seal::Ciphertext> arg1 = context.encryptVector(
      gsl::span(input1.data(), input1.size()), batch_size);

  std::vector<seal::Plaintext> arg2 =
      context.encodeVector(gsl::span(input2.data(), input2.size()), batch_size);

  util::Timer t;
  t.start();
  xehe::ext::clear_export_table();

  std::vector<seal::Ciphertext> cipher_dot =
      kernel_executor.dotPlainBatchAxis(arg1, arg2, dim1, dim2, dim3);

  t.stop();
  xehe::ext::export_table();

  std::cout << " dot app elapsed: " << t.elapsedMicroseconds() << std::endl;
  std::vector<seal::Plaintext> decrypted = context.decryptVector(cipher_dot);
  //return context.decodeVector(decrypted, batch_size);

}

}  // namespace heseal
}  // namespace he
}  // namespace intel
