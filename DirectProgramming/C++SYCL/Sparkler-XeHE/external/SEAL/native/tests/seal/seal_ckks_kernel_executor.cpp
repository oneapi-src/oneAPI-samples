// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "seal/seal_ckks_kernel_executor.h"

#include <stdexcept>
#include "seal/util/timer.h"

//#include "kernels/seal/seal_ckks_context.h"
#include "seal_omp_utils.h"

namespace intel {
namespace he {
namespace heseal {

const double SealCKKSKernelExecutor::sigmoid_coeff_3[] = {0.5, 0.15012, 0.0,
                                                          -0.001593008};
const double SealCKKSKernelExecutor::sigmoid_coeff_5[] = {
    0.5, 0.19131, 0.0, -0.0045963, 0.0, 0.000041233};
const double SealCKKSKernelExecutor::sigmoid_coeff_7[] = {
    0.5, 0.21687, 0.0, -0.008191543, 0.0, 0.000165833, 0.0, 0.000001196};

SealCKKSKernelExecutor::SealCKKSKernelExecutor(
    const seal::EncryptionParameters& params, double scale,
    const seal::PublicKey& public_key, const seal::RelinKeys& relin_keys,
    const seal::GaloisKeys& galois_keys) {
  if (params.scheme() != seal::scheme_type::ckks)
    throw std::invalid_argument("Only CKKS scheme supported.");
  m_scale = scale;
  m_public_key = public_key;
  m_relin_keys = relin_keys;
  m_galois_keys = galois_keys;
  m_pcontext.reset(new seal::SEALContext(params));
  m_pevaluator = std::make_shared<seal::Evaluator>(*m_pcontext);
  m_pencoder = std::make_shared<seal::CKKSEncoder>(*m_pcontext);
  m_pencryptor = std::make_shared<seal::Encryptor>(*m_pcontext, m_public_key);
}

SealCKKSKernelExecutor::~SealCKKSKernelExecutor() {
  m_pencryptor.reset();
  m_pencoder.reset();
  m_pevaluator.reset();
}

std::vector<seal::Ciphertext> SealCKKSKernelExecutor::add(
    const std::vector<seal::Ciphertext> &A, const std::vector<seal::Ciphertext> &B)
{
    // util::Timer t;
    std::vector<seal::Ciphertext> retval;
    if (A.size() != B.size())
        throw std::invalid_argument("A.size() != B.size()");
    retval.resize(A.size());

    //t.start();
    for (size_t i = 0; i < retval.size(); ++i)
    {
        m_pevaluator->add(A[i], B[i], retval[i]);
    }
    //std::cout << "add time: " << double(t.elapsedMicroseconds())/(1000*retval.size()) << " ms " << retval.size() <<std::endl;

    return retval;
}

seal::Ciphertext SealCKKSKernelExecutor::accumulate_internal(
    const seal::Ciphertext& cipher_input, std::size_t count) {
  seal::Ciphertext retval;
  if (count > 0) {
    retval = cipher_input;
    auto max_steps = (1 << seal::util::get_significant_bit_count(count));
    for (int steps = 1; steps < max_steps; steps <<= 1) {
      seal::Ciphertext rotated;
      m_pevaluator->rotate_vector(retval, steps, m_galois_keys, rotated,
                                  seal::MemoryPoolHandle::ThreadLocal());
      m_pevaluator->add_inplace(retval, rotated);
    }
  } else {
    m_pencryptor->encrypt_zero(retval);
    retval.scale() = cipher_input.scale();
  }

  return retval;
}

seal::Ciphertext SealCKKSKernelExecutor::accumulate(
    const std::vector<seal::Ciphertext>& V, std::size_t count) {
  seal::Ciphertext retval;
  m_pencryptor->encrypt_zero(retval.gpu());
  //retval.gpu();

  if (count > 0) {
    std::size_t slot_count = m_pencoder->slot_count();
    for (std::size_t i = 0; i < V.size(); ++i) {
      std::size_t chunk_count =
          i + 1 < V.size() ? slot_count : count % slot_count;
      seal::Ciphertext chunk_retval = accumulate_internal(V[i], chunk_count);

      matchLevel(&retval, &chunk_retval);
      retval.scale() = chunk_retval.scale();
      m_pevaluator->add_inplace(retval, chunk_retval);
    }
  }

  return retval;
}

seal::Ciphertext SealCKKSKernelExecutor::dot(
    const std::vector<seal::Ciphertext>& A,
    const std::vector<seal::Ciphertext>& B, size_t count) {
  seal::Ciphertext retval;
  util::Timer t;
  double mul_t = 0, relin_t = 0, rescl_t = 0, accum_t = 0; 
  if (count > 0) {
    std::vector<seal::Ciphertext> AB(A.size());
    for (size_t i = 0; i < AB.size(); ++i) {
      t.start();
      m_pevaluator->multiply(A[i], B[i], AB[i]);
      t.stop();
      mul_t += t.elapsedMicroseconds();
      t.start();
      m_pevaluator->relinearize_inplace(AB[i], m_relin_keys);
      t.stop();
      relin_t  += t.elapsedMicroseconds();
      t.start();
      m_pevaluator->rescale_to_next_inplace(AB[i]);
      rescl_t += t.elapsedMicroseconds();
      t.stop();
    }
    t.start();
    retval = accumulate(AB, count);
    t.stop();
    accum_t += t.elapsedMicroseconds();
  } else {
    m_pencryptor->encrypt_zero(retval);
    retval.scale() = m_scale;
  }
#if 0 
  std::cout 
  << "Mul: " << mul_t/(A.size() *1000) << "\n"
  << "Relin: " << relin_t/(A.size() *1000) << "\n"
  << "Rescl: " << rescl_t/(A.size() *1000) << "\n"
  << "Accum: " << accum_t/(A.size() *1000) << "\n";
#endif
  return retval;
}

std::vector<seal::Ciphertext> SealCKKSKernelExecutor::matMul(
    const std::vector<std::vector<seal::Ciphertext>>& A,
    const std::vector<std::vector<seal::Ciphertext>>& B_T, size_t cols) {
  std::vector<seal::Ciphertext> retval(A.size() * B_T.size());
  
#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for collapse(2) \
 num_threads(OMPUtilitiesS::getThreadsAtLevel())
 #else
  std::cout << "OMP removed!\n";
 #endif
  for (size_t A_r = 0; A_r < A.size(); ++A_r)
    for (size_t B_T_r = 0; B_T_r < B_T.size(); ++B_T_r)
      {
        retval[A_r * B_T.size() + B_T_r] = dot(A[A_r], B_T[B_T_r], cols);
      }
  return retval;
}

std::vector<seal::Ciphertext> SealCKKSKernelExecutor::evaluatePolynomial(
    const std::vector<seal::Ciphertext>& inputs,
    const gsl::span<const double>& coefficients) {
  if (coefficients.empty())
    throw std::invalid_argument("coefficients cannot be empty");

  std::vector<seal::Plaintext> plain_coeff(coefficients.size());
  for (size_t coeff_i = 0; coeff_i < coefficients.size(); ++coeff_i)
    m_pencoder->encode(coefficients[coeff_i], m_scale, plain_coeff[coeff_i]);

  std::vector<seal::Ciphertext> retval(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    seal::Ciphertext cipher_input_chunk = inputs[i];
    seal::Ciphertext& cipher_result_chunk = retval[i];

    auto it = plain_coeff.rbegin();
    m_pencryptor->encrypt(*it, cipher_result_chunk);
    for (++it; it != plain_coeff.rend(); ++it) {
      matchLevel(&cipher_input_chunk, &cipher_result_chunk);

      m_pevaluator->multiply_inplace(cipher_result_chunk, cipher_input_chunk);
      m_pevaluator->relinearize_inplace(cipher_result_chunk, m_relin_keys);
      m_pevaluator->rescale_to_next_inplace(cipher_result_chunk);
      auto result_parms_id = cipher_result_chunk.parms_id();
      m_pevaluator->mod_switch_to_inplace(*it, result_parms_id);

      // TODO(fboemer): check if scales are close enough
      cipher_result_chunk.scale() = m_scale;
      m_pevaluator->add_plain_inplace(cipher_result_chunk, *it);
    }
  }
  return retval;
}

std::vector<seal::Ciphertext> SealCKKSKernelExecutor::collapse(
    const std::vector<seal::Ciphertext>& ciphers) {
  //std::cout << "Collapse " << "\n";
  std::vector<seal::Ciphertext> retval;
  size_t slot_count = m_pencoder->slot_count();
  size_t total_chunks =
      ciphers.size() / slot_count + (ciphers.size() % slot_count == 0 ? 0 : 1);
  util::Timer t, c_t;
  double col_t = 0, gpu_t = 0; //chunk_t = 0
  c_t.start();    
  retval.resize(total_chunks);
  seal::Plaintext plain;
  m_pencoder->encode(0.0, m_scale, plain);
  for (size_t i = 0; i < retval.size(); ++i)
    m_pencryptor->encrypt(plain, retval[i]);
  std::vector<double> identity;
  size_t cipher_i = 0;
  for (size_t chunk_i = 0; chunk_i < total_chunks; ++chunk_i) {
    seal::Ciphertext& retval_chunk = retval[chunk_i];
    identity.resize((chunk_i + 1 == total_chunks ? ciphers.size() % slot_count
                                                 : slot_count),
                    0.0);
    for (size_t i = 0; i < identity.size(); ++i) {
      const seal::Ciphertext& cipher = ciphers[cipher_i++];
      if (i > 0) identity[i - 1] = 0.0;
      identity[i] = 1.0;
      m_pencoder->encode(identity, m_scale, plain);
      seal::Ciphertext tmp;
      t.start();
      m_pevaluator->rotate_vector(cipher, -static_cast<int>(i), m_galois_keys,
                                  tmp);
      m_pevaluator->mod_switch_to_inplace(plain, tmp.parms_id());
      m_pevaluator->multiply_plain_inplace(tmp, plain);
      m_pevaluator->relinearize_inplace(tmp, m_relin_keys);
      m_pevaluator->rescale_to_next_inplace(tmp);
      matchLevel(&retval_chunk, &tmp);
      tmp.scale() = m_scale;
      retval_chunk.scale() = m_scale;
      m_pevaluator->add_inplace(retval_chunk, tmp);
      t.stop();
      gpu_t += t.elapsedMicroseconds();
    }
    identity.back() = 0.0;
  }
  c_t.stop();
  col_t += c_t.elapsedMicroseconds();
  std::cout << "Collapse total compute time: " << col_t/(1000) << " ms\n";
  std::cout << "Collapse gpu compute time: " << gpu_t/(1000) << " ms\n";
  return retval;
}

std::vector<seal::Ciphertext> SealCKKSKernelExecutor::evaluateLinearRegression(
    std::vector<seal::Ciphertext>& weights,
    std::vector<std::vector<seal::Ciphertext>>& inputs, seal::Ciphertext& bias,
    size_t weights_count) {
  std::vector<std::vector<seal::Ciphertext>> weights_copy{weights};
  std::vector<seal::Ciphertext> retval =
      collapse(matMul(weights_copy, inputs, weights_count));
  weights_copy.clear();
  for (size_t i = 0; i < retval.size(); ++i) {
    matchLevel(&retval[i], &bias);
    bias.scale() = m_scale;
    retval[i].scale() = m_scale;
    m_pevaluator->add_inplace(retval[i], bias);
  }
  return retval;
}

std::vector<seal::Ciphertext>
SealCKKSKernelExecutor::evaluateLogisticRegression(
    std::vector<seal::Ciphertext>& weights,
    std::vector<std::vector<seal::Ciphertext>>& inputs, seal::Ciphertext& bias,
    size_t weights_count, unsigned int sigmoid_degree) {
  std::vector<seal::Ciphertext> retval =
      evaluateLinearRegression(weights, inputs, bias, weights_count);

  switch (sigmoid_degree) {
    case 5:
      retval = sigmoid<5>(retval);
      break;
    case 7:
      retval = sigmoid<7>(retval);
      break;
    default:
      retval = sigmoid<3>(retval);
      break;
  }
  return retval;
}

void SealCKKSKernelExecutor::matchLevel(seal::Ciphertext* a,
                                        seal::Ciphertext* b) const {
  int a_level = int(getLevel(*a));
  int b_level = int(getLevel(*b));
  a->gpu();
  b->gpu();
  if (a_level > b_level)
    m_pevaluator->mod_switch_to_inplace(*a, b->parms_id());
  else if (a_level < b_level)
    m_pevaluator->mod_switch_to_inplace(*b, a->parms_id());
}

std::vector<seal::Ciphertext> SealCKKSKernelExecutor::dotPlainBatchAxis(
    const std::vector<seal::Ciphertext>& arg1,
    const std::vector<seal::Plaintext>& arg2, size_t dim0, size_t dim1,
    size_t dim2) {
  if (arg1.size() != dim0 * dim1) {
    throw std::runtime_error("DotPlainBatchAxis arg1 wrong shape");
  }
  if (arg2.size() != dim1 * dim2) {
    throw std::runtime_error("DotPlainBatchAxis arg2 wrong shape");
  }

  std::vector<seal::Ciphertext> out(dim0 * dim2);
#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for collapse(2) \
num_threads(OMPUtilitiesS::getThreadsAtLevel())
#else
  std::cout << "OMP removed!\n";
#endif
  for (size_t out_ind0 = 0; out_ind0 < dim0; ++out_ind0) {
    for (size_t out_ind1 = 0; out_ind1 < dim2; ++out_ind1) {
      size_t out_idx = colMajorIndex(dim0, dim2, out_ind0, out_ind1);
      for (size_t inner_dim = 0; inner_dim < dim1; inner_dim++) {
        size_t arg1_idx = colMajorIndex(dim0, dim1, out_ind0, inner_dim);
        size_t arg2_idx = colMajorIndex(dim1, dim2, inner_dim, out_ind1);
        if (inner_dim == 0) {
          m_pevaluator->multiply_plain(arg1[arg1_idx], arg2[arg2_idx],
                                       out[out_idx]);
          continue;
        }
        // seal::Ciphertext tmp;
        // m_pevaluator->multiply_plain(arg1[arg1_idx], arg2[arg2_idx], tmp);
        // m_pevaluator->add_inplace(out[out_idx], tmp);
        m_pevaluator->multiply_plain_add_inplace(out[out_idx], arg1[arg1_idx], arg2[arg2_idx]);
      }
    }
  }
  return out;
}

std::vector<seal::Ciphertext> SealCKKSKernelExecutor::dotCipherBatchAxis(
    const std::vector<seal::Ciphertext>& arg1,
    const std::vector<seal::Ciphertext>& arg2, size_t dim0, size_t dim1,
    size_t dim2) {
  if (arg1.size() != dim0 * dim1) {
    throw std::runtime_error("DotCipherBatchAxis arg1 wrong shape");
  }
  if (arg2.size() != dim1 * dim2) {
    throw std::runtime_error("DotCipherBatchAxis arg2 wrong shape");
  }

  std::vector<seal::Ciphertext> out(dim0 * dim2);

#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for collapse(2) \
num_threads(OMPUtilitiesS::getThreadsAtLevel())
#else
  std::cout << "OMP removed!\n";
#endif
  for (size_t out_ind0 = 0; out_ind0 < dim0; ++out_ind0) {
    for (size_t out_ind1 = 0; out_ind1 < dim2; ++out_ind1) {
      size_t out_idx = colMajorIndex(dim0, dim2, out_ind0, out_ind1);
      for (size_t inner_dim = 0; inner_dim < dim1; inner_dim++) {
        size_t arg1_idx = colMajorIndex(dim0, dim1, out_ind0, inner_dim);
        size_t arg2_idx = colMajorIndex(dim1, dim2, inner_dim, out_ind1);

        if (inner_dim == 0) {
          m_pevaluator->multiply(arg1[arg1_idx], arg2[arg2_idx], out[out_idx]);
          continue;
        }
        // seal::Ciphertext tmp;
        // m_pevaluator->multiply(arg1[arg1_idx], arg2[arg2_idx], tmp);
        // m_pevaluator->add_inplace(out[out_idx], tmp);
        m_pevaluator->multiply_add_inplace(out[out_idx], arg1[arg1_idx], arg2[arg2_idx]);
      }
    }
  }

#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for num_threads(OMPUtilitiesS::getThreadsAtLevel())
#else
  std::cout << "OMP removed!\n";
#endif

  for (size_t out_idx = 0; out_idx < dim0 * dim2; ++out_idx) {
    m_pevaluator->relinearize_inplace(out[out_idx], m_relin_keys);
    m_pevaluator->rescale_to_next_inplace(out[out_idx]);
  }
  return out;
}

}  // namespace heseal
}  // namespace he
}  // namespace intel
