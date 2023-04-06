// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "seal/seal_ckks_context.h"

#include <seal/seal.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace intel {
namespace he {
namespace heseal {

std::vector<seal::Plaintext> SealCKKSContext::encodeVector(
    const gsl::span<const double>& values, size_t batch_size) {
  size_t total_chunks =
      values.size() / batch_size + (values.size() % batch_size == 0 ? 0 : 1);
  size_t last_chunk_size =
      values.size() % batch_size == 0 ? batch_size : values.size() % batch_size;

  std::vector<seal::Plaintext> ret(total_chunks);
#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for
#endif
  for (int64_t j = 0; j < int64_t(total_chunks); ++j)
  {
      auto i = size_t(j);
      size_t actual_chunk_size = (i == total_chunks - 1) ? last_chunk_size : batch_size;
      gsl::span data_chunk(&values[i * batch_size], actual_chunk_size);
      m_encoder->encode(data_chunk, m_scale, ret[i]);
  }
  return ret;
}

std::vector<seal::Plaintext> SealCKKSContext::encodeVector(
    const gsl::span<const double>& v) {
  std::size_t slot_count = m_encoder->slot_count();
  std::size_t total_chunks =
      v.size() / slot_count + (v.size() % slot_count == 0 ? 0 : 1);
  gsl::span<const double> data = v;
  std::vector<seal::Plaintext> retval;
  retval.reserve(total_chunks);
  while (!data.empty()) {
    std::size_t actual_chunk_size =
        (data.size() > slot_count ? slot_count : data.size());
    gsl::span data_chunk = data.first(actual_chunk_size);
    data = data.last(data.size() - actual_chunk_size);
    seal::Plaintext plain;
    m_encoder->encode(data_chunk, m_scale, plain);
    retval.emplace_back(std::move(plain));
  }
  return retval;
}

std::vector<double> SealCKKSContext::decodeVector(
    const std::vector<seal::Plaintext>& plain, size_t batch_size) {
  std::vector<double> ret(plain.size() * batch_size);
#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for
#endif
  for (int64_t j = 0; j < int64_t(plain.size()); ++j)
  {
      auto i = size_t(j);
      std::vector<double> tmp;
      m_encoder->decode(plain[i], tmp);
      std::copy(tmp.begin(), tmp.begin() + long(batch_size), ret.begin() + long(i * batch_size));
  }
  return ret;
}

std::vector<double> SealCKKSContext::decodeVector(
    const std::vector<seal::Plaintext>& plain) {
  std::size_t slot_count = m_encoder->slot_count();
  std::vector<double> ret(plain.size() * slot_count);
#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for
#endif
  for (int64_t j = 0; j < int64_t(plain.size()); ++j)
  {
      auto i = size_t(j);
      std::vector<double> tmp;
      m_encoder->decode(plain[i], tmp);
      std::size_t min_size = std::min(slot_count, tmp.size());
      std::copy(tmp.begin(), tmp.begin() + long(min_size), ret.begin() + long(i * slot_count));
  }
  return ret;
}

std::vector<seal::Ciphertext> SealCKKSContext::encryptVector(
    const std::vector<seal::Plaintext>& plain) {
  std::vector<seal::Ciphertext> ret(plain.size());
#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for
#endif
  for (int64_t j = 0; j < int64_t(plain.size()); ++j)
  {
      auto i = size_t(j);
      m_encryptor->encrypt(plain[i], ret[i]);
  }
  return ret;
}

std::vector<seal::Plaintext> SealCKKSContext::decryptVector(
    const std::vector<seal::Ciphertext>& cipher) {
  std::vector<seal::Plaintext> ret(cipher.size());
#ifdef SEALTEST_OMP_ENABLED
#pragma omp parallel for
#endif
  for (int64_t j = 0; j < int64_t(cipher.size()); ++j)
  {
      auto i = size_t(j);
      m_decryptor->decrypt(cipher[i], ret[i]);
  }
  return ret;
}

}  // namespace heseal
}  // namespace he
}  // namespace intel
