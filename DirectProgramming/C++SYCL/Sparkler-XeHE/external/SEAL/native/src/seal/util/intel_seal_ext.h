// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020-2021 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

#pragma once

#ifdef SEAL_USE_INTEL_LATTICE
#include "seal/util/locks.h"
#include <unordered_map>
#include "intel-lattice/intel-lattice.hpp"

namespace intel
{
    namespace seal_ext
    {
        struct HashPair
        {
            template <class T1, class T2>
            size_t operator()(const std::pair<T1, T2> &p) const
            {
                auto hash1 = std::hash<T1>{}(std::get<0>(p));
                auto hash2 = std::hash<T2>{}(std::get<1>(p));
                return hash_combine(hash1, hash2);
            }

            static size_t hash_combine(size_t lhs, size_t rhs)
            {
                lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
                return lhs;
            }
        };

        static std::unordered_map<std::pair<uint64_t, uint64_t>, intel::lattice::NTT, HashPair> ntt_cache_;

        static seal::util::ReaderWriterLocker ntt_cache_locker_;

        static intel::lattice::NTT get_ntt(size_t N, uint64_t p)
        {
            std::pair<uint64_t, uint64_t> key{ N, p };

            // Enable shared access of NTT already present
            {
                seal::util::ReaderLock reader_lock(ntt_cache_locker_.acquire_read());
                auto ntt_it = ntt_cache_.find(key);
                if (ntt_it != ntt_cache_.end())
                {
                    return ntt_it->second;
                }
            }

            // Deal with NTT not yet present
            seal::util::WriterLock write_lock(ntt_cache_locker_.acquire_write());

            // Check ntt_cache for value (maybe added by another thread)
            auto ntt_it = ntt_cache_.find(key);
            if (ntt_it == ntt_cache_.end())
            {
                ntt_it = ntt_cache_.emplace(std::move(key), intel::lattice::NTT(N, p)).first;
            }
            return ntt_it->second;
        }

        static void compute_forward_ntt(
            seal::util::CoeffIter operand, size_t N, uint64_t p, uint64_t input_mod_factor, uint64_t output_mod_factor)
        {
            get_ntt(N, p).ComputeForward(operand, operand, input_mod_factor, output_mod_factor);
        }

        static void compute_inverse_ntt(
            seal::util::CoeffIter operand, size_t N, uint64_t p, uint64_t input_mod_factor, uint64_t output_mod_factor)
        {
            get_ntt(N, p).ComputeInverse(operand, operand, input_mod_factor, output_mod_factor);
        }

    } // namespace seal_ext
} // namespace intel
#endif
