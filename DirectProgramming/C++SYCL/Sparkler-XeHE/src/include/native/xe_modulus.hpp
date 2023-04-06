/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_MODULUS_H
#define XeHE_MODULUS_H

#define XeHE_NODISCARD [[nodiscard]]

#include <random>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

// XeHE
#include "xe_uintarith_core.hpp"

namespace xehe
{
    namespace native 
    {

        // get min and max bounds for bit-length of all coefficient moduli
        template<typename T>
        inline int get_mod_bit_count_max(){
            if (sizeof(T) == 8)
                return 61;
            else if (sizeof(T) == 4)
                return 29;
            //else return ;
        }

        template<typename T>
        inline int get_mod_bit_count_min(){
            if (sizeof(T) == 8 || sizeof(T) == 4)
                return 2;
        }

        template<typename T>
        bool is_mod_prime(T value, const T *const_ratio, size_t num_rounds = 40)
        {
            
            // First check the simplest cases.
            if (value < 2)
            {
                return false;
            }
            if (2 == value)
            {
                return true;
            }
            if (0 == (value & 0x1))
            {
                return false;
            }
            if (3 == value)
            {
                return true;
            }
            if (0 == (value % 3))
            {
                return false;
            }
            if (5 == value)
            {
                return true;
            }
            if (0 == (value % 5))
            {
                return false;
            }
            if (7 == value)
            {
                return true;
            }
            if (0 == (value % 7))
            {
                return false;
            }
            if (11 == value)
            {
                return true;
            }
            if (0 == (value % 11))
            {
                return false;
            }
            if (13 == value)
            {
                return true;
            }
            if (0 == (value % 13))
            {
                return false;
            }

            // Second, Miller-Rabin test.
            // Find r and odd d that satisfy value = 2^r * d + 1.
            T d = value - 1;
            T r = 0;
            while (0 == (d & 0x1))
            {
                d >>= 1;
                r++;
            }
            if (r == 0)
            {
                return false;
            }

            // 1) Pick a = 2, check a^(value - 1).
            // 2) Pick a randomly from [3, value - 1], check a^(value - 1).
            // 3) Repeat 2) for another num_rounds - 2 times.
            std::random_device rand;
            std::uniform_int_distribution<T> dist(3, value - 1);
            for (size_t i = 0; i < num_rounds; i++)
            {
                T a = i ? dist(rand) : 2;
                T x = xehe::native::exp_uint_mod<T>(a, d, value, const_ratio);
                if (x == 1 || x == value - 1)
                {
                    continue;
                }
                T count = 0;
                do
                {
                    // x = xehe::native::mul_mod(x, x, modulus);
                    x = xehe::native::mul_mod<T>(x, x, value, const_ratio);
                    count++;
                } while (x != value - 1 && count < r - 1);
                if (x != value - 1)
                {
                    return false;
                }
            }
            return true;
        }


        /**
        Represent an integer modulus of up to 61 bits. An instance of the Modulus
        class represents a non-negative integer modulus up to 61 bits. In particular,
        the encryption parameter plain_modulus, and the primes in coeff_modulus, are
        represented by instances of Modulus. The purpose of this class is to
        perform and store the pre-computation required by Barrett reduction.
        */
        template<typename T>
        class Modulus
        {
        public:
            /**
            Creates a Modulus instance. The value of the Modulus is set to
            the given value, or to zero by default.
            */
            Modulus(T value = 0)
            {
                set_value(value);
            }

            /*
            Returns the significant bit count of the value of the current Modulus.
            */
            XeHE_NODISCARD inline int bit_count() const noexcept
            {
                return bit_count_;
            }

            /**
            Returns the size (in 64-bit words) of the value of the current Modulus.
            */
            XeHE_NODISCARD inline std::size_t uint_count() const noexcept
            {
                return uint_count_;
            }

            /**
            Returns a const pointer to the value of the current Modulus.
            */
            XeHE_NODISCARD inline const T *data() const noexcept
            {
                return &value_;
            }

            /**
            Returns the value of the current Modulus.
            */
            XeHE_NODISCARD inline T value() const noexcept
            {
                return value_;
            }

            /**
            Returns the Barrett ratio computed for the value of the current Modulus.
            The first two components of the Barrett ratio are the floor of 2^128/value,
            and the third component is the remainder.
            */
            XeHE_NODISCARD inline auto &const_ratio() const noexcept
            {
                return const_ratio_;
            }

            /**
            Returns whether the value of the current Modulus is zero.
            */
            XeHE_NODISCARD inline bool is_zero() const noexcept
            {
                return value_ == 0;
            }

            /**
            Returns whether the value of the current Modulus is a prime number.
            */
            XeHE_NODISCARD inline bool is_prime() const noexcept
            {
                return is_prime_;
            }

            /**
            Compares two Modulus instances.
            */
            XeHE_NODISCARD inline bool operator==(const Modulus<T> &compare) const noexcept
            {
                return value_ == compare.value_;
            }

            /**
            Compares a Modulus value to an unsigned integer.
            */
            XeHE_NODISCARD inline bool operator==(T compare) const noexcept
            {
                return value_ == compare;
            }

            /**
            Compares two Modulus instances.
            */
            XeHE_NODISCARD inline bool operator!=(const Modulus<T> &compare) const noexcept
            {
                return !operator==(compare);
            }

            /**
            Compares a Modulus value to an unsigned integer.
            */
            XeHE_NODISCARD inline bool operator!=(T compare) const noexcept
            {
                return !operator==(compare);
            }

            /**
            Compares two Modulus instances.
            */
            XeHE_NODISCARD inline bool operator<(const Modulus<T> &compare) const noexcept
            {
                return value_ < compare.value_;
            }

            /**
            Compares a Modulus value to an unsigned integer.
            */
            XeHE_NODISCARD inline bool operator<(T compare) const noexcept
            {
                return value_ < compare;
            }

            /**
            Compares two Modulus instances.
            */
            XeHE_NODISCARD inline bool operator<=(const Modulus<T> &compare) const noexcept
            {
                return value_ <= compare.value_;
            }

            /**
            Compares a Modulus value to an unsigned integer.
            */
            XeHE_NODISCARD inline bool operator<=(T compare) const noexcept
            {
                return value_ <= compare;
            }

            /**
            Compares two Modulus instances.
            */
            XeHE_NODISCARD inline bool operator>(const Modulus<T> &compare) const noexcept
            {
                return value_ > compare.value_;
            }

            /**
            Compares a Modulus value to an unsigned integer.
            */
            XeHE_NODISCARD inline bool operator>(T compare) const noexcept
            {
                return value_ > compare;
            }

            /**
            Compares two Modulus instances.
            */
            XeHE_NODISCARD inline bool operator>=(const Modulus<T> &compare) const noexcept
            {
                return value_ >= compare.value_;
            }

            /**
            Compares a Modulus value to an unsigned integer.
            */
            XeHE_NODISCARD inline bool operator>=(T compare) const noexcept
            {
                return value_ >= compare;
            }
#if 0
            /**
            Returns an upper bound on the size of the Modulus, as if it was
            written to an output stream.
            */
            XeHE_NODISCARD inline std::streamoff save_size(
                compr_mode_type compr_mode = Serialization::compr_mode_default) const
            {
                std::size_t members_size = Serialization::ComprSizeEstimate(util::add_safe(sizeof(value_)), compr_mode);

                return util::safe_cast<std::streamoff>(util::add_safe(sizeof(Serialization::SEALHeader), members_size));
            }

            /**
            Saves the Modulus to an output stream. The output is in binary format
            and not human-readable. The output stream must have the "binary" flag set.
            */
            inline std::streamoff save(
                std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
            {
                using namespace std::placeholders;
                return Serialization::Save(
                    std::bind(&Modulus::save_members, this, _1), save_size(compr_mode_type::none), stream, compr_mode,
                    false);
            }

            /**
            Loads a Modulus from an input stream overwriting the current Modulus.
            */
            inline std::streamoff load(std::istream &stream)
            {
                using namespace std::placeholders;
                return Serialization::Load(std::bind(&Modulus::load_members, this, _1, _2), stream, false);
            }

            /**
            Saves the Modulus to a given memory location. The output is in binary
            format and not human-readable.
            */
            inline std::streamoff save(
                seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
            {
                using namespace std::placeholders;
                return Serialization::Save(
                    std::bind(&Modulus::save_members, this, _1), save_size(compr_mode_type::none), out, size, compr_mode,
                    false);
            }

            /**
            Loads a Modulus from a given memory location overwriting the current
            Modulus.
            */
            inline std::streamoff load(const seal_byte *in, std::size_t size)
            {
                using namespace std::placeholders;
                return Serialization::Load(std::bind(&Modulus::load_members, this, _1, _2), in, size, false);
            }

            /**
            Reduces a given unsigned integer modulo this modulus.
            */
            XeHE_NODISCARD std::uint64_t reduce(std::uint64_t value) const;
#endif

        private:
            // void set_value(T value);
            void set_value(T value)
            {
                if (value == 0)
                {
                    // Zero settings
                    bit_count_ = 0;
                    uint_count_ = 1;
                    value_ = 0;
                    const_ratio_ = { { 0, 0, 0 } };
                    is_prime_ = false;
                }
                else if ((value >> xehe::native::get_mod_bit_count_max<T>() != 0) || (value == 1))
                {
                    const std::string message = "Value can be at most " + std::to_string(get_mod_bit_count_max<T>()) + "-bit and cannot be 1";
                    throw std::invalid_argument(message);
                }
                else
                {
                    // All normal, compute const_ratio and set everything
                    value_ = value;
                    bit_count_ = xehe::native::get_significant_bit_count<T>(value_);

                    // Compute Barrett ratios
                    T numerator[3]{ 0, 0, 1 };
                    T quotient[3]{ 0, 0, 0 };

                    // Use a special method to avoid using memory pool
                    xehe::native::div_uint3<T>(numerator, value_, quotient);

                    const_ratio_[0] = quotient[0];
                    const_ratio_[1] = quotient[1];

                    // We store also the remainder
                    const_ratio_[2] = numerator[0];

                    uint_count_ = 1;

                    is_prime_ = xehe::native::is_mod_prime<T>(value_, const_ratio_.data());
                }
            }
   
 #if 0
            void save_members(std::ostream &stream) const;

            void load_members(std::istream &stream, SEALVersion version);
#endif

            T value_ = 0;

            std::array<T, 3> const_ratio_{ { 0, 0, 0 } };

            std::size_t uint_count_ = 0;

            int bit_count_ = 0;

            bool is_prime_ = false;
        };

        template<typename T>
        std::vector<Modulus<T>> get_primes(size_t ntt_size, int bit_size, size_t count)
        {
#ifdef XeHE_DEBUG
            if (!count)
            {
                throw std::invalid_argument("count must be positive");
            }
            if ((int)(std::ilogb(ntt_size) + 0.5) < 0)
            {
                throw std::invalid_argument("ntt_size must be a power of two");
            }
            if (bit_size > xehe::native::get_mod_bit_count_max<T>() || bit_size < xehe::native::get_mod_bit_count_min<T>())
            {
                throw std::invalid_argument("bit_size is invalid");
            }
#endif
            std::vector<Modulus<T>> destination;
            // T factor = xehe::native::mul_safe(uint64_t(2), static_cast<uint64_t>(ntt_size));
            T factor = T(2) * static_cast<T>(ntt_size);

            // Start with 2^bit_size - 2 * ntt_size + 1
            T value = T(0x1) << bit_size;
            try
            {
                value = xehe::native::sub_safe<T>(value, factor) + 1;
            }
            catch (const std::logic_error &)
            {
                throw std::logic_error("failed to find enough qualifying primes");
            }

            T lower_bound = T(0x1) << (bit_size - 1);
            while (count > 0 && value > lower_bound)
            {                
                Modulus<T> new_mod(value);
                if (new_mod.is_prime())
                {
                    destination.emplace_back(std::move(new_mod));
                    count--;
                }
                value -= factor;
            }
            if (count > 0)
            {
                throw std::logic_error("failed to find enough qualifying primes");
            }
            return destination;
        }
    

                       
    } //namespace native  
} // namespace xehe
#endif /* XeHE_MODULUS_H */
