//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _TIMER_HPP
#define _TIMER_HPP

#include <chrono>

/** Gets the time between construction and destruction and stores it in a
 * reference result variable.
 * Create an object of this class in a scope, and when it's destructed, the
 * time is put into the constructor's parameter.
 */
template <bool enabled = true>
class scoped_timer
{
private:
    std::chrono::steady_clock::time_point t1_, t2_;
    typedef double time_point_t;
    time_point_t& res_;
public:
    scoped_timer(time_point_t& res) : res_(res)
    {
        if (enabled)
        {
            t1_ = std::chrono::steady_clock::now();
        }
    }
    ~scoped_timer()
    {
        if (enabled)
        {
            t2_ = std::chrono::steady_clock::now();
            res_ += std::chrono::duration<double>(t2_ - t1_).count();
        }
    }
};

typedef scoped_timer<> scoped_timer_t;

template <bool enabled = true>
class timer
{
public:
    timer() { start(); }
    void start() { if (enabled) t1_ = std::chrono::steady_clock::now(); }
    void stop() { if (enabled) t2_ = std::chrono::steady_clock::now(); }
    auto duration() {
        if (enabled)
            return std::chrono::duration<double>(t2_ - t1_).count();
        else
            return 0.0;
    }
private:
    std::chrono::steady_clock::time_point t1_, t2_;
};

typedef timer<> timer_type;

const bool timer_enabled = true;
const bool timer_disabled = false;

#endif // _TIMER_HPP
