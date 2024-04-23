//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <chrono>

class timer
{
public:
    timer() { start(); }
    void start() { t1_ = std::chrono::steady_clock::now(); }
    void stop() { t2_ = std::chrono::steady_clock::now(); }
    auto duration() { return std::chrono::duration<double>(t2_ - t1_).count(); }
private:
    std::chrono::steady_clock::time_point t1_, t2_;
};
