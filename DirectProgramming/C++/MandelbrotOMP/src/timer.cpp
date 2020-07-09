//==============================================================
//
// Copyright 2020 Intel Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// ===============================================================

#include "timer.hpp"

#include <chrono>

using namespace std::chrono;

// Description:
// Registers the current clock tick value in m_start_clock_tick, current time
// value in m_start_time Microsoft Windows* uses __rdtsc for clock ticks and
// QueryPerformanceFrequency/QueryPerformanceCounter for time Linux*/OS X* uses
// the rdtsc instruction for clock ticks and get_timeofday for time
void CUtilTimer::start() { m_start_time = high_resolution_clock::now(); }

// Description:
// Registers the current clock tick value in m_end_clock_tick, current time
// value in m_end_time Windows uses __rdtsc for clock ticks and
// QueryPerformanceFrequency/QueryPerformanceCounter for time Linux*/OS X* uses
// the rdtsc instruction for clock ticks and get_timeofday for time
void CUtilTimer::stop() { m_end_time = high_resolution_clock::now(); }

// Description:
// Returns the number of seconds taken between start and stop
double CUtilTimer::get_time() {
  duration<double> time_span =
      duration_cast<duration<double> >(m_end_time - m_start_time);
  return time_span.count();
}
