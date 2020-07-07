//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2013 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
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
