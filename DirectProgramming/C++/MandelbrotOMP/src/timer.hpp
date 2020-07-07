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

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

using namespace std::chrono;

class CUtilTimer {
 public:
  // Registers the current clock tick and time value in m_start_clock_tick and
  // m_start_time
  void start();
  // Registers the current clock tick and time value in m_end_clock_tick and
  // m_end_time
  void stop();
  // Returns the number of seconds taken between start and stop
  double get_time();

 private:
  // start and end time
  high_resolution_clock::time_point m_start_time, m_end_time;
};

#endif  // TIMER_H
