// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "omp_utils.h"

namespace intel {
namespace he {
namespace heseal {

class OMPUtilitiesS : public OMPUtilities {
 public:
  static const int MaxThreads;

  /**
   * @brief Sets number of threads to assign at specified nesting level.
   * @param level[in] OpenMP nesting level.
   * @param threads[in] Number of threads to assign at specified nesting level.
   */
  static void setThreadsAtLevel(int level, int threads);
  /**
   * @brief Retrieves threads at specified level.
   * @param level[in] OpenMP nesting level.
   * @return Number of threads to assign at specified nesting level.
   */
  static int getThreadsAtLevel(int level);
  /**
   * @brief Retrieves threads to assign to current OpenMP nesting level.
   */
  static int getThreadsAtLevel();

 private:
  static std::vector<int> m_threads_per_level;
};

}  // namespace heseal
}  // namespace he
}  // namespace intel
