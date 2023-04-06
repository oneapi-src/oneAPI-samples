// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "seal_omp_utils.h"

#include <omp.h>

namespace intel {
namespace he {
namespace heseal {

const int OMPUtilitiesS::MaxThreads = omp_get_max_threads();
std::vector<int> OMPUtilitiesS::m_threads_per_level(1,
                                                    OMPUtilitiesS::MaxThreads);

void OMPUtilitiesS::setThreadsAtLevel(int level, int threads) {
  if (threads < 1) threads = 1;
  std::size_t level_i = static_cast<std::size_t>(level < 0 ? 0 : level);
  if (level_i >= m_threads_per_level.size())
    m_threads_per_level.resize(level_i + 1, 1);
  m_threads_per_level[level_i] = threads;
}

int OMPUtilitiesS::getThreadsAtLevel() {
#if WIN32
    return getThreadsAtLevel(1);
#else
  return getThreadsAtLevel(omp_get_active_level());
#endif
}

int OMPUtilitiesS::getThreadsAtLevel(int level) {
  int retval = 1;
  std::size_t level_i = static_cast<std::size_t>(level < 0 ? 0 : level);
  if (level_i < m_threads_per_level.size()) {
    retval = m_threads_per_level[level_i];
  }
  return retval;
}

}  // namespace heseal
}  // namespace he
}  // namespace intel
