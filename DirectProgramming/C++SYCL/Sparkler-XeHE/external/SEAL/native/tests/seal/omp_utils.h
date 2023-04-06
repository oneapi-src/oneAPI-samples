// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace intel {
namespace he {

class OMPUtilities {
 public:
  static int assignOMPThreads(int& remaining_threads, int requested_threads) {
    int retval = (requested_threads > 0 ? requested_threads : 1);
    if (retval > remaining_threads) retval = remaining_threads;
    if (retval > 1)
      remaining_threads -= retval;
    else
      retval = 1;
    return retval;
  }
};

}  // namespace he
}  // namespace intel
