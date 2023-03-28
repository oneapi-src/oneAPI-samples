# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: MIT

if(WIN32)
  set(CMAKE_CXX_COMPILER icx-cl)
endif(WIN32)
if(UNIX)
  set(CMAKE_CXX_COMPILER icpx)
endif(UNIX)
