## Copyright 2009 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

add_library(stb_image INTERFACE)
target_include_directories(stb_image
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)
