## Copyright 2009 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

add_library(arcball_camera STATIC ArcballCamera.cpp)
target_link_libraries(arcball_camera PUBLIC rkcommon::rkcommon)
target_include_directories(arcball_camera INTERFACE ${CMAKE_CURRENT_LIST_DIR})
