## Copyright 2009 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(libType STATIC)
add_library(imgui ${libType}
  imgui.cpp
  imgui_draw.cpp
  imgui_widgets.cpp
  imgui_tables.cpp
  imgui_piemenu.cpp
)

target_include_directories(imgui
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)
