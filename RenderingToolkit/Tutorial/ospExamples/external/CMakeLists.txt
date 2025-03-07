## Copyright 2009 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

ospray_disable_compiler_warnings()

# if (OSPRAY_ENABLE_APPS_EXAMPLES)
add_subdirectory(imgui)
# endif()

add_subdirectory(stb_image)
