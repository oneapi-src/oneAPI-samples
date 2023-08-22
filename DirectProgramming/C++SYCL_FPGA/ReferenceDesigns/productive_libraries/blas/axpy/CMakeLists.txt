cmake_minimum_required(VERSION 3.10)

project(axpy)
set(KERNEL             "axpy")
set(PRECISIONS         "s" "d" "c" "z")

get_filename_component(CURRENT_KERNEL_PATH        "${CMAKE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(RECONFIGURABLE_VECADD_PATH "${CURRENT_KERNEL_PATH}/../reconfigurable_vecadd"  ABSOLUTE)

# Create targets for generating OneAPI files, reporting of resource usage, and synthesizing bitstreams.
include(${RECONFIGURABLE_VECADD_PATH}/CMakeLists.txt)
