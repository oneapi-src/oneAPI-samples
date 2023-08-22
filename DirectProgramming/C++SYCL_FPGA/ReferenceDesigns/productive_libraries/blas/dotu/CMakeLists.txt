cmake_minimum_required(VERSION 3.10)

project(dotu)
set(KERNEL             "dotu")
set(PRECISIONS         "c" "z")

get_filename_component(CURRENT_KERNEL_PATH        "${CMAKE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(RECONFIGURABLE_DOTPROD_PATH "${CURRENT_KERNEL_PATH}/../reconfigurable_dotprod"  ABSOLUTE)

# Create targets for generating OneAPI files, reporting of resource usage, and synthesizing bitstreams.
include(${RECONFIGURABLE_DOTPROD_PATH}/CMakeLists.txt)
