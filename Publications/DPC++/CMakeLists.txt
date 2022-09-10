# Copyright (c) 2020, Intel Corporation
# SPDX-License-Identifier: MIT
# https://opensource.org/licenses/MIT

##############################################################################
# This top-level CMakeLists.txt file does not define a project or setup any
# project-specific build configuration variables. That must be done in each
# sample project's project-specific CMakeLists.txt file. It does define some
# general, global options for compiling and linking.
#
# This CMakeLists.txt file finds those subdirectories located immediately
# below it that contain a CMakeLists.txt file, and adds them to its CMake
# add_subdirectory() list. All other folders and files are ignored.
#
# A build file (e.g., Ninja, Makefile, etc.) is generated as a result of
# running this CMakeLists.txt file. That build file will include targets
# defined by the project-level CMakeLists.txt files located during the
# add_subdirectory() scan. To see those build targets, type "make help"
# at the command-line.
##############################################################################

cmake_minimum_required(VERSION 3.4)
# cmake_minimum_required(VERSION 3.20)

# set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")

if(WIN32)
  #set(CMAKE_C_COMPILER clang-cl)
  #set(CMAKE_CXX_COMPILER dpcpp)
  set(CMAKE_C_COMPILER icx)
  set(CMAKE_CXX_COMPILER dpcpp)
  set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib <OBJECTS> /out:<TARGET>")
  set(CMAKE_CXX_LINK_EXECUTABLE "lld-link <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif(WIN32)
if(UNIX)
  set(CMAKE_C_COMPILER icx)
  set(CMAKE_CXX_COMPILER dpcpp)
  # set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib <OBJECTS> /out:<TARGET>")
  # set(CMAKE_CXX_LINK_EXECUTABLE "ld.lld <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif(UNIX)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(DPCPPSamples)
option(NODPL "Don't build samples requiring an installation of oneDPL and its dependencies")

enable_testing()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()


# Define project parameters for individual sample dirs.

project(DPCPPSamples)

function(add_book_sample)

    set(options TEST)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES LIBS)
    cmake_parse_arguments(BOOK_SAMPLE
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    add_executable(${BOOK_SAMPLE_TARGET} ${BOOK_SAMPLE_SOURCES})

    #target_compile_options(${BOOK_SAMPLE_TARGET} PRIVATE -fsycl -fsycl-unnamed-lambda -ferror-limit=1 -Wall -Wpedantic -Wno-pass-failed)
    target_compile_options(${BOOK_SAMPLE_TARGET} PRIVATE -fsycl -fsycl-unnamed-lambda -ferror-limit=1 -Wall -Wpedantic -Wno-pass-failed -Wno-unused-variable)

    target_link_libraries(${BOOK_SAMPLE_TARGET} PRIVATE sycl -fsycl)
    target_link_libraries(${BOOK_SAMPLE_TARGET} PRIVATE ${BOOK_SAMPLE_LIBS})

    if(CMAKE_CONFIGURATION_TYPES)
        set(BOOK_SAMPLE_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(BOOK_SAMPLE_CONFIGS ${CMAKE_BUILD_TYPE})
    endif()
    foreach(CONFIG ${BOOK_SAMPLE_CONFIGS})
        install(TARGETS ${BOOK_SAMPLE_TARGET} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
    endforeach()
    if(BOOK_SAMPLE_TEST)
        add_test(NAME ${BOOK_SAMPLE_TARGET} COMMAND ${BOOK_SAMPLE_TARGET})
    endif()
endfunction()


# Locate all folders in the passed array that contain a CMakeLists.txt file.
# Add all that are found to the CMake add_subdirectory() list.

function(find_sample_dirs)
  foreach(dir ${ARGN})
    if(IS_DIRECTORY ${dir})
      if(EXISTS "${dir}/CMakeLists.txt")
        message(STATUS "-- adding project dir: " ${dir})
        add_subdirectory(${dir})
      endif()
    endif()
  endforeach()
endfunction(find_sample_dirs)

# Make a list of all folders that contain samples and add them to
# the CMake add_subdirectory() list (see function defined above).

file(GLOB sample_dirs ${CMAKE_CURRENT_SOURCE_DIR}/*)
find_sample_dirs(${sample_dirs})

# When done here, CMake will finish the task at hand by recursing through
# all the folders that were added to the add_subdirectory() list and
# parsing the CMakeLists.txt files found in those folders.
