# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

FetchContent_Declare(
    xehe
#    GIT_REPOSITORY ssh://git@gitlab.devtools.intel.com:29418/alyashev/XeHE.git
#    GIT_TAG 2aac4e7c
)
FetchContent_GetProperties(xehe)

if(NOT xehe_POPULATED)
    FetchContent_Populate(xehe)

    set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "" FORCE)
    set(BUILD_WITH_SEAL OFF CACHE BOOL "" FORCE)
    set(BUILD_WITH_HEAAN OFF CACHE BOOL "" FORCE)
    set(SEAL_USE_INTEL_GPU OFF CACHE BOOL "" FORCE)
    set(BUILD_WITH_IGPU ON CACHE BOOL "" FORCE)
    set(EXCLUDE_FROM_ALL TRUE)

    mark_as_advanced(BUILD_INTEL_XEHE)
    mark_as_advanced(INSTALL_INTEL_XEHE)
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_INTEL_XEHE)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_INTEL_XEHE)

    message(>>>XeHE source directory: ${xehe_SOURCE_DIR})
    add_subdirectory(
	${xehe_SOURCE_DIR}
    	EXCLUDE_FROM_ALL
    )
endif()
