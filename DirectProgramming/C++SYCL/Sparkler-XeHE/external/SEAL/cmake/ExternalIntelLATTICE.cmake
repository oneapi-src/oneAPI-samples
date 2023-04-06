# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

FetchContent_Declare(
    intel_lattice
    PREFIX intel_lattice
    GIT_REPOSITORY ssh://git@gitlab.devtools.intel.com:29418/DBIO/glade/intel-lattice.git
    GIT_TAG 9be375c4 # Mar 1, 2021
)
FetchContent_GetProperties(intel_lattice)

if(NOT intel_lattice_POPULATED)
    FetchContent_Populate(intel_lattice)
    set(INTEL_LATTICE_DEBUG OFF) # Set to ON/OFF to toggle debugging

    set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "" FORCE)
    set(LATTICE_DEBUG ${INTEL_LATTICE_DEBUG} CACHE BOOL "" FORCE)
    set(LATTICE_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(LATTICE_EXPORT OFF CACHE BOOL "" FORCE)
    set(LATTICE_COVERAGE OFF CACHE BOOL "" FORCE)
    set(LATTICE_TESTING OFF CACHE BOOL "" FORCE)
    set(LATTICE_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(EXCLUDE_FROM_ALL TRUE)

    mark_as_advanced(BUILD_INTEL_LATTICE)
    mark_as_advanced(INSTALL_INTEL_LATTICE)
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_INTEL_LATTICE)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_INTEL_LATTICE)

    add_subdirectory(
        ${intel_lattice_SOURCE_DIR}
        EXCLUDE_FROM_ALL
    )
endif()
