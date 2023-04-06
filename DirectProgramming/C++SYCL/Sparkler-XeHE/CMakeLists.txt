CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

if (MSVC)
    cmake_policy(SET CMP0091 NEW)
endif()
project(XeHE VERSION 0.1 LANGUAGES CXX)

# DPC++ SYCL from oneAPI
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set (cmake_cxx_standard 17)
set (cmake_cxx_standard_required on)

if (MSVC)
    # ------------- this is presumably needed for SYLC on WIN
    # TODO: check this on WIN32
    # lowest version on windows GNU-style clang 3.15
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(CMAKE_CXX_STANDARD 17)
    cmake_policy(SET CMP0091 NEW)
    # TODO: change later whether it is library or binary executable
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    #add_compile_options("-U_MT" "-U_MD")
    #add_link_options("-U_MT" "-U_MD")
endif ()

# TODO: add CUDA(?)check here

## C++
include(CheckLanguage)

# ------- OPTIONS --------
OPTION(BUILD_WITH_SEAL "Building with MS SEAL" ON)
OPTION(BUILD_WITH_HEAAN "Building with HEAAN (Needs NTL and GMP)" OFF)
OPTION(BUILD_WITH_IGPU "Building with DPC++ SYCL" ON)
OPTION(SEAL_USE_INTEL_GPU "Building XeHE as SEAL Backend" OFF)
OPTION(VERBOSE_TEST_FLAG "Building with verbose tests (gpu)" OFF)
OPTION(SEALTEST_OMP_ENABLED "Building with enabled OMP test" OFF)

if (BUILD_WITH_SEAL)
    if (NOT MSVC)
        # invoke some SEAL dark magic with cmake to remove clumsy manual steps
        set(GTEST_DIR ${PROJECT_SOURCE_DIR}/external/SEAL/native/tests/thirdparty/googletest)
        if(NOT EXISTS ${GTEST_DIR}/CMakeLists.txt)
            message(STATUS "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            message(STATUS "PRE-CONFIGURE: Initializing Googletest repo. ${CMAKE_SOURCE_DIR}")
            execute_process(
                    COMMAND git submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE output
            )
            #message("RESULT: " ${result})
            #message("OUTPUT: " ${output})
        endif ()

        #message(STATUS "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #message(STATUS "PRE-CONFIGURE: Setting up cmake in ${PROJECT_SOURCE_DIR}/external/SEAL")
        #execute_process(
        #        COMMAND cmake -S . -DSEAL_BUILD_TESTS=ON -DSEAL_BUILD_EXAMPLES=ON -L 
        #        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/SEAL
        #        RESULT_VARIABLE result
        #        OUTPUT_VARIABLE output
        #)
        #message("RESULT: " ${result})
        #message("OUTPUT: " ${output})
        #message(STATUS "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #message(STATUS "PRE-CONFIURE: Setting up cmake in ${PROJECT_SOURCE_DIR}/external/SEAL/native/tests")
        #execute_process(
        #        COMMAND cmake .
        #        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/SEAL/native/tests
        #        RESULT_VARIABLE result
        #        OUTPUT_VARIABLE output
        #)
        ##message("RESULT: " ${result})
        ##message("OUTPUT: " ${output})
        #message(STATUS "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #message(STATUS "Setting up cmake in ${PROJECT_SOURCE_DIR}/external/SEAL/native/examples")
        #execute_process(
        #        COMMAND cmake .
        #        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/SEAL/native/examples
        #        RESULT_VARIABLE result
        #        OUTPUT_VARIABLE output
        #)
        ##message("RESULT: " ${result})
        ##message("OUTPUT: " ${output})
    endif ()
endif ()

if (MSVC)
    OPTION(BUILD_WITH_HEAAN "Building with HEAAN (Needs NTL and GMP)" OFF)
    message(STATUS ">>>>>> turn OFF HEAAN in WINDOWS")
endif ()

if (BUILD_WITH_IGPU)
    message(STATUS ">>>>>> building for IGPU with DPC++ SYCL")
    message(STATUS "Adding compile-time define BUILD_WITH_IGPU")
    add_compile_definitions(BUILD_WITH_IGPU)
    #
    # TODO: figure out how to set CMAKE_C and CMAKE_CXX compilers here
    #
    set(tools /opt/intel/oneapi/compiler/latest/linux/bin)
endif (BUILD_WITH_IGPU)

check_language(CXX)
if (CMAKE_CXX_COMPILER)
    enable_language(CXX)
endif ()

# ------ DEBUG FLAGS -------
# cmake and make debug info Flags
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

if (BUILD_WITH_SEAL)
    message(STATUS ">>>>>> building with SEAL: googletest has to be initialized and cmake . has been run in SEAL/native/src SEAL/native/examples")
    message(STATUS "Adding compile-time define BUILD_WITH_SEAL")
    set(ALLOW_COMMAND_LINE_BUILD ON)
    add_compile_definitions(BUILD_WITH_SEAL)

    if (SEALTEST_OMP_ENABLED)
        message(STATUS "Enable OpenMP for sealtest")
        add_compile_definitions(SEALTEST_OMP_ENABLED)
    else(SEALTEST_OMP_ENABLED)
        message(STATUS "Disable OpenMP for sealtest")
    endif(SEALTEST_OMP_ENABLED)

else (BUILD_WITH_SEAL)
    message(STATUS ">>>>>>  building WITHOUT SEAL")
endif (BUILD_WITH_SEAL)

if (BUILD_WITH_HEAAN)
    message(STATUS ">>>>>> building with HEAAN: expecting installed NTL and GMP libraries (on Linux)")
    message(STATUS "Adding compile-time define BUILD_WITH_HEAAN")
    add_compile_definitions(BUILD_WITH_HEAAN)
else ()
    message(STATUS ">>>>>>  building WITHOUT HEAAN")
endif (BUILD_WITH_HEAAN)

if (SEAL_USE_INTEL_GPU)
    message(STATUS ">>>>>> building for SEAL with XeHE Backend")
    message(STATUS "Adding compile-time define SEAL_USE_INTEL_GPU")
    add_compile_definitions(SEAL_USE_INTEL_GPU)
else ()
    message(STATUS ">>>>>>  building WITHOUT XeHE Backend")
endif (SEAL_USE_INTEL_GPU)

if (BUILD_WITH_IGPU OR SEAL_USE_INTEL_GPU OR SEAL_USE_INTEL_XEHE)
    add_subdirectory(src)
endif()

if (BUILD_WITH_SEAL OR BUILD_WITH_HEAAN OR (BUILD_WITH_IGPU AND (SEAL_USE_INTEL_GPU OR SEAL_USE_INTEL_XEHE)))
    add_subdirectory(external)
endif()

if (BUILD_WITH_SEAL AND (BUILD_WITH_IGPU AND (SEAL_USE_INTEL_GPU OR SEAL_USE_INTEL_XEHE)))
    add_subdirectory(tests)
    add_subdirectory(ntt_test)
endif()
