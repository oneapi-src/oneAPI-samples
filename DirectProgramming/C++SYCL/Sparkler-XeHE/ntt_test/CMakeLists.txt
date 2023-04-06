CMAKE_MINIMUM_REQUIRED (VERSION 3.13)
project (tests VERSION 0.1 LANGUAGES CXX)

set (cmake_cxx_standard 17)
set (cmake_cxx_standard_required on)

# debug cmake and make
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_VERBOSE_MAKEFILE ON)

#include_directories (${CMAKE_CURRENTS_SOURCE_DIR}/externals/catch2)
set (SRC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/include")

if(VERBOSE_TEST_FLAG)
    message("VERBOSE TESTS -- will printout debug information")
    add_compile_definitions(VERBOSE_TEST)
endif()
set(NTT_COMPILE_OPTS
        "-MMD;-Wall;-O3;-pthread"
        )

set(NTT_TEST_LINK_OPTS
            "-lm;-lpthread"
            )
SET (NTT_TEST_SOURCES
        driver.cpp
        run_driver.cpp
        main.cpp
        include/driver.h
        ${SRC_INCLUDE_DIR}/../../include/XeHE.hpp
    )

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)


add_executable (ntt_test
        ${NTT_TEST_SOURCES}
        )

     

if(BUILD_WITH_IGPU)
    add_sycl_to_target(
            TARGET ntt_test
            SOURCES ${NTT_TEST_SOURCES}
    )
endif(BUILD_WITH_IGPU)

target_include_directories(
    ntt_test
    PRIVATE "${CMAKE_SOURCE_DIR}/external" "${CMAKE_SOURCE_DIR}/src/include"
)

target_include_directories(
        ntt_test
        PRIVATE "${CMAKE_SOURCE_DIR}/src/include")

target_compile_options(
      ntt_test
      PRIVATE ${NTT_COMPILE_OPTS}
)   

target_link_libraries(ntt_test
        PRIVATE Threads::Threads
        PRIVATE ${NTT_TEST_LINK_OPTS}
        PRIVATE XeHE
        )

# -- enable later
#target_link_libraries(tests ${OUR_LIBS})

if (BUILD_WITH_SEAL)

    add_compile_definitions(SEAL_USE_INTEL_XEHE)

    # append SEAL-specific includes
    target_include_directories(
            ntt_test
            PRIVATE "${CMAKE_SOURCE_DIR}/external/SEAL/native/src/seal"
    )

    # append SEAL-specific link options
    add_dependencies(ntt_test seal)
    target_link_libraries(ntt_test
            PUBLIC seal)
endif (BUILD_WITH_SEAL)
