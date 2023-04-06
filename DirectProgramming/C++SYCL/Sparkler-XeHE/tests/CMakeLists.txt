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

SET (TEST_SOURCES
    tests_catch.cpp
    tests_withHEAAN_cpu.cpp
    tests_gpu/evaluator_seal.cpp
    tests_gpu/mul_ntv.cpp
    tests_gpu/add_ntv.cpp
    tests_gpu/polyops_ntv.cpp
        
    ${SRC_INCLUDE_DIR}/../../include/XeHE.hpp

    )



add_executable (tests
        ${TEST_SOURCES}
        )

if(BUILD_WITH_IGPU)
    add_sycl_to_target(
            TARGET tests
            SOURCES ${TEST_SOURCES}
    )
endif(BUILD_WITH_IGPU)

target_include_directories(
    tests
    PRIVATE "${CMAKE_SOURCE_DIR}/external" "${CMAKE_SOURCE_DIR}/src/include"
)

target_include_directories(
        tests
        PRIVATE "${CMAKE_SOURCE_DIR}/src/include")

target_link_libraries(tests
        PRIVATE ${TEST_LINK_OPTS}
        PRIVATE XeHE
        )

# -- enable later
#target_link_libraries(tests ${OUR_LIBS})

if (BUILD_WITH_SEAL)

    add_compile_definitions(SEAL_USE_INTEL_XEHE)
#    message("find package paths: ${CMAKE_CURRENT_BINARY_DIR}/../external/SEAL/cmake ")
#    message("find package paths: ${PROJECT_SOURCE_DIR}/../external/SEAL/cmake ")
    # This is no longer necessary. Note that find_package is usually used to find installed files,
    # and here we just need to locate and make build tree, so find_package is an overkill.
    #
    #find_package (SEAL 3.6.1 EXACT REQUIRED
    #        # Providing a path so this can be built without installing Microsoft SEAL
    #        #PATHS ${PROJECT_SOURCE_DIR}/../external/SEAL/native/src/cmake
    #        # It is assumed that external dir is in the parent dir
    #        #PATHS ${CMAKE_CURRENT_BINARY_DIR}/../external/SEAL/cmake
    #        PATHS ${PROJECT_SOURCE_DIR}/../external/SEAL/cmake
    #        )

    # append SEAL-specific includes
    target_include_directories(
            tests
            PRIVATE "${CMAKE_SOURCE_DIR}/external/SEAL/native/src/seal"
    )

    # append SEAL-specific link options
    add_dependencies(tests seal)
    target_link_libraries(tests
            PUBLIC seal)
endif (BUILD_WITH_SEAL)

if (BUILD_WITH_HEAAN)
    # append HEAAN-specific includes
    target_include_directories(
            tests
            PRIVATE "${CMAKE_SOURCE_DIR}/external/HEAAN/HEAAN/src"
    )

    # set HEAAN-specific link options
    set(TEST_LINK_OPTS
            "-lntl;-lgmp;-lm"
            )

    # append HEAAN-specific link options
    target_link_libraries(tests
            PRIVATE ${TEST_LINK_OPTS}
            PRIVATE HEAAN_lib)

endif (BUILD_WITH_HEAAN)
