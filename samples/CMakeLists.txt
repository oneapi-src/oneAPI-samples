# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)
project(DPCPPSamples)

function(add_book_sample)

    set(options TEST)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES LIBS ADDITIONAL_COMPILE_OPTIONS)
    cmake_parse_arguments(BOOK_SAMPLE
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    add_executable(${BOOK_SAMPLE_TARGET} ${BOOK_SAMPLE_SOURCES})

    target_compile_options(${BOOK_SAMPLE_TARGET} PRIVATE -fsycl -fsycl-unnamed-lambda -ferror-limit=1 -Wall -Wpedantic ${BOOK_SAMPLE_ADDITIONAL_COMPILE_OPTIONS})

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

add_subdirectory( Ch01_intro )
add_subdirectory( Ch02_where_code_runs )
add_subdirectory( Ch03_data_management )
add_subdirectory( Ch04_expressing_parallelism )
add_subdirectory( Ch05_error_handling )
add_subdirectory( Ch06_unified_shared_memory )
add_subdirectory( Ch07_buffers )
add_subdirectory( Ch08_graph_scheduling )
add_subdirectory( Ch09_work_item_communication )
add_subdirectory( Ch10_expressing_kernels )
add_subdirectory( Ch11_vectors )
add_subdirectory( Ch12_device_information )
add_subdirectory( Ch13_practical_tips )
add_subdirectory( Ch14_common_parallel_patterns )
add_subdirectory( Ch15_gpus )
add_subdirectory( Ch16_cpus )
add_subdirectory( Ch17_fpgas )
add_subdirectory( Ch18_using_libs )
add_subdirectory( Ch19_memory_model_and_atomics )
