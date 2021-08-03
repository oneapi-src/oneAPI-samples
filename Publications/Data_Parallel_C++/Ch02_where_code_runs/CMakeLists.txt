# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap2)
add_book_sample(
    TEST
    TARGET fig_2_2_simple_program
    SOURCES fig_2_2_simple_program.cpp)

add_book_sample(
    TEST
    TARGET fig_2_7_implicit_default_selector
    SOURCES fig_2_7_implicit_default_selector.cpp)

add_book_sample(
    TEST
    TARGET fig_2_9_host_selector
    SOURCES fig_2_9_host_selector.cpp)

add_book_sample(
    TEST
    TARGET fig_2_10_cpu_selector
    SOURCES fig_2_10_cpu_selector.cpp)

add_book_sample(
    TARGET fig_2_12_multiple_selectors
    SOURCES fig_2_12_multiple_selectors.cpp)

add_book_sample(
    TARGET fig_2_13_gpu_plus_fpga
    SOURCES fig_2_13_gpu_plus_fpga.cpp)

add_book_sample(
    TARGET fig_2_15_custom_selector
    SOURCES fig_2_15_custom_selector.cpp)

add_book_sample(
    TEST
    TARGET fig_2_18_simple_device_code
    SOURCES fig_2_18_simple_device_code.cpp)

add_book_sample(
    TEST
    TARGET fig_2_22_simple_device_code_2
    SOURCES fig_2_22_simple_device_code_2.cpp)

add_book_sample(
    TEST
    TARGET fig_2_23_fallback
    SOURCES fig_2_23_fallback.cpp)
