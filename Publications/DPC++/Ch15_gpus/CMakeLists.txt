# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap15)
add_book_sample(
    TEST
    TARGET fig_15_3_single_task_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_15_3_single_task_matrix_multiplication.cpp)

add_book_sample(
    TEST
    TARGET fig_15_5_somewhat_parallel_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_15_5_somewhat_parallel_matrix_multiplication.cpp)

add_book_sample(
    TEST
    TARGET fig_15_7_more_parallel_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_15_7_more_parallel_matrix_multiplication.cpp)

add_book_sample(
    TEST
    TARGET fig_15_10_divergent_control_flow
    SOURCES fig_15_10_divergent_control_flow.cpp)

add_book_sample(
    TEST
    TARGET fig_15_12_small_work_group_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_15_12_small_work_group_matrix_multiplication.cpp)

add_book_sample(
    TEST
    TARGET fig_15_18_columns_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_15_18_columns_matrix_multiplication.cpp)
