# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TEST
    TARGET fig_9_4_naive_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_9_4_naive_matrix_multiplication.cpp)

add_book_sample(
    TEST
    TARGET fig_9_7_local_accessors
    SOURCES fig_9_7_local_accessors.cpp)

add_book_sample(
    TEST
    TARGET fig_9_8_ndrange_tiled_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_9_8_ndrange_tiled_matrix_multiplication.cpp)

add_book_sample(
    TEST
    TARGET fig_9_9_local_hierarchical
    SOURCES fig_9_9_local_hierarchical.cpp)

add_book_sample(
    TEST
    TARGET fig_9_10_hierarchical_tiled_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_9_10_hierarchical_tiled_matrix_multiplication.cpp)

add_book_sample(
    TEST
    TARGET fig_9_11_sub_group_barrier
    SOURCES fig_9_11_sub_group_barrier.cpp)

add_book_sample(
    TEST
    TARGET fig_9_13_matrix_multiplication_broadcast
    SOURCES matrix_multiplication_harness.cpp fig_9_13_matrix_multiplication_broadcast.cpp)

add_book_sample(
    TARGET fig_9_14_ndrange_sub_group_matrix_multiplication
    SOURCES matrix_multiplication_harness.cpp fig_9_14_ndrange_sub_group_matrix_multiplication.cpp)
