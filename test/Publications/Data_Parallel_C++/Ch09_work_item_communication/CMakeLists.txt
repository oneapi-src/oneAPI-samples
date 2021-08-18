# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TEST
    TARGET fig_9_4
    SOURCES matrix_multiplication_harness.cpp fig_9_4.cpp)

add_book_sample(
    TEST
    TARGET fig_9_7_local_accessors
    SOURCES fig_9_7_local_accessors.cpp)

add_book_sample(
    TEST
    TARGET fig_9_8
    SOURCES matrix_multiplication_harness.cpp fig_9_8.cpp)

add_book_sample(
    TEST
    TARGET fig_9_9_local_hierarchical
    SOURCES fig_9_9_local_hierarchical.cpp)

add_book_sample(
    TEST
    TARGET fig_9_10
    SOURCES matrix_multiplication_harness.cpp fig_9_10.cpp)

add_book_sample(
    TEST
    TARGET fig_9_11_sub_group_barrier
    SOURCES fig_9_11_sub_group_barrier.cpp)

add_book_sample(
    TEST
    TARGET fig_9_13
    SOURCES matrix_multiplication_harness.cpp fig_9_13.cpp)

add_book_sample(
    TARGET fig_9_14
    SOURCES matrix_multiplication_harness.cpp fig_9_14.cpp)
