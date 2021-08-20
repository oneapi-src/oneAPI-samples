# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap9)
add_book_sample(
    TEST
    TARGET fig_9_4_naive_matmul
    SOURCES matmul_harness.cpp fig_9_4_naive_matmul.cpp)

add_book_sample(
    TEST
    TARGET fig_9_7_local_accessors
    SOURCES fig_9_7_local_accessors.cpp)

add_book_sample(
    TEST
    TARGET fig_9_8_ndrange_tiled_matmul
    SOURCES matmul_harness.cpp fig_9_8_ndrange_tiled_matmul.cpp)

add_book_sample(
    TEST
    TARGET fig_9_9_local_hier
    SOURCES fig_9_9_local_hier.cpp)

add_book_sample(
    TEST
    TARGET fig_9_10_hier_tiled_matmul
    SOURCES matmul_harness.cpp fig_9_10_hier_tiled_matmul.cpp)

add_book_sample(
    TEST
    TARGET fig_9_11_sub_group_barrier
    SOURCES fig_9_11_sub_group_barrier.cpp)

add_book_sample(
    TEST
    TARGET fig_9_13_matmul_broadcast
    SOURCES matmul_harness.cpp fig_9_13_matmul_broadcast.cpp)

add_book_sample(
    TARGET fig_9_14_ndrange_sub_group_matmul
    SOURCES matmul_harness.cpp fig_9_14_ndrange_sub_group_matmul.cpp)
