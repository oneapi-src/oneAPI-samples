# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap4)
add_book_sample(
    TEST
    TARGET fig_4_5_vector_add
    SOURCES fig_4_5_vector_add.cpp)

add_book_sample(
    TEST
    TARGET fig_4_6_matrix_add
    SOURCES fig_4_6_matrix_add.cpp)

add_book_sample(
    TEST
    TARGET fig_4_7_basic_matrix_multiply
    SOURCES fig_4_7_basic_matrix_multiply.cpp)

add_book_sample(
    TEST
    TARGET fig_4_13_nd_range_matrix_multiply
    SOURCES fig_4_13_nd_range_matrix_multiply.cpp)

add_book_sample(
    TEST
    TARGET fig_4_20_hierarchical_matrix_multiply
    SOURCES fig_4_20_hierarchical_matrix_multiply.cpp)

add_book_sample(
    TEST
    TARGET fig_4_22_hierarchical_logical_matrix_multiply
    SOURCES fig_4_22_hierarchical_logical_matrix_multiply.cpp)
