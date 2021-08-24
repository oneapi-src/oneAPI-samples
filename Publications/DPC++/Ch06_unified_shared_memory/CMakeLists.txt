# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap6)
add_book_sample(
    TEST
    TARGET fig_6_5_allocation_styles
    SOURCES fig_6_5_allocation_styles.cpp)

add_book_sample(
    TEST
    TARGET fig_6_6_usm_explicit_data_movement
    SOURCES fig_6_6_usm_explicit_data_movement.cpp)

add_book_sample(
    TEST
    TARGET fig_6_7_usm_implicit_data_movement
    SOURCES fig_6_7_usm_implicit_data_movement.cpp)

add_book_sample(
    TEST
    TARGET fig_6_8_prefetch_memadvise
    SOURCES fig_6_8_prefetch_memadvise.cpp)

add_book_sample(
    TEST
    TARGET fig_6_9_queries
    SOURCES fig_6_9_queries.cpp)
