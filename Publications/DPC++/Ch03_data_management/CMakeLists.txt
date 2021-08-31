# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap3)
add_book_sample(
    TEST
    TARGET fig_3_4_usm_explicit_data_movement
    SOURCES fig_3_4_usm_explicit_data_movement.cpp)

add_book_sample(
    TEST
    TARGET fig_3_5_usm_implicit_data_movement
    SOURCES fig_3_5_usm_implicit_data_movement.cpp)

add_book_sample(
    TEST
    TARGET fig_3_6_buffers_and_accessors
    SOURCES fig_3_6_buffers_and_accessors.cpp)

add_book_sample(
    TEST
    TARGET fig_3_10_in_order
    SOURCES fig_3_10_in_order.cpp)

add_book_sample(
    TEST
    TARGET fig_3_11_depends_on
    SOURCES fig_3_11_depends_on.cpp)

add_book_sample(
    TEST
    TARGET fig_3_13_read_after_write
    SOURCES fig_3_13_read_after_write.cpp)

add_book_sample(
    TEST
    TARGET fig_3_15_write_read_and_write_write
    SOURCES fig_3_15_write_read_and_write_write.cpp)
