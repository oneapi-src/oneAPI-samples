# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap13)
add_book_sample(
    TEST
    TARGET fig_13_4_stream
    SOURCES fig_13_4_stream.cpp)

add_book_sample(
    TEST
    TARGET fig_13_6_common_buffer_pattern
    SOURCES fig_13_6_common_buffer_pattern.cpp)

add_book_sample(
    TEST
    TARGET fig_13_7_common_pattern_bug
    SOURCES fig_13_7_common_pattern_bug.cpp)

add_book_sample(
    TEST
    TARGET fig_13_8_host_accessor
    SOURCES fig_13_8_host_accessor.cpp)

add_book_sample(
    TEST
    TARGET fig_13_9_host_accessor_for_init
    SOURCES fig_13_9_host_accessor_for_init.cpp)

add_book_sample(
    TARGET fig_13_10_host_accessor_deadlock
    SOURCES fig_13_10_host_accessor_deadlock.cpp)

