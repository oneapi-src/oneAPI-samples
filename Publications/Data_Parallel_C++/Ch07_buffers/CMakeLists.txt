# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap7)
add_book_sample(
    TEST
    TARGET fig_7_2_3_4_creating_buffers
    SOURCES fig_7_2_3_4_creating_buffers.cpp)

add_book_sample(
    TEST
    TARGET fig_7_5_buffer_properties
    SOURCES fig_7_5_buffer_properties.cpp)

add_book_sample(
    TEST
    TARGET fig_7_8_accessors_simple
    SOURCES fig_7_8_accessors_simple.cpp)

add_book_sample(
    TEST
    TARGET fig_7_10_accessors
    SOURCES fig_7_10_accessors.cpp)
