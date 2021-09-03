# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap16)
add_book_sample(
    TEST
    TARGET fig_16_6_stream_triad
    SOURCES fig_16_6_stream_triad.cpp)

add_book_sample(
    TEST
    TARGET fig_16_12_forward_dep  
    SOURCES fig_16_12_forward_dep.cpp)

add_book_sample(
    TEST
    TARGET fig_16_18_vector_swizzle  
    SOURCES fig_16_18_vector_swizzle.cpp)
