# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap11)
add_book_sample(
    TEST
    TARGET fig_11_6_load_store
    SOURCES fig_11_6_load_store.cpp)

add_book_sample(
    TEST
    TARGET fig_11_7_swizzle_vec
    SOURCES fig_11_7_swizzle_vec.cpp)

add_book_sample(
    TEST
    TARGET fig_11_8_vector_exec
    SOURCES fig_11_8_vector_exec.cpp)

