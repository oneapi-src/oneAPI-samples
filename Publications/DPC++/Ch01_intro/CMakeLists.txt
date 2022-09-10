# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap1)

add_book_sample(
    TEST
    TARGET fig_1_1_hello
    SOURCES fig_1_1_hello.cpp)

add_book_sample(
    TEST
    TARGET fig_1_3_race
    SOURCES fig_1_3_race.cpp)

add_book_sample(
    TEST
    TARGET fig_1_4_lambda
    SOURCES fig_1_4_lambda.cpp)

add_book_sample(
    TEST
    TARGET fig_1_6_functor
    SOURCES fig_1_6_functor.cpp)
