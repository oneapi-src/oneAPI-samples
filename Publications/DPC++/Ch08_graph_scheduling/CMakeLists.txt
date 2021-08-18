# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap8)
add_book_sample(
    TEST
    TARGET fig_8_3_linear_dependence_in_order
    SOURCES fig_8_3_linear_dependence_in_order.cpp)

add_book_sample(
    TEST
    TARGET fig_8_4_linear_dependence_events
    SOURCES fig_8_4_linear_dependence_events.cpp)

add_book_sample(
    TEST
    TARGET fig_8_5_linear_dependence_buffers
    SOURCES fig_8_5_linear_dependence_buffers.cpp)

add_book_sample(
    TEST
    TARGET fig_8_6_y_in_order
    SOURCES fig_8_6_y_in_order.cpp)

add_book_sample(
    TEST
    TARGET fig_8_7_y_events
    SOURCES fig_8_7_y_events.cpp)

add_book_sample(
    TEST
    TARGET fig_8_8_y_buffers
    SOURCES fig_8_8_y_buffers.cpp)


