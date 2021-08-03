# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap12)
add_book_sample(
    TEST
    TARGET fig_12_1_assigned_device
    SOURCES fig_12_1_assigned_device.cpp)

add_book_sample(
    TEST
    TARGET fig_12_2_try_catch
    SOURCES fig_12_2_try_catch.cpp)

add_book_sample(
    TEST
    TARGET fig_12_3_device_selector
    SOURCES fig_12_3_device_selector.cpp)

add_book_sample(
    TEST
    TARGET fig_12_4_curious
    SOURCES fig_12_4_curious.cpp)

add_book_sample(
    TEST
    TARGET fig_12_6_very_curious
    SOURCES fig_12_6_very_curious.cpp)

add_book_sample(
    TEST
    TARGET fig_12_7_invocation_parameters
    SOURCES fig_12_7_invocation_parameters.cpp)
