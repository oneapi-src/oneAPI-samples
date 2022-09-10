# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap20)

add_book_sample(
    TARGET fig_ep_1_mdspan
    SOURCES fig_ep_1_mdspan.cpp)

add_book_sample(
    TEST
    TARGET fig_ep_2-4_generic_space
    SOURCES fig_ep_2-4_generic_space.cpp)

add_book_sample(
    TEST
    TARGET fig_ep_5_extension_mechanism
    SOURCES fig_ep_5_extension_mechanism.cpp)

add_book_sample(
    TEST
    TARGET fig_ep_6_device_constexpr
    SOURCES fig_ep_6_device_constexpr.cpp)

add_book_sample(
    TEST
    TARGET fig_ep_7_hierarchical_reduction
    SOURCES fig_ep_7_hierarchical_reduction.cpp)

