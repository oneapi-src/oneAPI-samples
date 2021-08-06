# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap19)

add_book_sample(
    TARGET fig_19_3_data_race
    SOURCES fig_19_3_data_race.cpp)

add_book_sample(
    TEST
    TARGET fig_19_6_avoid_data_race_with_barrier
    SOURCES fig_19_6_avoid_data_race_with_barrier.cpp)

add_book_sample(
    TEST
    TARGET fig_19_7_avoid_data_race_with_atomics
    SOURCES fig_19_7_avoid_data_race_with_atomics.cpp)

add_book_sample(
    TEST
    TARGET fig_19_15_buffer_and_atomic_ref
    SOURCES fig_19_15_buffer_and_atomic_ref.cpp)

add_book_sample(
    TEST
    TARGET fig_19_16_atomic_accessor
    SOURCES fig_19_16_atomic_accessor.cpp)

add_book_sample(
    TEST
    TARGET fig_19_18_histogram
    SOURCES fig_19_18_histogram.cpp)

add_book_sample(
    TEST
    TARGET fig_19_19-20_device_latch
    SOURCES fig_19_19-20_device_latch.cpp)
