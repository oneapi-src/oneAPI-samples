# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap10)
add_book_sample(
    TEST
    TARGET fig_10_2_kernel_lambda
    SOURCES fig_10_2_kernel_lambda.cpp)

add_book_sample(
    TEST
    TARGET fig_10_3_optional_kernel_lambda_elements
    SOURCES fig_10_3_optional_kernel_lambda_elements.cpp)

add_book_sample(
    TEST
    TARGET fig_10_4_named_kernel_lambda
    SOURCES fig_10_4_named_kernel_lambda.cpp)

add_book_sample(
    TEST
    TARGET fig_10_5_unnamed_kernel_lambda
    SOURCES fig_10_5_unnamed_kernel_lambda.cpp)

add_book_sample(
    TEST
    TARGET fig_10_6_kernel_functor
    SOURCES fig_10_6_kernel_functor.cpp)

add_book_sample(
    TEST
    TARGET fig_10_8_opencl_object_interop
    SOURCES fig_10_8_opencl_object_interop.cpp
    LIBS OpenCL)

