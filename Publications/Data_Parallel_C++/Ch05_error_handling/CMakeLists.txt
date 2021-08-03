# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.4 )
project(Chap5)
add_book_sample(
    TEST
    TARGET fig_5_1_async_task_graph
    SOURCES fig_5_1_async_task_graph.cpp)

add_book_sample(
    TARGET fig_5_2_sync_error
    SOURCES fig_5_2_sync_error.cpp)

add_book_sample(
    TARGET fig_5_3_async_error
    SOURCES fig_5_3_async_error.cpp)

add_book_sample(
    TARGET fig_5_4_unhandled_exception
    SOURCES fig_5_4_unhandled_exception.cpp)

add_book_sample(
    TARGET fig_5_5_terminate
    SOURCES fig_5_5_terminate.cpp)

add_book_sample(
    TEST
    TARGET fig_5_6_catch_snip
    SOURCES fig_5_6_catch_snip.cpp)

add_book_sample(
    TARGET fig_5_7_catch
    SOURCES fig_5_7_catch.cpp)

add_book_sample(
    TARGET fig_5_8_lambda_handler
    SOURCES fig_5_8_lambda_handler.cpp)

add_book_sample(
    TARGET fig_5_9_default_handler_proxy
    SOURCES fig_5_9_default_handler_proxy.cpp)
