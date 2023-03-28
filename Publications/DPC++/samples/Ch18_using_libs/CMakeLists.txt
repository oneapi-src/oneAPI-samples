# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TEST
    TARGET fig_18_1_builtin
    SOURCES fig_18_1_builtin.cpp)

add_book_sample(
    TEST
    TARGET fig_18_7_swap
    SOURCES fig_18_7_swap.cpp)

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_11_std_fill
    SOURCES fig_18_11_std_fill.cpp)
endif()

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_13_binary_search
    SOURCES fig_18_13_binary_search.cpp)
endif()

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_15_pstl_usm
    SOURCES fig_18_15_pstl_usm.cpp)
endif()
