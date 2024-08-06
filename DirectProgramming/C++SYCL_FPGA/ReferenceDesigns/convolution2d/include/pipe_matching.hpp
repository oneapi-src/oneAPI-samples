#pragma once

#include <sycl/ext/intel/experimental/pipes.hpp>

template <typename PipeT, typename MatchT>
struct is_pipe_of_type : std::false_type {};

template <typename MatchT, typename ID, int Capacity, typename Properties>
struct is_pipe_of_type<
    sycl::ext::intel::experimental::pipe<ID, MatchT, Capacity, Properties>, MatchT>
    : std::true_type {};