//=========================================================
// Modifications Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
//=========================================================

// based on samples from "C++17 Parallel Algorithms" at CppCon 2016
// a talk by Bryce Adelstein Lelbach

#include <cctype>
#include <chrono>
#include <algorithm>
#include <execution>  // for the execution policy
#include <iostream>
#include <numeric>  // for transform_reduce()
#include <string>
#include <string_view>
#include <vector>

bool is_word_beginning(char left, char right) {
  return std::isspace(left) && !std::isspace(right);
}

template <typename Policy>
std::size_t word_count(std::string_view s, Policy policy) {
  if (s.empty()) return 0;

  std::size_t wc = (!std::isspace(s.front()) ? 1 : 0);
  wc += std::transform_reduce(policy, s.begin(), s.end() - 1, s.begin() + 1,
                              std::size_t(0), std::plus<std::size_t>(),
                              is_word_beginning);

  return wc;
}

template <typename TFunc>
void RunAndMeasure(const char* title, TFunc func) {
  const auto start = std::chrono::steady_clock::now();
  auto ret = func();
  const auto end = std::chrono::steady_clock::now();
  std::cout << title << ": "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << " ms, res " << ret << "\n";
}

int main(int argc, char* argv[]) {
  const int COUNT = argc > 1 ? atoi(argv[1]) : 1'000'000;
  std::string str(COUNT, 'a');

  for (int i = 0; i < COUNT; ++i) {
    if (i % 5 == 0 || i % 17 == 0) str[i] = ' ';  // add a space
  }

  std::cout << "string length: " << COUNT << ", first 60 letters: \n";
  std::cout << str.substr(0, 60) << std::endl;

  RunAndMeasure("word_count seq",
                [&str] { return word_count(str, std::execution::seq); });

  RunAndMeasure("word_count par",
                [&str] { return word_count(str, std::execution::par); });

  RunAndMeasure("word_count par_unseq",
                [&str] { return word_count(str, std::execution::par_unseq); });
}
