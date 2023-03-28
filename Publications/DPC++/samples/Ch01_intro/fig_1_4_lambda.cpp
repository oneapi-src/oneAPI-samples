// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>

void print_values( const int& i, const int& j, const int& k, const int& l ) {
  std::cout << "i == " << i << "\n";
  std::cout << "j == " << j << "\n";
  std::cout << "k == " << k << "\n";
  std::cout << "l == " << l << "\n";
}

int main() {

// START BOOK SNIP
  int i = 1, j = 10, k = 100, l = 1000;

  auto lambda = [i, &j] (int k0, int &l0) -> int {
    j = 2 * j;
    k0 = 2 * k0;
    l0 = 2 * l0;
    return i + j + k0 + l0;
  };

  print_values( i, j, k, l );
  std::cout << "First call returned " << lambda( k, l ) << "\n";
  print_values( i, j, k, l );
  std::cout << "Second call returned " << lambda( k, l ) << "\n";
  print_values( i, j, k, l );
// END BOOK SNIP

  return 0;
}


