// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>

// START BOOK SNIP
class Functor {
  public:
    Functor(int i, int &j) : my_i{i}, my_jRef{j} { }

    int operator()(int k0, int &l0) {
      my_jRef = 2 * my_jRef;
      k0 = 2 * k0;
      l0 = 2 * l0;
      return my_i + my_jRef + k0 + l0;
    }

  private:
    int my_i;
    int &my_jRef;
};
// END BOOK SNIP

int main() {
  int i = 1, j = 10, k = 100, l = 1000;

  Functor F{i, j};

  std::cout << "First call returned " << F( k, l ) << "\n";
  std::cout << "Second call returned " << F( k, l ) << "\n";

  return 0;
}


