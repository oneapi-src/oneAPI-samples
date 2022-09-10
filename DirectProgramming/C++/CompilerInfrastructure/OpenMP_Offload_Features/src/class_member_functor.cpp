//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <bits/stdc++.h>
using namespace std;

// A Functor
class Inc {
 private:
  int num;

 public:
  Inc(int n) : num(n) {}
  int operator()(int arr_num) const { return num + arr_num; }
};

int main() {
  int arr[] = {1, 2, 3, 4, 5};
  int n = sizeof(arr) / sizeof(arr[0]);
  int add5 = 5;
  Inc a_inc(add5);

#pragma omp target teams distribute parallel for map(arr [0:n]) map(to : a_inc)
  for (int k = 0; k < n; k++) {
    arr[k] = arr[k] + a_inc(k);
  }
  for (int i = 0; i < n; i++) cout << arr[i] << " ";
  cout << "\n"
       << "Done ......\n";
}
