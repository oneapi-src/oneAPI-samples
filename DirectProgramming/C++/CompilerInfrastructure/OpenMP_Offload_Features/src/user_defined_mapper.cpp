//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

extern "C" int printf(const char *, ...);
struct MyArr {
  int num;
  int *arr;
};
#pragma omp declare mapper(id : MyArr c) map(c.num, c.arr [0:c.num])

void foo(int num, int *arr, int *arr_one) {
  int i;
  MyArr c;
  c.num = num;
  c.arr = arr;
  for (i = 0; i < num; ++i)
    printf("%s%3d %s", (i == 0 ? "In : " : ""), c.arr[i],
           (i == num - 1 ? "\n" : ""));

#pragma omp target map(mapper(id), tofrom : c)
  {
    int j;
    for (j = 0; j < c.num; ++j) c.arr[j] *= 2;
  }
  for (i = 0; i < num; ++i)
    printf("%s%3d %s", (i == 0 ? "Out: " : ""), c.arr[i],
           (i == num - 1 ? "\n" : ""));

  MyArr c_one;
  c_one.num = num;
  c_one.arr = arr_one;
  for (i = 0; i < num; ++i)
    printf("%s%3d %s", (i == 0 ? "In : " : ""), c_one.arr[i],
           (i == num - 1 ? "\n" : ""));

#pragma omp target map(mapper(id), tofrom : c_one)
  {
    int j;
    for (j = 0; j < c_one.num; ++j) c_one.arr[j] *= 2;
  }
  for (i = 0; i < num; ++i)
    printf("%s%3d %s", (i == 0 ? "Out: " : ""), c_one.arr[i],
           (i == num - 1 ? "\n" : ""));
}

int main() {
  int arr4[] = {1, 2, 4, 8};
  int arr8[] = {1, 2, 4, 8, 16, 32, 64, 128};
  int arr4_one[] = {1, 2, 4, 8};
  int arr8_one[] = {1, 2, 4, 8, 16, 32, 64, 128};
  foo(sizeof(arr4) / sizeof(arr4[0]), arr4, arr4_one);
  foo(sizeof(arr8) / sizeof(arr8[0]), arr8, arr8_one);
  return 0;
}
