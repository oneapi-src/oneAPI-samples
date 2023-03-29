//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "multiply.hpp"
#include <iostream>

typedef unsigned long long UINT64;
#define xstr(s) x_str(s)
#define x_str(s) #s

using namespace std;

// routine to initialize an array with data
void InitArr(TYPE row, TYPE col, TYPE off, TYPE a[][NUM]) {
  int i, j;

  for (i = 0; i < NUM; i++) {
    for (j = 0; j < NUM; j++) {
      a[i][j] = row * i + col * j + off;
    }
  }
}

// routine to print out contents of small arrays
void PrintArr(char *name, TYPE Array[][NUM]) {
  int i, j;

  cout << "\n" << name << "\n";

  for (i = 0; i < NUM; i++) {
    for (j = 0; j < NUM; j++) {
      cout << Array[i][j] << "\t";
    }
    cout << endl;
  }
}

int main() {

  char *buf1, *buf2, *buf3, *buf4;
  char *addr1, *addr2, *addr3, *addr4;
  Array *a, *b, *c, *t;
  int Offset_Addr1 = 128, Offset_Addr2 = 192, Offset_Addr3 = 0,
      Offset_Addr4 = 64;

  // malloc arrays space

  buf1 = new char[NUM * NUM * (sizeof(double)) + 1024];
  cout << "Address of buf1 = " << buf1 << endl;
  addr1 = buf1 + 256 - ((UINT64)buf1 % 256) + (UINT64)Offset_Addr1;
  cout << "Offset of buf1 = " << addr1 << endl;

  buf2 = new char[NUM * NUM * (sizeof(double)) + 1024];
  cout << "Address of buf2 = " << buf2 << endl;
  addr2 = buf2 + 256 - ((UINT64)buf2 % 256) + (UINT64)Offset_Addr2;
  cout << "Offset of buf2 = " << addr2 << endl;

  buf3 = new char[NUM * NUM * (sizeof(double)) + 1024];
  cout << "Address of buf3 = " << buf3 << endl;
  addr3 = buf3 + 256 - ((UINT64)buf3 % 256) + (UINT64)Offset_Addr3;
  cout << "Offset of buf3 = " << addr3 << endl;

  buf4 = new char[NUM * NUM * (sizeof(double)) + 1024];
  cout << "Address of buf4 = " << buf4 << endl;
  addr4 = buf4 + 256 - ((UINT64)buf4 % 256) + (UINT64)Offset_Addr4;
  cout << "Offset of buf4 = " << addr4 << endl;

  a = (Array *)addr1;
  b = (Array *)addr2;
  c = (Array *)addr3;
  t = (Array *)addr4;

  // initialize the arrays with data
  InitArr(3, -2, 1, a);
  InitArr(-2, 1, 3, b);

  cout << "Using multiply kernel: " << xstr(MULTIPLY) << "\n";

  // start timing the matrix multiply code
  TimeInterval matrix_time;
  ParallelMultiply(NUM, a, b, c, t);
  double matrix_elapsed = matrix_time.Elapsed();
  cout << "Elapsed Time: " << matrix_elapsed << "s\n";

  // free memory
  delete[] buf1;
  delete[] buf2;
  delete[] buf3;
  delete[] buf4;
}
