//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>

int main(){

    //# initialize some data array
    const int N = 16;
    float data[N];
    for(int i=0;i<N;i++) data[i] = i;

    //# computation on CPU
    for(int i=0;i<N;i++) data[i] = data[i] * 5;

    //# print output
    for(int i=0;i<N;i++) std::cout << data[i] << "\n"; 
}
