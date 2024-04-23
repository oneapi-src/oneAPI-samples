//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>

//# STEP 1 : Include header for SYCL
//# YOUR CODE GOES HERE





int main(){
    
    //# STEP 2: Create a SYCL queue and device selection for offload
    //# YOUR CODE GOES HERE
    
    
    
    

    //# initialize some data array
    const int N = 16;
    
    //# STEP 3: Allocate memory so that both host and device can access
    //# MODIFY THE CODE BELOW    
    float a[N], b[N], c[N];
    
    
    
    
    for(int i=0;i<N;i++) {
        a[i] = 1;
        b[i] = 2;
        c[i] = 0;
    }

    
    //# STEP 4: Submit computation to Offload device
    //# MODIFY THE CODE BELOW      
    
    //# computation
    for(int i=0;i<N;i++) c[i] = a[i] + b[i];

    
    
    
    //# print output
    for(int i=0;i<N;i++) std::cout << c[i] << "\n"; 
}
