#include "iostream"
#include "stdlib.h"

int main(){
    int n = 100;
    double * data = (double *)malloc( n * sizeof(double));
    for( int x = 0; x < n; ++x){
	    data[x] = x;	    
    }

    double readSum = 0.0;
    double writeSum = 0.0;
    double captureSum = 0.0;
    double updateSum = 0.0;

#pragma omp target teams loop map(tofrom:data[0:n]) map(from:readSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
#pragma omp atomic read
            readSum = x;
        }
    }

#pragma omp target teams loop map(tofrom:data[0:n]) map(from:writeSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
#pragma omp atomic write
            writeSum = x*2 + 1;
        }
    }

#pragma omp target teams loop map(tofrom:data[0:n]) map(from:captureSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
#pragma omp atomic capture
            captureSum = data[x]--;
            }
    }

    std::cout << captureSum << std::endl;

#pragma omp target teams loop map(tofrom:data[0:n]) map(from:updateSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
#pragma omp atomic update
    	    updateSum++;
        }
    }
    return 0;
}

