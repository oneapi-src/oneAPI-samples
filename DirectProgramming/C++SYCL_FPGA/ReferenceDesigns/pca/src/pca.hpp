#include<iostream>
#include<math.h> 
#include<cstdlib>
#include<algorithm>

// #include "qr_decom.hpp"
// #include <sycl/sycl.hpp>
// #include <sycl/ext/intel/fpga_extensions.hpp>
// #include <sycl/ext/intel/ac_types/ac_complex.hpp>

/*
this source implements the steps to 
identify principal compoents (eigen vectors) 
of a matrix and finally transform input matrix
along the directions of the principal components

Following are the main steps in order to transform a 
matrix A. Matrix A will contain n samples with p features
making it nxp order matrix

1. Calculating the mean vector u
   (F_0, F_1, ..., F_(p-1))

2. Calculating zero mean matrix 
   B = A - h^{T}*u                 
   here h is a vector with ones of size n

3. Calculate covariance matrix of size pxp
   C = (1.0/(n-1)) * B*B^{T}

4. Calculating eigen vectors and eigen values QR decomposition
   in an iterative loop

5. sort eigen vectors using eigen values in a decending order

6. form the tranformation matrix using eigen vectors
*/
typedef double F_type;

template <typename T> class PCA {
 public: 
    // n - number of samples, p - number of features 
    int n, p, matrixCount, debug;
    std::vector<T> matA, matUA, matC;



 public: 
    PCA(int n, int p, int count, int debug = 0);
    ~PCA();
    void populate_A();
    void normalizeSamples();
    void calculate_covariance();

};


template<typename T> PCA<T>::PCA(int n,int p, int count,  int debug){
    this->n = n;
    this->p = p;
    this->matrixCount = count;
    this->debug = debug;
    this->matA.resize(n*p*this->matrixCount);
    this->matUA.resize(n*p*this->matrixCount);
    this->matC.resize(p*p*this->matrixCount);

}

template<typename T> PCA<T>::~PCA(){
}



// populating matrix a with random numbers
template<typename T> void PCA<T>::populate_A(){
    constexpr size_t kRandomMin = 1;
    constexpr size_t kRandomMax = 1000;
    for(int m_id  =0; m_id < this->matrixCount; m_id++){
        if(this->debug) std::cout << "Initial input Matrix A for PCA :"  << this->matrixCount << " \n";
        int offset = m_id * this->n * this->p;
        for(int i = 0; i < this->n; i++){
            for(int j = 0; j < this->p; j++){
                this->matA[offset+ i*p+j] = (rand() % (kRandomMax - kRandomMin) + kRandomMin);
                if(this->debug) std::cout << this->matA[i*p+j] << " ";
            }
            if(this->debug) std::cout << "\n";
        }
    }
}


// Pre process steps for PCA
// Samples need to be normalised 
// First mean vector is computed
// standard devaition comutation 
// Normalized   
template<typename T> void PCA<T>::normalizeSamples(){
    // setting mean vector to zero
    T* meanVec = new T[this->p];
    T* Var = new T[this->p];
    T* stDev = new T[this->p];

    for(int m_id  =0; m_id < this->matrixCount; m_id++){
        int offset = m_id * this->n * this->p;
        for(int i = 0; i < p; i++){
            meanVec[i] = 0;
            Var[i] = 0;
        }

        // getting vector sum of the samples
        for(int i = 0; i < n; i++){
            for(int j = 0; j < p; j++){
                meanVec[j] += this->matA[offset+ i*p+j];
            }
        }

        // Calculating the mean vector
        if(this->debug) std::cout << "\nMean Vec is: \n";
        for(int i = 0; i < p; i++){
            meanVec[i] /= this->n;
            if(this->debug) std::cout << meanVec[i] << " ";
        }
        if(this->debug) std::cout << "\n";

        // calculating the variance vector 
        for(int i = 0; i < n; i++){
            for(int j = 0; j < p; j++){
                T val = this->matA[offset+ i*p+j] - meanVec[j];
                Var[j] += val*val;

            }
        }

        // // calculating Standard Deviation 
        if(this->debug) std::cout << "\nStandard deviation is: \n";
        for(int i = 0; i < p; i++){
            stDev[i] = sqrt(Var[i]/this->n);
            if(this->debug) std::cout << stDev[i] << " ";
        }
        if(this->debug) std::cout << "\n";

        // normalising the input matrix 
        if(this->debug) std::cout << "\nNormalized matrix is: \n";
        for(int i = 0; i < n; i++){
            for(int j = 0; j < p; j++){
                this->matUA[offset + i*p+j] = (this->matA[offset + i*p+j]-meanVec[j])/stDev[j];
                if(this->debug) std::cout << this->matUA[offset + i*p+j] << " ";
            }
            if(this->debug) std::cout << "\n";
        }
    }

    // Standard deviation 
    delete[]  meanVec;
    delete[]  Var;
    delete[]  stDev;
}




template<typename T> void PCA<T>::calculate_covariance(){
    // covariance matrix matdA^{T} * matdA
    // this corresponds to matrix order pxp
    if(this->debug) std::cout << "\nCovariance matrix is: \n";

    for(int m_id  =0; m_id < this->matrixCount; m_id++){
        int offset = m_id * this->p * this->p;
        for(int i = 0; i < p; i++){
            for(int j = 0; j < p; j++ ){
                this->matC[offset + i*p+j] = 0;
                for(int k = 0; k < n; k++){
                    this->matC[offset + i*p+j] += this->matUA[offset + k*p+i]*this->matUA[offset + k*p+j];
                }
                this->matC[offset + i*p+j] = (1.0/(n-1))*this->matC[offset + i*p+j];
                if(this->debug) std::cout << this->matC[offset + i*p+j] << " ";
            }
            if(this->debug) std::cout << "\n";
        }
    }
}


