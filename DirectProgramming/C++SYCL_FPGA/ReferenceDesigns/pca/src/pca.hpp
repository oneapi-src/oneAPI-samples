#include<iostream>
#include<math.h> 
#include<cstdlib>
#include<algorithm>


#include "qr_MGS.hpp"
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
    constexpr size_t kRandomMin = 0;
    constexpr size_t kRandomMax = 1000;

    size_t kEigenMin = 2*this->p;
    size_t kEigenMax = 6*this->p;

    constexpr size_t kNoiseMin = 0;
    constexpr size_t kNoiseMax = 5000;


    T* TeigVec = new T[this->p * this->p];
    T* Teigval = new T[this->p];
    T* noise = new T[this->p];

    // int Teigval[5] = {100, 50, 25, 15, 2};
    for(int m_id  =0; m_id < this->matrixCount; m_id++){
    // initialising TeigVec with random numbers

        for(int i = 0; i < this->p; i++){
            // making sure two eigen values are unlikely same
            Teigval[i] = (rand() % (kEigenMax - kEigenMin) + kEigenMin) + (((double)rand()-RAND_MAX/2)/(double)RAND_MAX);
        }

        for(int i =0; i < this->p; i++){
            for(int j = 0; j < this->p; j++){
                TeigVec[i*this->p+j] = (((double)rand()-RAND_MAX/2)/(double)RAND_MAX);//(rand() % (kRandomMax - kRandomMin) + kRandomMin);
            }
        }

        // setting eigen vectors
        QR_Decmp<T> qr_decom(TeigVec, this->p, m_id);
        qr_decom.QR_decompose(this->p);
        T* Q = qr_decom.get_Q();

        // T Q[5*5] = {-0.6267675666742987,0.22499524355785394,0.6380304839079842,0.23211364638804452,-0.3091600144037226,-0.01129237663111344,-0.654231795494806,-0.06151038599510441,0.7535663032385346,-0.014406215040944245,-0.4777076293368368,-0.0020245259873950677,-0.6972641085018443,-0.0759440963299726,-0.5290052735045824,-0.0332016368092065,-0.6997500858368986,0.29611963370306005,-0.5890567264953692,-0.27308021567035545,-0.6145957791766623,-0.17805509486045076,-0.12356981649889559,-0.15970519398891908,0.7414803860358399};

        // std::cout << "Matrix Q is: \n";
        // for(int i = 0; i < this->p; i++){
        //     for(int j = 0; j < this->p; j++){
        //         std::cout << Q[i*this->p+j] << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "\n";


        // std::cout << "Sum is: \n";
        // for(int i = 0; i < this->p; i++){
        //     double sum = 0;
        //     for(int j = 0; j < this->p; j++){
        //         sum += Q[j*this->p+0] * Q[j*this->p+i];
        //     }
        //     std::cout << sum << " ";
        // }
        // std::cout << "\n";

        //setting eigen values 
        // for(int i = 0; i < this->p; i++){
        //     Teigval[i] = (rand() % (kEigenMax - kEigenMin) + kEigenMin);
        // }



        if(this->debug) std::cout << "Initial input Matrix A for PCA :"  << this->matrixCount << " \n";
        int offset = m_id * this->n * this->p;
        for(int i = 0; i < this->n; i++){ // samples 

            // std::default_random_engine generator;
            // std::normal_distribution<double> distribution(0,Teigval[k]);

            for(int k = 0; k < this->p; k++){
                noise[k] = (((double)rand()-RAND_MAX/2)/(double)RAND_MAX) * Teigval[k];
            }

            for(int j = 0; j < this->p; j++){ // features 
                this->matA[offset+ i*p+j] = 0;
                for(int k = 0; k < this->p; k++){ // vectors
                    // int noise =  (rand() % (kNoiseMax - kNoiseMin) + kNoiseMin);
                    this->matA[offset+ i*p+j] +=  noise[k] * Q[j*this->p+k];
                }
                if(this->debug) std::cout << this->matA[i*p+j] << " ";
            }
            if(this->debug) std::cout << "\n";
        }
    }

    delete[] TeigVec;
    delete[] Teigval;
    delete[] noise;


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
                this->matUA[offset + i*p+j] = (this->matA[offset + i*p+j]-meanVec[j]); // /stDev[j];
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
                for(int k = 0; k < this->n; k++){
                    this->matC[offset + i*p+j] += this->matUA[offset + k*p+i]*this->matUA[offset + k*p+j];
                }
                this->matC[offset + i*p+j] = (1.0/(this->n-1))*this->matC[offset + i*p+j];
                if(this->debug) std::cout << this->matC[offset + i*p+j] << " ";
            }
            if(this->debug) std::cout << "\n";
        }
    }
}


