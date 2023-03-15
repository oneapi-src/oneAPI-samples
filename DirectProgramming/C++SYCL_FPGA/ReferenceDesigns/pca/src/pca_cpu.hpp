#include<iostream>
#include<math.h> 
#include<cstdlib>
#include<algorithm>
#include <random>


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
    std::vector<T> eigVal, eigVec;
    T *Q, *R;




 public: 
    PCA(int n, int p, int count, int debug = 0);
    ~PCA();
    void populate_A();
    void normalizeSamples();
    void calculate_covariance();
    void QR_iteration();

};


template<typename T> PCA<T>::PCA(int n,int p, int count,  int debug){
    this->n = n;
    this->p = p;
    this->matrixCount = count;
    this->debug = debug;

    this->matA.resize(n*p*this->matrixCount);
    this->matUA.resize(n*p*this->matrixCount);
    this->matC.resize(p*p*this->matrixCount);

    this->eigVal.resize(p*p*this->matrixCount);
    this->eigVec.resize(p*p*this->matrixCount);

    // initialising the eigen vector to identical matrix 
    // for()


}


template<typename T> PCA<T>::~PCA(){
}



// populating matrix a with random numbers
template<typename T> void PCA<T>::populate_A(){
    // constexpr size_t kRandomMin = 0;
    // constexpr size_t kRandomMax = 1000;

    size_t kEigenMin = 50*this->p;
    size_t kEigenMax = 60*this->p;

    // constexpr size_t kNoiseMin = 0;
    // constexpr size_t kNoiseMax = 5000;


    T* TeigVec = new T[this->p * this->p];
    T* Teigval = new T[this->p];
    T* noise = new T[this->p];

    // int Teigval[5] = {100, 50, 25, 15, 2};
    for(int m_id  =0; m_id < this->matrixCount; m_id++){
    // initialising TeigVec with random numbers

        for(int i = 0; i < this->p; i++){
            // making sure two eigen values are unlikely same
            Teigval[i] =  (rand() % (kEigenMax - kEigenMin) + kEigenMin) + (((double)rand()-RAND_MAX/2)/(double)RAND_MAX);
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

    
        std::random_device rd{};
        std::mt19937 gen{rd()};
 
        // values near the mean are the most likely
        // standard deviation affects the dispersion of generated values from the mean
        std::normal_distribution<> d{0, 1};

        if(this->debug) std::cout << "Initial input Matrix A for PCA :"  << this->matrixCount << " \n";
        int offset = m_id * this->n * this->p;
        for(int i = 0; i < this->n; i++){ // samples 

            // std::default_random_engine generator;
            // std::normal_distribution<double> distribution(0,Teigval[k]);



            for(int k = 0; k < this->p; k++){
                noise[k] = rand() % ((int)(Teigval[k])); //(((double)rand()-RAND_MAX/2)/(double)RAND_MAX) * Teigval[k];
                // noise[k] = (((double)rand()-RAND_MAX/2)/(double)RAND_MAX) * Teigval[k];
            }

            for(int j = 0; j < this->p; j++){ // features 
                this->matA[offset+ i*p+j] = 0;
                for(int k = 0; k < this->p; k++){ // vectors
                    // int noise =  (rand() % (kNoiseMax - kNoiseMin) + kNoiseMin);
                   this->matA[offset+ i*p+j] +=  noise[k] * Q[j*this->p+k];
                }
                if(this->debug) std::cout << this->matA[offset+i*p+j] << " ";
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
            stDev[i] = sqrt(Var[i]/(this->n));
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
        int offsetUA = m_id * this->n * this->p;
        int offsetC = m_id * this->p * this->p;
        for(int i = 0; i < p; i++){
            for(int j = 0; j < p; j++ ){
                this->matC[offsetC + i*p+j] = 0;
                for(int k = 0; k < this->n; k++){
                    this->matC[offsetC + i*p+j] += this->matUA[offsetUA + k*p+i]*this->matUA[offsetUA + k*p+j];
                }
                this->matC[offsetC + i*p+j] = (1.0/(this->n))*this->matC[offsetC + i*p+j];
                if(this->debug) std::cout << this->matC[offsetC + i*p+j] << " ";
            }
            if(this->debug) std::cout << "\n";
        }
    }
}


// template<typename T> void PCA<T>::QR_iteration(){
//   // QR decomposition on CPU 

//     for(int m_id  =0; m_id < this->matrixCount; m_id++){
//         int offsetC = m_id * this->p * this->p;

//         QR_Decmp<DTypeCPU> qrd_cpu(this->matC, this->p, 0);
//         int kP = kRows;
//         int iter = kRows*ITER_PER_EIGEN;

//         for(int li = 0; li < iter; li++){
//             // convergence test 
//             bool close2zero = 1;

//             // Wilkinson shift computation 
//             T a_wilk = this->matC[(kP-2)*kRows+kP-2];
//             T b_wilk = this->matC[(kP-1)*kRows+kP-2];
//             T c_wilk = this->matC[(kP-1)*kRows+kP-1];

//             T lamda = (a_wilk - c_wilk)/2.0;
//             T sign_lamda = (lamda > 0) ? 1.0 : -1.0;
//             T shift = RELSHIFT ? c_wilk : c_wilk - (sign_lamda*b_wilk*b_wilk)/(fabs(lamda) + sqrt(lamda * lamda + b_wilk*b_wilk));

//             // adjustment for better accuracy when using floating point operations 
//             shift -= shift*SHIFT_NOISE;
//             shift = (li < NO_SHIFT_ITER) ? 0 : shift;

//             // if(DEBUGEN && matrix_index == DEBUGMINDEX) {dAMat << "\n\nA Matrix before shift at iteration: " << li << "\n";}
//             // for(int i = 0; i < kP; i++){
//             //   for(int j = 0; j < kP; j++){
//             //       if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << a_matrix_cpu[matrix_offset+i*kRows+j]  << " ";
//             //   }
//             //   if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << "\n";
//             // }

//             for(int i = 0; i < kP; i++){
//               this->matC[i*kRows+i] -= shift;
//             }

//             // if(DEBUGEN && matrix_index == DEBUGMINDEX) {dAMat << "\n\nA Matrix after shift at iteration: " << li << "\n";}
//             // for(int i = 0; i < kP; i++){
//             //   for(int j = 0; j < kP; j++){
//             //       if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << a_matrix_cpu[matrix_offset+i*kRows+j]  << " ";
//             //   }
//             //   if(DEBUGEN && matrix_index == DEBUGMINDEX) dAMat << "\n";
//             // }

//             qrd_cpu.QR_decompose(kP);
//             R = qrd_cpu.get_R();
//             Q = qrd_cpu.get_Q();
//             // RQ computation and updating A 
            
//             for(int i = 0; i < kP; i++){
//               for(int j = 0; j < kP; j++){
//                 this->matC[i*kRows+j] = 0;
//                 for(int k = 0; k < kP; k++){
//                   this->matC[i*kRows+j] += R[i*kRows+k]*Q[k*kRows+j];
//                 } 
//               }
//             }

//             // if(DEBUGEN && matrix_index == DEBUGMINDEX) {dQMat << "\n\nQ Matrix at iteration: " << li << "\n";}
//             // if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRMat << "\n\nR Matrix at iteration: " << li << "\n";}
//             // for(int i = 0; i < kP; i++){
//             //   for(int j = 0; j < kP; j++){
//             //       if(DEBUGEN && matrix_index == DEBUGMINDEX) dQMat << Q[i*kRows+j] << " ";
//             //       if(DEBUGEN && matrix_index == DEBUGMINDEX) dRMat << R[i*kRows+j] << " ";
//             //   }
//             //   if(DEBUGEN && matrix_index == DEBUGMINDEX) dQMat << "\n";
//             //   if(DEBUGEN && matrix_index == DEBUGMINDEX) dRMat << "\n";
//             // }

//             // adding back the shift from the matrix
//             for(int i = 0; i < kP; i++){
//               this->matC[i*kRows+i] += shift;
//             }

//             // if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRQ << "\n\nRQ Matrix at iteration: " << li << "\n";}
//             // for(int i = 0; i < kRows; i++){
//             //   for(int j = 0; j < kRows; j++){
//             //     if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRQ << a_matrix_cpu[matrix_offset+i*kRows+j] << " ";}
//             //   }
//             //   if(DEBUGEN && matrix_index == DEBUGMINDEX) {dRQ << "\n";}
//             // }

//             // Eigen vector accumulation 
//             // if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << "QQ Matrix at iteration: " << li << "\n";
//             for(int i = 0; i < kRows; i++){
//               std::fill(TmpRow.begin(), TmpRow.end(), 0);
//               for(int j = 0; j < kRows; j++){
//                 for(int k = 0; k < kRows; k++){
//                   T I_val = (k==j) ? 1 : 0;
//                   T q_val = (j >= kP || k >= kP) ? I_val : Q[k*kRows+j];
//                   TmpRow[j] += eigen_vectors_cpu[matrix_offset+i*kRows+k]*q_val;
//                 }
//               }
//               for(int k = 0; k < kRows; k++) {
//                 eigen_vectors_cpu[matrix_offset+i*kRows+k] = TmpRow[k];
//                 if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << eigen_vectors_cpu[matrix_offset+i*kRows+k] << " ";
//               }
//               if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << "\n";
//             }
//             if(DEBUGEN && matrix_index == DEBUGMINDEX) dQQ << "\n";

//             for(int j = 0; j < kP-1; j++){
//               if(std::fabs(a_matrix_cpu[matrix_offset + (kP-1)*kRows+j]) > KTHRESHOLD){
//                 close2zero = 0;
//                 break;
//               }
//             }

//             if(close2zero && kP == KDEFLIM){
//               // total_iteration += li+1;
//               break;
//             } else if(close2zero){
//               kP -= 1;
//             }
          
//           }
//     }

// }

