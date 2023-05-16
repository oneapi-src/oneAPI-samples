#include<iostream>
#include<math.h> 
#include<cstdlib>

/*

    this implements the QR decmposition using Gram-Schmidt process

*/
#ifndef __QR_MGS_HPP__
#define __QR_MGS_HPP__

typedef union {
  float f;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

template<typename T> 
class QR_Decmp{
    
    private:
        // this is for square matrix
        int n;
        int matrix_id;
        T  *matA;
        T  *matR, *matQ, *matIR, *matS,*matP;

    public: 
        QR_Decmp(T *InA, int n, int id);
        ~QR_Decmp();

        T dotA(int a, int b, int p);
        void initializeI0(int p);
        void do_mainloop(int p);
        void QR_decompose(int p);

        T* get_Q();
        T* get_R();

};



template<typename T>  QR_Decmp<T>::QR_Decmp(T *InA, int n, int id){
    this->n = n;
    this->matrix_id = id;
    this->matA = InA;

    this->matR = new T[n*n];
    this->matQ = new T[n*n];
    this->matIR = new T[n*n];
    this->matS = new T[n*n];
    this->matP = new T[n*n];

    // Initialising the R matrix to zero
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matR[i*n+j] = 0;
        }
    }
}

template <typename T> QR_Decmp<T>::~QR_Decmp(){

    delete[] this->matR;
    delete[] this->matQ;
    delete[] this->matIR;
    delete[] this->matS;
    delete[] this->matP;
}

template<typename T> T  QR_Decmp<T>::dotA(int a, int b, int p){
    T sum = 0;
    // bool C_flag = false;
    for(int i = 0; i < p; i++){
        T mul = this->matA[i*n+a] * this->matA[i*n+b];
        // float_cast d1 = {.f = sum};
        // float_cast d2 = {.f = mul};
        sum += mul;
        // float_cast res = {.f = sum};
        // int shift = std::max(d1.parts.exponent, d2.parts.exponent) - res.parts.exponent;
        // if(shift > 10){
        //     C_flag = true;
        //     // std::cout << "floating point cancellatin matrix id: " << this->matrix_id <<"\n";
        //     // std::cout << "inputs: " << d1.f << "," << d2.f << "\n";
        // }

        
    }
    // if(C_flag && a == b){
    //     std::cout << "floating point cancellatin matrix id: " << this->matrix_id <<"\n";
    // }
    return sum;
}

template<typename T> void  QR_Decmp<T>::initializeI0(int p){
    this->matP[0] = this->dotA(0, 0, p);
    this->matIR[0] = 1.0/sqrt(this->matP[0]);
    this->matR[0] = sqrt(this->matP[0]);

    // std::cout << "this->matP[0]: " << this->matP[0] << "\n";

    for(int j = 1; j < p; j++){
        this->matP[j] = this->dotA(0,j, p);
        this->matS[j] = this->matP[j]/this->matP[0];
        this->matR[j] = this->matP[j] * this->matIR[0];
    }
}

template<typename T> void  QR_Decmp<T>::do_mainloop(int p){

    for(int i = 0; i < p-1; i++){
        // q_i = a_i * ir_{i,i}
        for(int k = 0; k < p; k++){
            this->matQ[k*n+i] = this->matA[k*n+i] * this->matIR[i*n+i];
        }
        // inner loop iteration 
        for(int j = i+1; j < p; j++){
            // a_{j} = a_{j} - s_{i,j}*a_{i};
            for(int k = 0; k < p; k++){
                this->matA[k*n+j] -= this->matS[i*n+j]*this->matA[k*n+i];
            }
            if( j == i+1){
                this->matP[(i+1)*n+(i+1)] = this->dotA(i+1, i+1, p);
                this->matIR[(i+1)*n+(i+1)] = 1.0/sqrt(this->matP[(i+1)*n+(i+1)]);
                this->matR[(i+1)*n+(i+1)] = sqrt(this->matP[(i+1)*n+(i+1)]);
            } else {
                this->matP[(i+1)*n+j] = this->dotA(i+1,j, p);
                this->matS[(i+1)*n+j] = this->matP[(i+1)*n+j]/this->matP[(i+1)*n+(i+1)];
                this->matR[(i+1)*n+j] = this->matP[(i+1)*n+j] * this->matIR[(i+1)*n+(i+1)];
            }
        }
    }

    for(int k = 0; k < p; k++){
        this->matQ[k*n+p-1] = this->matA[k*n+p-1] * this->matIR[(p-1)*n+(p-1)];
    }
}


template<typename T> void QR_Decmp<T>::QR_decompose(int p){

    this->initializeI0(p);
    this->do_mainloop(p);
}

template<typename T> T* QR_Decmp<T>::get_Q(){
    return this->matQ;
}

template<typename T> T* QR_Decmp<T>::get_R(){
    return this->matR;
}

#endif 