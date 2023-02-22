#include<iostream>
#include<math.h> 
#include<cstdlib>

/*

    this implements the QR decmposition using Gram-Schmidt process

*/

template<typename T> 
class QR_Decmp{
    
    private:
        // this is for square matrix
        int n;
        T  *matA_ptr;
        T  *vecPrj;
        T  *matU, *matR, *matQ;

    public: 
        QR_Decmp(T *matA_ptr, int n);
        ~QR_Decmp();
        void calculate_projection(int a, int b, int p);
        void calculate_U(int p);
        void calculate_Q(int p);
        void calculate_R(int p);
        void QR_decompose(int p);
        T* get_Q();
        T* get_R();

};



template<typename T>  QR_Decmp<T>::QR_Decmp(T *matA, int n){
    this->n = n;
    this->matA_ptr = matA;

    this->matU = new T[n*n];
    this->vecPrj = new T[n];
    this->matR = new T[n*n];
    this->matQ = new T[n*n];

    // Initialising the R matrix to zero
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matR[i*n+j] = 0;
        }
    }
}

template <typename T> QR_Decmp<T>::~QR_Decmp(){

    delete this->matU;
    delete this->vecPrj;
    delete this->matR;
    delete this->matQ;
}

template<typename T> void QR_Decmp<T>::calculate_projection(int a , int b, int p){
    T inner_ua = 0;
    T inner_uu = 0;

    // inner product <u,a>
    for(int i = 0; i < p; i++){
        inner_ua += this->matU[i*n+a] * this->matA_ptr[i*n+b];
    }

    // inner product <u,u>
    for(int i = 0; i < p; i++){
        inner_uu += this->matU[i*n+a] * this->matU[i*n+a];
    }

    // projection vector 
    for(int i = 0; i < p; i++){
        this->vecPrj[i] = this->matU[i*n+a] * inner_ua/inner_uu;
    }
}

template<typename T> void QR_Decmp<T>::calculate_U(int p){

    // U_{k} = a_{k} - sigma_{j=1}^{k-1}proj_{uj}ak
    // std::cout << "\n last element in this->matU is: " << this->matA_ptr[(p-1)*p + p-1] << "\n";
    for(int i = 0; i < p; i++){

        //initially assigning U_{k} to a_{k}
        for(int k = 0; k < p; k++){
            this->matU[k*n+i] = this->matA_ptr[k*n+i];
        }

        for(int j = 0; j < i; j++){
            this->calculate_projection(j,i,p);
            // subtracting the projections
            for(int k  = 0; k < p; k++){
                this->matU[k*n+i] -= this->vecPrj[k];
            }
        }
    }

}

template<typename T> void QR_Decmp<T>::calculate_Q(int p){
    // Q = [e_{0}, e_{1} .. e_{n-1}]
    // e_{i} = u_{i}/||u_{i}||
    
    for(int i = 0; i < p; i++){
        // calculating the modulus 
        T mag = 0;
        for(int k = 0; k < p; k++){
            mag += this->matU[k*n+i] * this->matU[k*n+i];
        }
        mag = sqrt(mag);
        // std::cout << "mag is " << mag << " i is: " << i <<"\n";
        mag = 1.0/mag;

        for(int k = 0; k < p; k++){
            this->matQ[k*n+i] = this->matU[k*n+i]*mag; 
            if(isnan(this->matQ[k*n+i])){
                std::cout << "modulus is: " << mag <<  " i: is: " << i <<  " p is:" << p << "\n";
                std::cout << "something went wrong\n";
                exit(0);
            }
        }

    }

}

template<typename T> void QR_Decmp<T>::calculate_R(int p){
    // R matrix is an upper trangular matrix with element (i,j)
    // corrsponds to <e_{i}, a_{j}>
    for(int i = 0; i < p; i++){
        for (int j = i; j < p; j++){
            this->matR[i*n+j] = 0;
            for(int k = 0; k < p; k++){
                this->matR[i*n+j] += this->matQ[k*n+i] * this->matA_ptr[k*n+j];
            }
        }
    }

}

template<typename T> void QR_Decmp<T>::QR_decompose(int p){

    calculate_U(p);
    calculate_Q(p);
    calculate_R(p);
}

template<typename T> T* QR_Decmp<T>::get_Q(){
    return this->matQ;
}

template<typename T> T* QR_Decmp<T>::get_R(){
    return this->matR;
}