#include<iostream>
#include<math.h> 
#include<cstdlib>



/*

    this implements the Hessenberg QR decmposition 

*/

template<typename T> 
class Hess_QR_Decmp{
    
    public:
        // this is for square matrix
        int n;
        T *matA_ptr, *matH, *matQ;
        T *vecU, *vecV, *vecTmp; 
        T *vecC, *vecS;

        // shift value
        T mu;

    public: 
        Hess_QR_Decmp(T *matA_ptr, int n);
        ~Hess_QR_Decmp();
        void hessXform();
        void calculateShift(int p);
        void hess_qr_rq(int p);
        void do_hess_qr_iteration();
};


template<typename T> 
Hess_QR_Decmp<T>::Hess_QR_Decmp(T *matA_ptr, int n){
    this->matA_ptr = matA_ptr;
    this->n = n;

    this->matH = new T[n*n];
    this->matQ = new T[n*n];
    this->vecU = new T[n];
    this->vecV = new T[n];
    this->vecTmp = new T[n];

    this->vecC = new T[n];
    this->vecS = new T[n];

    // copying input matrix to H
    for(int i = 0; i < n*n; i++){
        this->matH[i] = this->matA_ptr[i];
    }

    // initialising Q matrix to identity matrix 
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matQ[i*n+j] = (i==j) ? 1.0 : 0;
        }
    }
}

template<typename T> 
Hess_QR_Decmp<T>::~Hess_QR_Decmp(){


    delete this->matH;
    delete this->matQ;
    delete this->vecU;
    delete this->vecV;
    delete this->vecTmp;

    delete this->vecC;
    delete this->vecS;
}


template<typename T> 
void Hess_QR_Decmp<T>::hessXform(){
    int n = this->n;
    for(int j = 0; j < n-2; j++){
        
        // copying j^th columns j+1:n entries 
        T sum = 0;
        for(int i = j+1; i < n; i++){
            this->vecU[i] = this->matH[i*n+j];
            sum += this->matH[i*n+j]*this->matH[i*n+j];
        }
        T norm_u = sqrt(sum);
        // update the first element and vecU
        this->vecU[j+1] +=  this->vecU[j+1]/fabs(this->vecU[j+1]) * norm_u;

        // calculation the new norm 
        sum = 0;
        for(int i = j+1; i < n; i++){
            sum += this->vecU[i]*this->vecU[i];
        }

        // compute vecV
        norm_u = sqrt(sum);
        for(int i =j+1; i < n; i++){
            this->vecV[i] = this->vecU[i]/norm_u;
        }

        // H[j+1:n,:] -=  2*v@(np.transpose(v)@H[j+1:n,:])    
        for(int k = 0; k < n; k++){
            this->vecTmp[k] = 0;
            for(int i = j+1; i < n; i++){
                this->vecTmp[k] += this->vecV[i] * this->matH[i*n+k];
            }
        }

        // updating H 
        for(int i = j+1; i < n; i++){
            for(int k = 0; k < n; k++){
                this->matH[i*n+k] -= 2 * this->vecV[i] * this->vecTmp[k];
            }
        }

        // H[:,j+1:n] -= (H[:,j+1:n] @ (2*v)) @ np.transpose(v)
        for(int k = 0; k < n; k++){
            this->vecTmp[k] = 0;
            for(int i = j+1; i < n; i++){
                this->vecTmp[k] += 2*this->matH[k*n+i]*this->vecV[i];
            }
        }

        // updating H
        for(int k = 0; k < n; k++){
            for(int i = j+1; i < n; i++){
                this->matH[k*n+i] -= this->vecV[i] * this->vecTmp[k]; 
            }
        }

        // Q computation 
        // Q[:,j+1:n] -= (Q[:,j+1:n] @ (2*v)) @ np.transpose(v)
        for(int k = 0; k < n; k++){
            this->vecTmp[k] = 0;
            for(int i = j+1; i < n; i++){
                this->vecTmp[k] += 2*this->matQ[k*n+i]*this->vecV[i];
            }
        }

        // updating H
        for(int k = 0; k < n; k++){
            for(int i = j+1; i < n; i++){
                this->matQ[k*n+i] -= this->vecV[i] * this->vecTmp[k]; 
            }
        }


    }

    // elements other than on tri-diagonal are set to zero
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matH[i*n+j] = (abs(i-j) > 1 ) ? 0 : this->matH[i*n+j];
        }
    }

    


}


template<typename T> 
void Hess_QR_Decmp<T>::hess_qr_rq( int p){

    // subtracting the shift from the main diagonal 
    for(int i = 0; i < p; i++){
        this->matH[i*n+i] -= this->mu; 
    }


    // b1   c1  *   *   *   *   
    // a2   b2  c2  *   *   *  
    // *    a3  b3  c3  *   *
    // *    *   a4  b4  c4  *
    // *    *   *   a5  b5  c5   
    // *    *   *   *   a6  b6  

    // Rotation matrix G
    //  c   s
    //  -s  c 

    // QR decomposition 
    for(int j = 0; j < p-1; j++){
        T u_0 = this->matH[j*n+j];
        T u_1 = this->matH[(j+1)*n+j];

        T norm = sqrt(u_0*u_0 + u_1*u_1);
        T c = u_0/norm;
        T s = u_1/norm;

        // b1, c1, d1 computations 
        T b1 = c*this->matH[j*n+j] + s*this->matH[(j+1)*n+j];
        T c1 = c*this->matH[j*n+j+1] + s*this->matH[(j+1)*n+j+1];
        T d1 = c*this->matH[j*n+j+2] + s*this->matH[(j+1)*n+j+2];

        T a2 = 0 -s*this->matH[j*n+j] + c*this->matH[(j+1)*n+j];
        T b2 = 0 -s*this->matH[j*n+j+1] + c*this->matH[(j+1)*n+j+1];
        T c2 = 0 -s*this->matH[j*n+j+2] + c*this->matH[(j+1)*n+j+2];

        // updating the H matrix 
        this->matH[j*n+j] = b1;     this->matH[j*n+j+1] = c1;   this->matH[j*n+j+2] = d1;
        this->matH[(j+1)*n+j] = 0;  this->matH[(j+1)*n+j+1] = b2; this->matH[(j+1)*n+j+2] = c2;

        // storing the
        this->vecC[j] = c;
        this->vecS[j] = s;

    }

    // RQ computation 
    for(int j = 0; j < p-1; j++){

        T c = this->vecC[j];
        T s = this->vecS[j];


        // b1  c1  d1  *   *   *   
        // *   b2  c2  d2  *   *  
        // *    *  b3  c3  d3  *
        // *    *   *  b4  c4  d4
        // *    *   *   *  b5  c5   
        // *    *   *   *   *  b6  

        T cl = j > 0 ? this->matH[(i-1)*n+j]*c + this->matH[(i-1)*n+j+1]*s : 0;
        T bl = this->matH[i*n+j]*c + this->matH[i*n+j+1]*s;
        T al = this->matH[(i+1)*n+j]*c + this->matH[(i+1)*n+j+1]*s;

        T cr = 0-this->matH[i*n+j]*s +this->matH[i*n+j+1]*c;
        T br = 0-this->matH[i*n+j]*s +this->matH[i*n+j+1]*c;


        for(int i = 0; i < p; i++){
            T l_val = this->matH[i*n+j]*c +this->matH[i*n+j+1]*s;
            T r_val = 0-this->matH[i*n+j]*s +this->matH[i*n+j+1]*c;
            this->matH[i*n+j] = l_val;
            this->matH[i*n+j+1] = r_val;
        }

        // Eigen vector update 
        for(int i = 0; i < n; i++){
            T l_val = this->matQ[i*n+j]*c + this->matQ[i*n+j+1]*s;
            T r_val = 0-this->matQ[i*n+j]*s + this->matQ[i*n+j+1]*c;
            this->matQ[i*n+j] = l_val;
            this->matQ[i*n+j+1] = r_val;
        }
    }

    // elements other than on tri-diagonal are set to zero
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            this->matH[i*n+j] = (abs(i-j) > 1 ) ? 0 : this->matH[i*n+j];
        }
    }

    // adding back subtracted mu value
    // subtracting the shift from the main diagonal 
    for(int i = 0; i < p; i++){
        this->matH[i*n+i] += this->mu; 
    }
}



template<typename T> 
void Hess_QR_Decmp<T>::calculateShift(int p){

    T a = this->matH[(p-1)*n+p-1];
    T b = this->matH[(p-1)*n+p];
    T c = this->matH[p*n+ p];

    // computing the wilkinson shift
    T lamda = (c-a)/2.0;
    T sign = fabs(lamda) < 1e-6 ? 1 : lamda/fabs(lamda);
    this->mu = c - (sign *b*b)/(fabs(lamda) + sqrt(lamda*lamda+b*b));

}



template<typename T> 
void Hess_QR_Decmp<T>::do_hess_qr_iteration(){
    T threshold = 1e-6;
    int counter = 0;
    for(int p = n-1; p > 0; p--){
        for(int itr = 0; itr < 100; itr++){
            counter++;
            if(fabs(this->matH[p*n+p-1]) < threshold){
                // std::cout << "converging at itr: " << itr << " for p: " << p << "\n";
                break;
            }
            this->calculateShift(p);
            this->hess_qr_rq(p+1);
            
        }
    }
    std::cout << "Converged after: " << counter << " iterations\n";

}