#include<iostream>
#include<math.h> 
#include<cstdlib>
#include<algorithm>

#include "qr_decom.hpp"
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
    int n, p, debug;
    T *matA, *matdA, *vecU, *matC, *matC_tmp, *matQ, *matR;
    T *eigen_vecs, *eigen_vecs_tmp, *eigen_vals;
    int* sorted_index;
    T *matTrans;


 public: 
    PCA(int n, int p, int debug);
    ~PCA();
    void populate_A();
    void calculate_mean_vec();
    void calculate_deviation_vec();
    void calculate_covariance();
    void do_qrd_iteration(int n);
    void sort_eigen_vecs();
    T* do_pca_steps();
    T* get_eigen_vals();
    T* get_eigen_vecs();
    int* sort_eigen_vals();

};


template<typename T> PCA<T>::PCA(int n,int p, int debug = 0){
    this->n = n;
    this->p = p;
    this->debug = debug;
    this->matA = new T[n*p];
    this->matdA = new T[n*p];
    this->vecU = new T[p];
    this->matC = new T[p*p];
    this->matC_tmp = new T[p*p];
    this->matQ = new T[p*p];
    this->matR = new T[p*p];

    this->eigen_vecs= new T[p*p];
    this->eigen_vecs_tmp= new T[p*p];
    this->eigen_vals = new T[p];
    this->sorted_index = new int[p];
    this->matTrans = new T[p*p]; 

    // initialing the index for sorting
    for(int i = 0; i < p; i++){
        this->sorted_index[i] = i;
    }
}

template<typename T> PCA<T>::~PCA(){
    delete this->matA;
    delete this->matdA;
    delete this->vecU;
    delete this->matC;
    delete this->matC_tmp;
    delete this->matQ;
    delete this->matR;

    delete this->eigen_vecs;
    delete this->eigen_vecs_tmp;
    delete this->eigen_vals;
    delete this->sorted_index;
    delete this->matTrans;
}

F_type sample[20*5] = {
0.5434049417909654,0.27836938509379616,0.4245175907491331,0.8447761323199037,0.004718856190972565,
0.12156912078311422,0.6707490847267786,0.8258527551050476,0.13670658968495297,0.57509332942725,
0.891321954312264,0.20920212211718958,0.18532821955007506,0.10837689046425514,0.21969749262499216,
0.9786237847073697,0.8116831490893233,0.1719410127325942,0.8162247487258399,0.2740737470416992,
0.4317041836631217,0.9400298196223746,0.8176493787767274,0.3361119501208987,0.17541045374233666,
0.37283204628992317,0.005688507352573424,0.25242635344484043,0.7956625084732873,0.01525497124633901,
0.5988433769284929,0.6038045390428536,0.10514768541205632,0.38194344494311006,0.03647605659256892,
0.8904115634420757,0.9809208570123115,0.05994198881803725,0.8905459447285041,0.5769014994000329,
0.7424796890979773,0.6301839364753761,0.5818421923987779,0.020439132026923157,0.2100265776728606,
0.5446848781786475,0.7691151711056516,0.2506952291383959,0.2858956904068647,0.8523950878413064,
0.9750064936065875,0.8848532934911055,0.35950784393690227,0.5988589458757472,0.3547956116572998,
0.34019021537064575,0.17808098950580487,0.23769420862405044,0.04486228246077528,0.5054314296357892,
0.376252454297363,0.5928054009758866,0.6299418755874974,0.14260031444628352,0.933841299466419,
0.9463798808091013,0.6022966577308656,0.38776628032663074,0.3631880041093498,0.20434527686864423,
0.27676506139633517,0.24653588120354963,0.17360800174020508,0.9666096944873236,0.9570126003527981,
0.5979736843289207,0.7313007530599226,0.3403852228374361,0.09205560337723862,0.4634980189371477,
0.508698893238194,0.08846017300289077,0.5280352233180474,0.9921580365105283,0.3950359317582296,
0.3355964417185683,0.8054505373292797,0.7543489945823536,0.3130664415885097,0.6340366829622751,
0.5404045753007164,0.2967937508800147,0.11078790118244575,0.3126402978757431,0.4569791300492658,
0.6589400702261969,0.2542575178177181,0.6411012587007017,0.20012360721840317,0.6576248055289837

};

 // populating matrix a with random numbers
template<typename T> void PCA<T>::populate_A(){
    if(debug) std::cout << "Matrix A: \n";
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            this->matA[i*p+j] = sample[i*p+j]; //(1.0*std::rand())/RAND_MAX;
            if(debug) std::cout << this->matA[i*p+j] << " ";
        }
        if(debug) std::cout << "\n";
    }
}

template<typename T> void PCA<T>::calculate_mean_vec(){
    // setting initial vector value to zero
    for(int i = 0; i < p; i++){
        this->vecU[i] = 0;
    }

    // getting vector sum of the samples
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            this->vecU[j] += this->matA[i*p+j];
        }
    }
    if(debug) std::cout << "\nMean vector is: \n";
    
    // calculating the average
    for(int i = 0; i < p; i++){
        this->vecU[i] /= n;
        if(debug) std::cout << this->vecU[i] << " ";
    }
    if(debug) std::cout <<"\n";


}

template<typename T> void PCA<T>::calculate_deviation_vec(){
    //subtracting mean vec from all the sample
    if(debug) std::cout << "\n Deviation matrix is: \n";
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            this->matdA[i*p+j] = this->matA[i*p+j]-this->vecU[j];
            if(debug) std::cout << this->matdA[i*p+j] << " ";
        }
        if(debug) std::cout << "\n";
    }
}

template<typename T> void PCA<T>::calculate_covariance(){
    // covariance matrix matdA^{T} * matdA
    // this corresponds to matrix order pxp
    if(debug) std::cout << "\nCovariance matrix is: \n";
    for(int i = 0; i < p; i++){
        for(int j = 0; j < p; j++ ){
            this->matC[i*p+j] = 0;
            for(int k = 0; k < n; k++){
                this->matC[i*p+j] += this->matdA[k*p+i]*this->matA[k*p+j];
            }
            this->matC[i*p+j] = (1.0/(n-1))*this->matC[i*p+j];
            if(debug) std::cout << this->matC[i*p+j] << " ";
        }
        if(debug) std::cout << "\n";
    }
}


template<typename T> void PCA<T>::do_qrd_iteration(int itr){

    double tolerence;
    if(sizeof(T) == sizeof(double)){
        tolerence = 1.0e-18;
    } else if(sizeof(T) == sizeof(double)){
        tolerence = 1.0e-10;
    } else {
        tolerence = 1.0e-10;
    }

    // copy matrix C to matrix C_tmp
    for(int i = 0; i< p*p; i++){
        this->matC_tmp[i] = this->matC[i];
    }

    // setting initial value for eigen vector
    // everything is zero except digonal elements
    for(int i =0; i < p; i++){
        for(int j = 0; j < p; j++){
            if(i==j){this->eigen_vecs[i*p+j] = 1;} else {this->eigen_vecs[i*p+j] = 0;}
        }
    }


    QR_Decmp<T> qr_decmp(this->matC_tmp, p);
    //QRD iteration
    for(int itr_i = 0;itr_i < itr; itr_i++){
        // shift value calculation using Wilkinson's shift
        T w_a = this->matC_tmp[p*(p-2)+p-2];
        T w_b = this->matC_tmp[p*(p-2)+p-1];
        T w_c = this->matC_tmp[p*(p-1)+p-1];
        T delta = (w_a - w_c)/2;

        double eigen_sq_diff = 0;
        double eigen_total = 0;

        int sign = (delta > 0) ? 1 : -1;

        T s_val = w_c - (sign*w_b*w_b)/(fabs(delta) + sqrt(delta*delta+w_b*w_b));

        // T s_val = this->matC_tmp[p*p-1];
        // subtract that from all diagonal elements of matC_tmp
        for(int i = 0; i < p; i++){
            this->matC_tmp[i*p+i] -= s_val;
        }

        // QR decompostion of matrix C_tmp
        qr_decmp.QR_decompose();

        T* Q = qr_decmp.get_Q();
        T* R = qr_decmp.get_R();

        // Q matrix
        if(this->debug){
            std::cout << "\nMatrix Q is : \n";
            for(int i =0; i < p; i++){
                for(int j = 0; j < p; j++){
                    std::cout << Q[i*p+j] << " ";
                }
                std::cout << "\n";
            }
        }

        // R matrix
        if(this->debug){
            std::cout << "\nMatrix R is : \n";
            for(int i =0; i < p; i++){
                for(int j = 0; j < p; j++){
                    std::cout << R[i*p+j] << " ";
                }
                std::cout << "\n";
            }
        }

        // R*Q calculation
        for(int i = 0; i < p; i++){
            for(int j = 0; j < p; j++){
                this->matC_tmp[i*p+j] = 0;
                for(int k = 0; k < p; k++){
                    this->matC_tmp[i*p+j] += R[i*p+k]*Q[k*p+j];
                }
            }
        }

        // adding back subtracted s_val
        for(int i = 0; i < p; i++){
            this->matC_tmp[i*p+i] += s_val;
        }


        //coping old eigen vecs 
        for(int i = 0; i < p*p; i++){
            this->eigen_vecs_tmp[i] = this->eigen_vecs[i];
        }

        // upating the eigen vectors 
        for(int i = 0; i < p; i++){
            for(int j = 0; j < p; j++){
                this->eigen_vecs[i*p+j] = 0;
                for(int k = 0; k < p; k++){
                    this->eigen_vecs[i*p+j] += this->eigen_vecs_tmp[i*p+k]*Q[k*p+j];
                }
            }
        }


        for(int i = 0; i < p; i++){
            eigen_sq_diff += (this->eigen_vals[i]- this->matC_tmp[i*p+i])*(this->eigen_vals[i]- this->matC_tmp[i*p+i]);
            eigen_total += this->matC_tmp[i*p+i]*this->matC_tmp[i*p+i];
        }

        for(int i = 0; i < p; i++){
            this->eigen_vals[i] = this->matC_tmp[i*p+i];
        }



        if(eigen_sq_diff/eigen_total < tolerence){
            std::cout << "Convergence achieved at iteration: " << itr_i << "\n\n";
            break;
        }

    }

}


template<typename T> T* PCA<T>::do_pca_steps(){
    this->populate_A();
    this->calculate_mean_vec();
    this->calculate_deviation_vec();
    this->calculate_covariance();
    this->do_qrd_iteration(200);

    return this->matC_tmp;
}

template<typename T> int* PCA<T>::sort_eigen_vals(){
    std::sort(this->sorted_index, this->sorted_index+p, [&] (int i, int j) \
    {return this->eigen_vals[i] > this->eigen_vals[j];});
    return this->sorted_index;
}


template<typename T> T* PCA<T>::get_eigen_vals(){
    return this->eigen_vals;
}


int main(){
    
    int n = 20, p = 5;
    PCA< F_type> pca(n, p, 0);
    F_type * cov = pca.do_pca_steps();
    F_type * eigen_vals = pca.get_eigen_vals();

    std::cout << "\nCovariance matrix is: \n";
    for(int i = 0; i < p; i++){
        for(int j = 0; j < p; j++){
            std::cout << cov[i*p+j] << " ";
        }
        std::cout << "\n";
    }

    int* index_ptr = pca.sort_eigen_vals();
    std::cout << "\nEigen values are: \n";
    for(int i = 0; i < p; i++){
        // std::cout << index_ptr[i] << " ";
        std::cout << eigen_vals[pca.sorted_index[i]] << " ";
    }
    std::cout << "\n\n";


    // Eigen vectors 
    std::cout << "\ncorresponding eigen vectors are: \n";
    for(int i =0; i < p; i++){
        for(int j = 0; j < p; j++){
            std::cout << pca.eigen_vecs[j*p+pca.sorted_index[i]] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";

    return 0;

}