#include <bits/stdc++.h>
#include <CL/sycl.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <oneapi/dpl/random>
#include "mkl/mkl.h"

extern "C"{void genmat_(double *A, double *B, double n);}
using namespace sycl;

// Selectors for specific targets
#ifdef _GPU
gpu_selector selector;
#elif _CPU
cpu_selector selector;
#else
default_selector selector;
#endif
typedef double real;

extern void print_matrix( char* desc, int m, int n, double* a, int lda);

#define PI 3.14159265
// Program variables, feel free to change anything
// make run 30000 1e-10 1e-15 -1000 1000 100 777
const real result_error = 1e-10;
const real calculation_error = 1e-15;
const real min_rand = -1000.0;
const real max_rand = 1000.0;
const std::uint32_t seed = 666;
int max_sweeps = 100;
static const int N = 3;

std::ofstream outfile;

real* generate_matrix(real* matrix, int N);

void print_matrix(std::vector<real> matrix, std::vector<real> results, int N);

void print_results(real* data, int N);

bool check_if_finished(real* matrix, int N);

real find_determinant(real* matrix, int n);

void multiply(real* &P, real* &matrix, real* &P_inv, int N);

int main(int argc, char *argv[])
{  
    auto begin_runtime = std::chrono::high_resolution_clock::now();
    queue q(selector);

    // outfile.open("report.txt", std::ios_base::out);

    real* A = malloc_shared<real>(N*N, q);
    memset(A, 0, sizeof(*A)*N*N);
    real* B = malloc_shared<real>(N*N, q);
    memset(B, 0, sizeof(*B)*N*N);
    real* matrix = malloc_shared<real>(N*N, q);
    memset(matrix, 0, sizeof(*matrix)*N*N);

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;

    auto begin_matrix = std::chrono::high_resolution_clock::now();
    
    q.submit([&](handler& h){
    stream out(1024, 256, h);
    h.parallel_for(range<1>(N), [=](id<1> id){
        int i = id;


        for(int j=i*N+i; j<i*N+N; ++j) 
        {
            oneapi::dpl::minstd_rand engine(seed, j);
            oneapi::dpl::uniform_real_distribution<real> distr(-2, 8);
            A[j] = distr(engine);
        }     
        });
    });
    q.wait();

     for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << A[j] << " ";
        }
        std::cout<<std::endl;
    }

    std::cout<<std::endl;
    q.submit([&](handler& h){
        stream out(1024, 256, h);
        h.parallel_for(range<1>(N*N), [=](id<1> id){
            int i = id/N;
            int j = id%N;
            B[id] = A[N*j+i];
            
        });
    });
    q.wait();

    for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << B[j] << " ";
        }
        std::cout<<std::endl;
    }

    q.submit([&](handler& h){
        h.parallel_for(range<1>(N), [=](id<1> i){  
            for(int j=0; j<N; j++) {
            real tmp = 0;
            for(int l=0; l<N; l++) {
                tmp += A[N*i+l]*B[N*l+j];
            }
            matrix[N*i + j] = tmp;
        }
        });
    });
    q.wait();

    std::cout<<std::endl;    
    
    // genmat_(matrix, B, N);
    for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            // if(j%N==j/N) matrix[j]*=10;
            std::cout << matrix[j] << " ";
        }
        std::cout<<std::endl;
    }

    real sqr = sqrt(2);

    matrix[0] = 1;
    matrix[1] = sqr;
    matrix[2] = 2;
    matrix[3] = sqr;
    matrix[4] = 3;
    matrix[5] = sqr;
    matrix[6] = 2;
    matrix[7] = sqr;
    matrix[8] = 1;

    auto end_matrix = std::chrono::high_resolution_clock::now();
    auto elapsed_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(end_matrix - begin_matrix);

    std::cout << "\nMatrix generated, time elapsed: " << elapsed_matrix.count() * 1e-9 << " seconds.\n";
    real ei[N*N];
    for(int i=0; i<N*N; ++i) ei[i] = matrix[i];

    for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << ei[j] << " ";
        }
        std::cout<<std::endl;
    }

    std::cout<<std::endl;

    auto begin_computations = std::chrono::high_resolution_clock::now();

    bool finished = false;
    int sweeps = 0;
    

    int *position = malloc_shared<int> (N, q);
    real* P = malloc_shared<real>(N*N, q);
    real* P_inv = malloc_shared<real>(N*N, q);

    // The main functionality of the Jacobi Solver. 
    // Every iteration calculates new values until 
    // there are no changes detected between the values
    // calculated this iteration and the one before.
    // Casting to double had to be added in this place
    // as the error calculation could be invalid for a very 
    // small error rate because of the float type representation.
    do{
        q.submit([&](handler& h){
            stream out(1024, 256, h);
            h.parallel_for(range<1>(N), [=](id<1> id){  
                position[id]=0;
                real maximum = 0;

                for(int j=N*id; j<N*id+N; ++j)
                {
                    if(j!=N*id+id && matrix[j]>maximum)
                    {
                        maximum = matrix[j];
                        position[id] = j;
                    }
                }
            });
        }).wait();
        
        int it = position[0];
        real maximum = matrix[it];

        for(int z=1; z<N; ++z) 
        {
            if(matrix[position[z]]>maximum)
            {
                it = position[z];
                maximum = matrix[position[z]];
            }
        }

        int a = it/N;
        int b = it%N;

        double theta = atan((2*matrix[N*a+b])/(matrix[N*a+a]-matrix[N*b+b]))/2.;

        for(int i=0;i<N;++i)
            for(int j=i*N;j<N*(i+1);++j)
                if(j%N==i) P[j] = 1;
                else P[j] = 0;
        
        P[a*N+b]=sin(theta);
        P[b*N+a]=-sin(theta);
        P[a*N+a]=cos(theta);
        P[b*N+b]=cos(theta);
        
        
        int lda = N;
        int ipiv = N;
        int lwork = N;
        int info;
        double* work = (double*)malloc(lwork*sizeof(double));
        
        q.submit([&](handler& h){
            h.parallel_for(range<1>(N*N), [=](id<1> id){  
                P_inv[id] = P[id];
            });
        }).wait();

        dgetrf(&N, &N, P_inv, &lda, &ipiv, &info);
        dgetri(&N, P_inv, &lda, &ipiv, work, &lwork, &info);

        
        multiply(P, matrix, P_inv, N);

        for(int i=0; i<N; ++i)
        {
            for(int j=i*N; j<N*(i+1); ++j)
            {
                std::cout << matrix[j] << " ";
            }
            std::cout<<std::endl;
        }        
        free( (void*)work );

        ++sweeps;

        finished=check_if_finished(matrix, N);
        
    }while(finished!=true && sweeps<100); //!is_equal && sweeps<max_sweeps);

    for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << matrix[j] << " ";
        }
        std::cout<<std::endl;
    }
    auto end_computations = std::chrono::high_resolution_clock::now();
    auto elapsed_computations = std::chrono::duration_cast<std::chrono::nanoseconds>(end_computations - begin_computations);

    std::cout << "\nComputations complete, time elapsed: " << elapsed_computations.count() * 1e-9 << " seconds.\n";
    std::cout << "Total number of sweeps: " << sweeps << std::endl;
    std::cout << "Checking results\n";

    ////////////////////////////////////////////////////////////////////////////////////////////
    {
    int LDA = N;
    int n = N, lda = LDA, info, lwork, liwork;
    int iwkopt;
    int* iwork;
    double wkopt;
    double* work;
    /* Local arrays */
    double w[N];

    /* Executable statements */
    printf( " DSYEVD Example Program Results\n" );
    /* Query and allocate the optimal workspace */
    lwork = -1;
    liwork = -1;
    dsyevd_("Vectors", "Upper", &n, ei, &lda, w, &wkopt, &lwork, &iwkopt, &liwork, &info);
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double));
    liwork = iwkopt;
    iwork = (int*)malloc( liwork*sizeof(int) );
    /* Solve eigenproblem */
    dsyevd_( "Vectors", "Upper", &n, ei, &lda, w, work, &lwork, iwork,
                    &liwork, &info );
    /* Check for convergence */
    if( info > 0 ) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
    }
    /* Print eigenvalues */
    print_matrix( (char*)"Eigenvalues", 1, n, w, 1 );
    /* Print eigenvectors */
    print_matrix( (char*)"Eigenvectors (stored columnwise)", n, n, ei, lda );
    /* Free workspace */
    free( (void*)iwork );
    free( (void*)work );
    }
    auto begin_check = std::chrono::high_resolution_clock::now();

    auto end_check = std::chrono::high_resolution_clock::now();
    auto elapsed_check = std::chrono::duration_cast<std::chrono::nanoseconds>(end_check - begin_check);

    std::cout << "\nCheck complete, time elapsed: " << elapsed_check.count() * 1e-9 << " seconds.\n";

    auto end_runtime = std::chrono::high_resolution_clock::now();
    auto elapsed_runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end_runtime - begin_runtime);

    std::cout << "Total runtime is " << elapsed_runtime.count() * 1e-9 << " seconds.\n";
    
    free(position, q); 
    free(matrix, q);
    free(A, q);
    free(B, q);
    // outfile.close();
    
    return 0; 
}

// Function responsible for generating a float type
// diagonally dominant matrix. Float had to be used 
// as using double would result in segmentation faults
// for extreamlly large matrixes. This is also an example
// of using sycl based RNG which had to be used as using
// external (non sycl) functions slows down the execution
// drasticly.

//symetric positive definite, xto converg
real* generate_matrix(real* matrix, int N)
{
    queue q(selector);   

    real *mat_T = malloc_shared<real>(N*N, q);
    q.submit([&](handler& h){
        stream out(1024, 256, h);
        h.parallel_for(range<1>(N), [=](id<1> id){
            int i = id;

            oneapi::dpl::minstd_rand engine(seed, i);

            oneapi::dpl::uniform_real_distribution<real> distr(0, 5);

            for(int j=i*N; j<=N*i+i; ++j)
            {
                matrix[j] = distr(engine);
                // matrix[j] = round(10. * matrix[j]) / 10.;
            }
        });
    });
    q.wait();

    for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << matrix[j] << " ";
            
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    q.submit([&](handler& h){
        stream out(1024, 256, h);
        h.parallel_for(range<1>(N*N), [=](id<1> id){
            int i = id/N;
            int j = id%N;
            mat_T[id] = matrix[N*j+i];
            
        });
    });
    q.wait();

    for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << mat_T[j] << " ";            
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    real *C = new real(N*N);
    memset(C, 0, sizeof(*C)*N*N);
    double alpha = 1.0; 
    double beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                N, N, N, alpha, matrix, N, mat_T, N, beta, C, N);

    for(int i=0; i<N; ++i)
    {
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << C[j] << " ";
            
        }
        std::cout<<std::endl;
    }

    return C;
}
// Function responsible for printing the matrix, called only for N < 10 
void print_matrix(std::vector<real> matrix, std::vector<real> results, int N)
{
    for(int i=0; i<N; ++i)
    {
        std::cout << '[';
        for(int j=i*N; j<N*(i+1); ++j)
        {
            std::cout << matrix[j] << " ";
        }
        std::cout << "][" << results[i] << "]\n";
    }
}

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}

// Function responsible for printing the results
void print_results(real* data, int N)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(11);
    for(int i=0; i<N; ++i) std::cout << "X" << i+1 << " equals: " << data[i] << std::endl;
}
// Function responsible for checking if the algorithm has finished.
// TRUE convergence stops, FALSE convergence continues
bool check_if_finished(real* matrix, int N)
{
    bool result=true;
    for(int i = 0; i < N; i++)
    {   
        for(int j=i*N;j<i*N+3;++j)
        {
            if(j%N!=i && matrix[j]!=0) result=false;
        }
    }

    return result;
}
// Function responsible for multiplying three matrices
// using parallel_for 
void multiply(real* &P, real* &matrix, real* &P_inv, int N)
{
    queue q(selector);

    real* temp = malloc_shared<real>(N*N,q);
    real* result = malloc_shared<real>(N*N,q);

    q.submit([&](handler& h){
        h.parallel_for(range<1>(N), [=](id<1> i){  
            for(int j=0; j<N; j++) {
            real tmp = 0;
            for(int l=0; l<N; l++) {
                tmp += P_inv[N*i+l]*matrix[N*l+j];
            }
            temp[N*i + j] = tmp;
        }
        });
    }).wait();

    q.submit([&](handler& h){
        h.parallel_for(range<1>(N), [=](id<1> i){  
            for(int j=0; j<N; j++) {
            real tmp = 0;
            for(int l=0; l<N; l++) {
                tmp += temp[N*i+l]*P[N*l+j];
            }
            result[N*i + j] = tmp;
        }
        });
    }).wait();

    q.submit([&](handler& h){
        stream out(1024, 256, h);
        h.parallel_for(range<1>(N*N), [=](id<1> i){  
            if(fabs(result[i])<1e-15)  result[i] = 0;
            matrix[i]=result[i];
        });
    }).wait();
}