#include <bits/stdc++.h>
#include <CL/sycl.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <oneapi/dpl/random>

using namespace sycl;

// Selectors for specific targets
//#ifdef _GPU
gpu_selector selector;
//#elif _CPU
//cpu_selector selector;
//#else
//default_selector selector;
//#endif
typedef double real;

// Program variables, feel free to change anything 
// make run 30000 0.000000001 -1000 1000 100 777
static const int N = 3;
static const real check_error = 1e-15;
static const real calculation_error = 1e-10;
static const int min_rand = -1000;
static const int max_rand = 1000;
int max_sweeps = 100;
static const std::uint32_t seed = 666;

std::ofstream outfile;

// Function responsible for generating a float type
// diagonally dominant matrix. Float had to be used 
// as using double would result in segmentation faults
// for extreamlly large matrixes. This is also an example
// of using sycl based RNG which had to be used as using
// external (non sycl) functions slows down the execution
// drasticly.

// Function responsible for printing the matrix, called only for N < 10 
void print_matrix(std::vector<float> matrix, std::vector<real> results)
{
    for(int i=0; i<N; ++i)
    {
        outfile << '[';
        for(int j=i*N; j<N*(i+1); ++j)
        {
            outfile << matrix[j] << " ";
        }
        outfile << "][" << results[i] << "]\n";
    }
}
// Function responsible for printing the results
void print_results(real* data, int N)
{
    outfile << std::fixed;
    outfile << std::setprecision(11);
    for(int i=0; i<N; ++i) outfile << "X" << i+1 << " equals: " << data[i] << std::endl;
}
// Function responsible for checking if the algorithm has finished
bool check_if_equal(real *data, real *old_values)
{
    int number = 0;

    for(int i = 0; i < N; ++i)
    {
        if(fabs(data[i]-old_values[i])<check_error) number++;
    }

    return number==N;
}

int main(int argc, char *argv[])
{  
//     for(int i =0; i<argc; ++i)  std::cout << argv[i] << std::endl;
//     auto begin_runtime = std::chrono::high_resolution_clock::now();

// //    outfile.open("report.txt", std::ios_base::out);

//     float matrix[N*N] = {0,0,0,0,0,0,0,0,0};
//     real results[N] = {0,0,0};

//     queue q(selector);

//   //  outfile << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;

//     auto begin_matrix = std::chrono::high_resolution_clock::now();

//    /* generate_matrix(matrix, results);
//    for(int i=0; i<N; ++i)
//     {
//         std::cout << '[';
//         for(int j=i*N; j<N*(i+1); ++j)
//         {
//             std::cout << matrix[j] << " ";
//         }
//         std::cout << "][" << results[i] << "]\n";
//     }
//     buffer buf_mat(matrix);
//     buffer buf_res(results);*/
// 	std::vector<queue> qa;
//     for (const auto &p : platform::get_platforms()) {
//         if (p.get_info<info::platform::name>().find("Level-Zero") != std::string::npos){
//         for (const auto &d : p.get_devices()) {
//             if(d.is_gpu() && d.get_info<info::device::name>().find("Intel") != std::string::npos){
//                 qa.push_back(queue(d));
//                 std::cout << "--Found GPU: " << d.get_info<info::device::name>() << "\n";
//             }
//         }
//       }
//     }
//     std::cout << qa.size() << std::endl;

//     {
//     buffer buf_mat(matrix, range(N*N));
//     buffer buf_res(results, range(N));

//     qa[0].submit([&](handler& h){
//         stream out(1024, 256, h);
//         accessor M {buf_mat, h};
//         accessor R {buf_res, h};
//         h.parallel_for(N/2, [=](auto id){
//             int i = id;
//             int j = N*i;
//             int it = N*i+i;

//             real sum = 0;

//             oneapi::dpl::minstd_rand engine(seed, i+j);

//             oneapi::dpl::uniform_real_distribution<real> distr(min_rand, max_rand);

//             for(int j=i*N; j<N*(i+1); ++j)
//             {
//                 M[j] = distr(engine);
//                 M[j] = round(100. * M[j]) / 100.;
//                 sum += fabs(M[j]);
//             }

//             oneapi::dpl::uniform_int_distribution<int> distr2(0, 100);
//             int gen_neg = distr2(engine);

//             if(gen_neg<50) M[i*N+i] = sum +1;
//             else M[i*N+i] = -1*(sum +1);

//             R[i] = distr(engine);
//         });
//     });
//     }
//     std::cout << "Q1 finished\n";
//     {
//         buffer buf_mat(matrix, range(N*N));
//         buffer buf_res(results, range(N));
    
//         qa[1].submit([&](handler& h){   
//             stream out(1024, 256, h);
//             accessor M {buf_mat, h};
//             accessor R {buf_res, h};
//             h.parallel_for(N/2, [=](auto id){
//                 int i = N/2+id;
//                 int j = N*i;
//                 int it = N*i+i;

//                 real sum = 0;

//                 oneapi::dpl::minstd_rand engine(seed, i+j);

//                 oneapi::dpl::uniform_real_distribution<real> distr(min_rand, max_rand);

//                 for(int j=i*N; j<N*(i+1); ++j)
//                 {
//                     M[j] = distr(engine);
//                     M[j] = round(100. * M[j]) / 100.;
//                     sum += fabs(M[j]);
//                 }

//                 oneapi::dpl::uniform_int_distribution<int> distr2(0, 100);
//                 int gen_neg = distr2(engine);

//                 if(gen_neg<50) M[i*N+i] = sum +1;
//                 else M[i*N+i] = -1*(sum +1);

//                 R[i] = distr(engine);
//             });
//         });
//     }
//     auto end_matrix = std::chrono::high_resolution_clock::now();
//     auto elapsed_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(end_matrix - begin_matrix);

//     std::cout << "\nMatrix generated, time elapsed: " << elapsed_matrix.count() * 1e-9 << " seconds.\n";

//     // if(N<10) print_matrix(matrix, results);

//     auto begin_computations = std::chrono::high_resolution_clock::now();

//     real *data = malloc_shared<real>(N, q);
//     real *old_values = malloc_shared<real>(N, q);

//     for(int i=0; i<N; i++) data[i] = 0;

//     bool is_equal = false;
//     int sweeps = 0;
    
//     // The main functionality of the Jacobi Solver. 
//     // Every iteration calculates new values until 
//     // there are no changes detected between the values
//     // calculated this iteration and the one before.
//     // Casting to double had to be added in this place
//     // as the error calculation could be invalid for a very 
//     // small error rate because of the float type representation.
//     {
//         buffer buf_mat(matrix, range(N*N));
//         buffer buf_res(results, range(N));
//         do{
//             for(int i=0; i<N;++i) old_values[i] = data[i];
//             qa[0].submit([&](handler& h){
//                 accessor M {buf_mat, h, read_only};
//                 accessor R {buf_res, h, read_only};
//                 h.parallel_for(range<1>(N), [=](id<1> id){
//                     int i = id;
//                     int j = N*i;
//                     int it = N*i+i;

//                     data[i] = R[i];
//                     for(int z=0; z<N; ++z){
//                         if(z!=i) 
//                             data[i] = data[i] - (old_values[z] * static_cast<real>(M[j])); 
//                         j=j+1;}                
//                     data[i] = data[i]/static_cast<real>(M[it]);
//                 });
//             });

//             qa[1].submit([&](handler& h){
//                 accessor M {buf_mat, h, read_only};
//                 accessor R {buf_res, h, read_only};
//                 h.parallel_for(range<1>(N/2), [=](id<1> id){
//                     int i = N/2+id;
//                     int j = N*i;
//                     int it = N*i+i;

//                     data[i] = R[i];
//                     for(int z=0; z<N; ++z){
//                         if(z!=i) 
//                             data[i] = data[i] - (old_values[z] * static_cast<real>(M[j])); 
//                         j=j+1;}                
//                     data[i] = data[i]/static_cast<real>(M[it]);
//                 });
//             });
            
//             qa[0].wait();
//             qa[1].wait();
            
//             ++sweeps;
//             is_equal = check_if_equal(data, old_values);
//         }while(!is_equal && sweeps<max_sweeps);
//     }
//     auto end_computations = std::chrono::high_resolution_clock::now();
//     auto elapsed_computations = std::chrono::duration_cast<std::chrono::nanoseconds>(end_computations - begin_computations);

//     std::cout << "\nComputations complete, time elapsed: " << elapsed_computations.count() * 1e-9 << " seconds.\n";
//     std::cout << "Total number of sweeps: " << sweeps << std::endl;
//     std::cout << "Checking results\n";

//     auto begin_check = std::chrono::high_resolution_clock::now();

//     std::vector<real> new_results(N, 0);

//     for(int i=0; i<N*N; ++i)
//     {
//         new_results[i/N] += data[i%N]*static_cast<real>(matrix[i]);
//     } 
 
//     bool *all_eq = malloc_shared<bool>(1, q);
//     all_eq[0] = true;
//     buffer buf_new_res(new_results);

//     {
//         buffer buf_mat(matrix, range(N*N));
//         buffer buf_res(results, range(N));
//         qa[0].submit([&](handler& h){
//             stream out(1024, 256, h);
//             accessor R {buf_res, h};
//             accessor NR {buf_new_res, h};
//             h.parallel_for(range<1>(N), [=](id<1> id){       
//                 real diff = fabs(NR[id]-R[id]);
//                 if(diff>calculation_error) all_eq[0] = false;
//             });
//         });
//     }
//     if(all_eq[0]) outfile << "All values are correct.\n";
//     else outfile << "There has been some errors. The values are not correct.\n";

//     auto end_check = std::chrono::high_resolution_clock::now();
//     auto elapsed_check = std::chrono::duration_cast<std::chrono::nanoseconds>(end_check - begin_check);

//     outfile << "\nCheck complete, time elapsed: " << elapsed_check.count() * 1e-9 << " seconds.\n";

//     auto end_runtime = std::chrono::high_resolution_clock::now();
//     auto elapsed_runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end_runtime - begin_runtime);

//     outfile << "Total runtime is " << elapsed_runtime.count() * 1e-9 << " seconds.\n";

//     print_results(data, N);
//     free(data, q);
//     free(old_values, q);

    return 0; 
}