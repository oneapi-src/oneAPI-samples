//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Hidden Markov Models: this code sample implements the Viterbi algorithm which is a dynamic 
// programming algorithm for findingthe most likely sequence of hidden states—
// called the Viterbi path—that results in a sequence of observed events,
// especially in the context of Markov information sources and HMM.
//
// - Initially, the dataset for algorithm processing is generated : initial states probability 
// distribution Pi, transition matrix A, emission matrix Band the sequence or the observations 
// produced by hidden Markov process.
// - First, the matrix of Viterbi values on the first states are initialized using distribution Pi 
// and emission matrix B.The matrix of back pointers is initialized with default values - 1.
// - Then, for each time step the Viterbi matrix is set to the maximal possible value using A, B and Pi.
// - Finally, the state with maximum Viterbi value on the last step is set as a final state of 
// the Viterbi pathand the previous nodes of this path are detemined using the correspondent rows 
// of back pointers matrix for each of the steps except the last one.
//
// Note: The implementation uses logarithms of the probabilities to process small numbers correctly
// and to replace multiplication operations with addition operations.

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include <math.h>
#include <iostream>
#include <cstdio>
#include <ctime>

using namespace sycl;
using namespace std;

// Matrix size constants.
// The number of hidden states N.
constexpr int N = 20;
// The number of possible observations M.
constexpr int M = 20;
// The lenght of the hidden states sequence T.
constexpr int T = 20;
// The parameter for generatinf the sequence.
constexpr int seed = 0;
// Minimal double to initialize  logarithms for Viterbi values equal to 0.
constexpr double MIN_DOUBLE = -1.0 * std::numeric_limits<double>::max();

bool ViterbiCondition(double x, double y, double z, double compare);

int main() {
    // Initializing and generating initial probabilities for the hidden states.
    double(*Pi) = new double[N];
    for (int i = 0; i < N; ++i) {
        Pi[i] = sycl::log10(1.0f / N);
    }

    // Сatch asynchronous exceptions.
    auto exception_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
            }
        }
    };
    
    try {
        //Device initialization.
        queue q(default_selector{}, exception_handler);
        cout << "Device: " << q.get_device().get_info<info::device::name>() << " "
            << q.get_device().get_platform().get_info<info::platform::name>() << "\n";

        //Buffers initialization.
        buffer<double, 2> Viterbi(range<2>(N, T));
        buffer<int, 2> Back_pointer(range<2>(N, T));
        buffer<double, 2> A(range<2>(N, N));
        buffer<double, 2> B(range<2>(N, M));

        // Generating transition matrix A for the Markov process.
        q.submit([&](handler& h) {
            auto accessor = A.get_access<access::mode::write>(h);
            h.parallel_for(range<2>(N, N), [=](id<2> index) {
                // The sum of the probabilities in each row of the matrix A  has to be equal to 1.
                double Prob = 1.0f / N;
                // The algorithm computes logarithms of the probability values to improve small numbers processing
                accessor[index] = sycl::log10(Prob);
                });
            });

        // Generating emission matrix B for the Markov process.
        q.submit([&](handler& h) {
            auto accessor = B.get_access<access::mode::write>(h);
            h.parallel_for(range<2>(N, M), [=](id<2> index) {
                // The sum of the probabilities in each row of the matrix B has to be equal to 1.
                double Prob = ((index[0] + index[1]) % M) * 2.0f / M / (M - 1);
                // The algorithm computes logarithms of the probability values to improve small numbers processing
                accessor[index] = (Prob == 0.0f) ? MIN_DOUBLE : sycl::log10(Prob);
                });
            });

        // Generating the sequence of the observations produced by the hidden Markov chain.
        int(*seq) = new int[T];
        for (int i = 0; i < T; ++i) {
            seq[i] = (i * i + seed) % M;
        }
        buffer<int, 1> seq_buf(seq, T);

        // Initialization of the Viterbi matrix and the matrix of back pointers.
        q.submit([&](handler& h) {
            auto V = Viterbi.get_access<access::mode::read_write>(h);
            auto BP = Back_pointer.get_access<access::mode::write>(h);
            auto b = B.get_access<access::mode::read>(h);
            h.parallel_for(range<2>(N, T), [=](id<2> index) {
                int i = index[0];
                int j = index[1];
                // At starting point only the first Viterbi values are defined and these Values are substituted 
                // with logarithms  due to the following equation: log(x*y) = log(x) + log(y).
                V[index] = (j != 0) ? MIN_DOUBLE : Pi[i] + b[i][seq[0]];
                // Default values of all the back pointers are (-1) to show that they are not determined yet. 
                BP[index] = -1;
                });
            });
        delete[] Pi;

        // The sequential steps of the Viterbi algorithm that define the Viterbi matrix and the matrix 
        // of back pointers. The product of the Viterbi values and the probabilities is substituted with the sum of 
        // the logarithms due to the following equation: log (x*y*z) = log(x) + log(y) + log(z)
        for (int j = 0; j < T - 1; ++j) {
            q.submit([&](handler& h) {
                auto V = Viterbi.get_access<access::mode::read_write>(h);
                auto BP = Back_pointer.get_access<access::mode::read_write>(h);
                auto a = A.get_access <access::mode::read>(h);
                auto b = B.get_access <access::mode::read>(h);
                auto seq_acc = seq_buf.get_access <access::mode::read>(h);

                h.parallel_for(range<2>(N, N), [=](id<2> index) {
                    int i = index[0], k = index[1];
                    // This conditional block finds the maximum possible Viterbi value on the current step j for the step i.
                    if (ViterbiCondition(V[k][j], b[i][seq_acc[j + 1]], a[k][i], V[i][j + 1])) {
                        V[i][j + 1] = V[k][j] + a[k][i] + b[i][seq_acc[j + 1]];
                        BP[i][j + 1] = k;
                    }
                });
            });
        }
        delete[] seq;

        // Getting the Viterbi path based on the matrix of back pointers
        buffer<int, 1> ViterbiPath(range<1> {T});
        auto V = Viterbi.get_access<access::mode::read>();
        auto BP = Back_pointer.get_access<access::mode::read>();
        auto Path = ViterbiPath.get_access<access::mode::read_write>();
        double Max = MIN_DOUBLE;
        // Constructing the Viterbi path. The last state of this path is the one with the biggest Viterbi value (the most likely state).
        for (int i = 0; i < N; ++i) {
            if (V[i][T - 1] > Max) {
                Max = V[i][T - 1];
                Path[T - 1] = i;
            }
        }

        for (int i = T - 2; i >= 0; --i) {
            Path[i] = BP[Path[i + 1]][i + 1];
        }

        cout << "The Viterbi path is: ";
        for (int k = 0; k < T; ++k) {
            cout << Path[k] << " ";
        }
        cout << std::endl;

    } catch (sycl::exception const& e) {
        // Exception processing
        cout << "An exception is caught!\n";
        cout << "Error message:" << e.what();
        terminate();
    }
    cout << "The sample completed successfully!" << std::endl;
    return 0;
}

// The method checks if all three components of the sum are not equivalent to logarithm of zero 
// (that is incorrect value and is substituted with minimal possible value of double) and that 
// the Viterbi value on the new step exceeds the current one.
bool ViterbiCondition(double x, double y, double z, double compare) {
    return (x > MIN_DOUBLE) && (y > MIN_DOUBLE) && (z > MIN_DOUBLE) && (x + y + z > compare);
}
