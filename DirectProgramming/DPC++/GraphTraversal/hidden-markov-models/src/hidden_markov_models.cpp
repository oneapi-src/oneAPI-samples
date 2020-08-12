//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Hidden Markov Model. Add description of Viterbi algorithm
//

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include <math.h>
#include <dpc_common.hpp>
#include <iostream>

using namespace sycl;
using namespace std;

// Matrix size constants.
constexpr int N = 20;
constexpr int M = 15;
constexpr int T = 100;
constexpr int seed = 0;

int main() {
	//float A[N][N];// = { {0.6, 0.4}, {0.5, 0.5} };
	//float B[N][M];// = { {0.2, 0.4, 0.4}, {0.5, 0.4, 0.1} };
	float Pi[N];// = { 0.8, 0.2 };

	for (int i = 0; i < N; ++i) {
		Pi[i] = log(1.0f / N);
	}

	//Device initialization
	try {
		queue q(default_selector{}, dpc_common::exception_handler);
		cout << "Device: " << q.get_device().get_info<info::device::name>() << " " 
			<< q.get_device().get_platform().get_info<info::platform::name>()<< "\n";

		buffer<double, 2> Viterbi(range<2>(N, M));
		buffer<int, 2> Back_pointer(range<2>(N, M));
		buffer<double, 2> A(range<2>(N, N));
		buffer<double, 2> B(range<2>(N, M));

		q.submit([&](handler &h) {
			auto accessor = A.get_access<access::mode::write>(h);
			h.parallel_for(range<2>(N, N), [=](id<2> index) {
				accessor[index] = log(1.0 / N);
			});
		});

		q.submit([&](handler &h) {
			auto accessor = B.get_access<access::mode::write>(h);
			h.parallel_for(range<2>(N, M), [=](id<2> index) {
				double Prob = ((index[0] + index[1]) % M) * 2.0f / M / (M - 1);
				accessor[index] = (Prob == 0.0f) ? (DBL_MAX * (-1.0)) : log(Prob) ;
				});
			});
		/*for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				A[i][j] = 1.0f / N;
			}
		}*/
		int seq[T];// = { 2, 0, 2 };
		for (int i = 0; i < T; ++i) {
			seq[i] = (i * i + seed) % M;
		}

		q.submit([&](handler &h) {
			auto V = Viterbi.get_access<access::mode::write>(h);
			auto BP = Back_pointer.get_access<access::mode::write>(h);
			auto b = B.get_access<access::mode::read>(h);
			h.parallel_for(range<2>(N, M), [=](id<2> index) {
				int i = V.get_range()[0];
				int j = V.get_range()[1];
				V[index] = (j != 0) ? 0.0f : Pi[i] + b[i][seq[0]];
				BP[index] = -1;
			});
		});

		q.submit([&](handler& h) {
			auto V = Viterbi.get_access<access::mode::read_write>(h);
			auto BP = Back_pointer.get_access<access::mode::read_write>(h);
			auto a = A.get_access <access::mode::read>(h);
			auto b = B.get_access <access::mode::read>(h);
			
			for (int j = 1; j < T; ++j) {
				for (int i = 0; i < N; ++i) {
					for (int k = 0; k < N; ++k) {
						if (V[k][j] + a[k][i] + b[i][seq[j + 1]] > V[i][j + 1]) {
							V[i][j + 1] = V[k][j] + a[k][i] + b[i][seq[j + 1]];
							BP[i][j] = k;
						}
					}
				}
			}

			vector<int> ViterbiPath;
			for (int i = M - 1; i >= 0; --i) {
				double max = -1.0 * DBL_MAX;
				int num = -1;
				for (int j = 0; j < N; ++j) {
					if (V[i][j] > max) {
						max = V[i][j];
						num = j;
					}
					ViterbiPath.push_back(num);
				}
			}
		});
	} catch (sycl::exception const& e) {
	  cout << "An exception is caught!\n";
	  cout << "Error message:" << e.what();
      terminate();
	}



	/*for (int j = 1; j < T; ++j) {
		for (int i = 0; i < N; ++i) {
			for (int k = 0; k < N; ++k) {
				if (Viterbi[k][j - 1] * A[k][i] > Viterbi[i][j]) {
					Viterbi[i][j] = Viterbi[k][j - 1] * A[k][i];
					Back_pointer[i][j] = k;
				}
			}
			Viterbi[i][j] *= B[i][seq[j]];
		}
	}*/

	return 0;
}
