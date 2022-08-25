
//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <bits/stdc++.h>

#include <CL/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <oneapi/dpl/random>
#include <vector>

using namespace sycl;

typedef double real;

// Program variables, feel free to change anything .
static const int kN = 30000;
static const real kCheckError = 1e-15;
static const real kCalculationError = 1e-10;
static const int kMinRand = -1000;
static const int kMaxRand = 1000;
static const int kMaxSweeps = 100;
static const std::uint32_t kSeed = 666;
gpu_selector selector;
std::ofstream outfile;

// Function responsible for generating a float type
// diagonally dominant matrix. Float had to be used
// as using double would result in segmentation faults
// for extreamlly large matrixes. This is also an example
// of using sycl based RNG which had to be used as using
// external (non sycl) functions slows down the execution
// drasticly.
void GenerateMatrix(std::vector<float> &input_matrix,
                    std::vector<real> &input_results) {
  std::vector<queue> q;
  for (const auto &p : platform::get_platforms()) {
    if (p.get_info<info::platform::name>().find("Level-Zero") !=
        std::string::npos) {
      for (const auto &d : p.get_devices()) {
        if (d.is_gpu() && d.get_info<info::device::name>().find("Intel") !=
                              std::string::npos) {
          q.push_back(queue(d));
          outfile << "--Found GPU: " << d.get_info<info::device::name>()
                  << "\n";
        }
      }
    }
  }

  buffer bufin_mat(input_matrix);
  buffer bufin_res(input_results);

  q[0].submit([&](handler &h) {
    accessor M{bufin_mat, h};
    accessor R{bufin_res, h};
    h.parallel_for(range<1>(kN / 2), [=](id<1> id) {
      int i = id;
      int j = kN * i;
      int it = kN * i + i;

      real sum = 0;

      oneapi::dpl::minstd_rand engine(kSeed, i + j);

      oneapi::dpl::uniform_real_distribution<real> distr(kMinRand, kMaxRand);

      for (int j = i * kN; j < kN * (i + 1); ++j) {
        M[j] = distr(engine);
        M[j] = round(100. * M[j]) / 100.;
        sum += fabs(M[j]);
      }

      oneapi::dpl::uniform_int_distribution<int> distr2(0, 100);
      int gen_neg = distr2(engine);

      if (gen_neg < 50)
        M[i * kN + i] = sum + 1;
      else
        M[i * kN + i] = -1 * (sum + 1);

      R[i] = distr(engine);
      R[i] = round(100. * R[i]) / 100.;
    });
  });
  q[1].submit([&](handler &h) {
    accessor M{bufin_mat, h};
    accessor R{bufin_res, h};
    h.parallel_for(range<1>(kN / 2), [=](id<1> id) {
      int i = kN / 2 + id;
      int j = kN * i;
      int it = kN * i + i;

      real sum = 0;

      oneapi::dpl::minstd_rand engine(kSeed, i + j);

      oneapi::dpl::uniform_real_distribution<real> distr(kMinRand, kMaxRand);

      for (int j = i * kN; j < kN * (i + 1); ++j) {
        M[j] = distr(engine);
        M[j] = round(100. * M[j]) / 100.;
        sum += fabs(M[j]);
      }

      oneapi::dpl::uniform_int_distribution<int> distr2(0, 100);
      int gen_neg = distr2(engine);

      if (gen_neg < 50)
        M[i * kN + i] = sum + 1;
      else
        M[i * kN + i] = -1 * (sum + 1);

      R[i] = distr(engine);
      R[i] = round(100. * R[i]) / 100.;
    });
  });
}
// Function responsible for printing the matrix, called only for N < 10.
void PrintMatrix(std::vector<float> input_matrix,
                 std::vector<real> input_results) {
  for (int i = 0; i < kN; ++i) {
    std::cout << '[';
    for (int j = i * kN; j < kN * (i + 1); ++j) {
      std::cout << input_matrix[j] << " ";
    }
    std::cout << "][" << input_results[i] << "]\n";
  }

  for (int i = 0; i < kN; ++i) {
    outfile << '[';
    for (int j = i * kN; j < kN * (i + 1); ++j) {
      outfile << input_matrix[j] << " ";
    }
    outfile << "][" << input_results[i] << "]\n";
  }
}
// Function responsible for printing the results.
void PrintResults(std::vector<real> data, int N) {
  outfile << std::fixed;
  outfile << std::setprecision(11);
  for (int i = 0; i < N; ++i)
    outfile << "X" << i + 1 << " equals: " << data[i] << std::endl;
}
// Function responsible for checking if the algorithm has finished.
// For each of the newly calculated results the difference is checked
// betwenn it and the corresponding result from the previous iteration.
// If the difference between them is less than the error variable the
// number is incremented by one, if all the results are correct the function
// returns a bool value that is true and the main function can stop.
bool CheckIfEqual(std::vector<real> data, std::vector<real> old_output_data) {
  int correct_result = 0;

  for (int i = 0; i < kN; ++i) {
    if (fabs(data[i] - old_output_data[i]) < kCheckError) correct_result++;
  }

  return correct_result == kN;
}

int main(int argc, char *argv[]) {
  auto begin_runtime = std::chrono::high_resolution_clock::now();

  outfile.open("report.txt", std::ios_base::out);

  std::vector<float> input_matrix(kN * kN);
  std::vector<real> input_results(kN);

  std::vector<queue> q;
  for (const auto &p : platform::get_platforms()) {
    if (p.get_info<info::platform::name>().find("Level-Zero") !=
        std::string::npos) {
      for (const auto &d : p.get_devices()) {
        if (d.is_gpu() && d.get_info<info::device::name>().find("Intel") !=
                              std::string::npos) {
          q.push_back(queue(d));
          std::cout << "--Found GPU: " << d.get_info<info::device::name>()
                    << "\n";
        }
      }
    }
  }

  if (q.size() < 2) {
    std::cout << "Two GPUs are needed to run this code. Aborting\n";
    return 0;
  }

  auto begin_matrix = std::chrono::high_resolution_clock::now();

  GenerateMatrix(input_matrix, input_results);

  buffer bufin_mat(input_matrix);
  buffer bufin_res(input_results);

  auto end_matrix = std::chrono::high_resolution_clock::now();
  auto elapsed_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_matrix - begin_matrix);

  std::cout << "\nMatrix generated, time elapsed: "
            << elapsed_matrix.count() * 1e-9 << " seconds.\n";
  outfile << "\nMatrix generated, time elapsed: "
          << elapsed_matrix.count() * 1e-9 << " seconds.\n";

  if (kN < 10) PrintMatrix(input_matrix, input_results);

  auto begin_computations = std::chrono::high_resolution_clock::now();

  std::vector<real> output_data(kN, 0);
  std::vector<real> old_output_data(kN, 0);

  for (int i = 0; i < kN; i++) output_data[i] = 0;

  bool is_equal = false;
  int sweeps = 0;

  // The main functionality of the Jacobi Solver. Every iteration
  // calculates new values until the difference between the values
  // calculatedthis iteration and the one before is less than the error.
  {
    do {
      buffer bufout_data(output_data);
      buffer bufold_out_data(old_output_data);
      for (int i = 0; i < kN; ++i) old_output_data[i] = output_data[i];
      q[0].submit([&](handler &h) {
            accessor D{bufout_data, h};
            accessor OV{bufold_out_data, h};
            accessor M{bufin_mat, h, read_only};
            accessor R{bufin_res, h, read_only};
            h.parallel_for(range<1>(kN / 2), [=](id<1> id) {
              int i = id;
              int j = kN * i;
              int it = kN * i + i;

              D[i] = R[i];
              for (int z = 0; z < kN; ++z) {
                if (z != i) D[i] = D[i] - (OV[z] * static_cast<real>(M[j]));
                j = j + 1;
              }
              D[i] = D[i] / static_cast<real>(M[it]);
            });
          })
          .wait();

      q[1].submit([&](handler &h) {
            accessor D{bufout_data, h};
            accessor OV{bufold_out_data, h};
            accessor M{bufin_mat, h, read_only};
            accessor R{bufin_res, h, read_only};
            h.parallel_for(range<1>(kN / 2), [=](id<1> id) {
              int i = kN / 2 + id;
              int j = kN * i;
              int it = kN * i + i;

              D[i] = R[i];
              for (int z = 0; z < kN; ++z) {
                if (z != i) D[i] = D[i] - (OV[z] * static_cast<real>(M[j]));
                j = j + 1;
              }
              D[i] = D[i] / static_cast<real>(M[it]);
            });
          })
          .wait();

      bufout_data.get_access<access::mode::read>();
      bufold_out_data.get_access<access::mode::read>();

      ++sweeps;
      is_equal = CheckIfEqual(output_data, old_output_data);
    } while (!is_equal && sweeps < kMaxSweeps);
  }
  auto end_computations = std::chrono::high_resolution_clock::now();
  auto elapsed_computations =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_computations -
                                                           begin_computations);

  std::cout << "\nComputations complete, time elapsed: "
            << elapsed_computations.count() * 1e-9 << " seconds.\n";
  std::cout << "Total number of sweeps: " << sweeps << "\nChecking results\n";
  outfile << "\nComputations complete, time elapsed: "
          << elapsed_computations.count() * 1e-9 << " seconds.\n";
  outfile << "Total number of sweeps: " << sweeps << "\nChecking results\n";

  auto begin_check = std::chrono::high_resolution_clock::now();

  std::vector<real> output_results(kN, 0);

  // Calculating a new set of results from the calculated values.
  for (int i = 0; i < kN * kN; ++i) {
    output_results[i / kN] +=
        output_data[i % kN] * static_cast<real>(input_matrix[i]);
  }

  bool *all_eq = malloc_shared<bool>(1, q[0]);
  all_eq[0] = true;

  // Comparing the newly calculated results with the ones that were
  // given. If the difference is less than the error rate for each of
  // the elements, then all values have been calculated correctly.
  {
    buffer bufout_res(output_results);

    q[0].submit([&](handler &h) {
      accessor R{bufin_res, h, read_only};
      accessor NR{bufout_res, h, read_only};
      h.parallel_for(range<1>(kN), [=](id<1> id) {
        real diff = fabs(NR[id] - R[id]);
        if (diff > kCalculationError) all_eq[0] = false;
      });
    });
  }

  if (all_eq[0]) {
    std::cout << "All values are correct.\n";
    outfile << "All values are correct.\n";
  } else {
    std::cout << "There have been some errors. The values are not correct.\n";
    outfile << "There have been some errors. The values are not correct.\n";
  }

  auto end_check = std::chrono::high_resolution_clock::now();
  auto elapsed_check = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_check - begin_check);

  std::cout << "\nCheck complete, time elapsed: "
            << elapsed_check.count() * 1e-9 << " seconds.\n";
  outfile << "\nCheck complete, time elapsed: " << elapsed_check.count() * 1e-9
          << " seconds.\n";

  auto end_runtime = std::chrono::high_resolution_clock::now();
  auto elapsed_runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_runtime - begin_runtime);

  std::cout << "Total runtime is " << elapsed_runtime.count() * 1e-9
            << " seconds.\n";
  outfile << "Total runtime is " << elapsed_runtime.count() * 1e-9
          << " seconds.\n";

  PrintResults(output_data, kN);

  return 0;
}
