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
static const int N = 30000;
static const real check_error = 1e-15;
static const real calculation_error = 1e-10;
static const int min_rand = -1000;
static const int max_rand = 1000;
int max_sweeps = 100;
static const std::uint32_t seed = 666;
gpu_selector selector;
std::ofstream outfile;

// Function responsible for generating a float type
// diagonally dominant matrix. Float had to be used
// as using double would result in segmentation faults
// for extreamlly large matrixes. This is also an example
// of using sycl based RNG which had to be used as using
// external (non sycl) functions slows down the execution
// drasticly.
void generate_matrix(std::vector<float> &matrix, std::vector<real> &results) {
  queue q(selector);

  buffer buf_mat(matrix);
  buffer buf_res(results);

  q.submit([&](handler &h) {
    accessor M{buf_mat, h};
    accessor R{buf_res, h};
    h.parallel_for(range<1>(N), [=](id<1> id) {
      int i = id;
      int j = N * i;
      int it = N * i + i;

      real sum = 0;

      oneapi::dpl::minstd_rand engine(seed, i + j);

      oneapi::dpl::uniform_real_distribution<real> distr(min_rand, max_rand);

      for (int j = i * N; j < N * (i + 1); ++j) {
        M[j] = distr(engine);
        M[j] = round(100. * M[j]) / 100.;
        sum += fabs(M[j]);
      }

      oneapi::dpl::uniform_int_distribution<int> distr2(0, 100);
      int gen_neg = distr2(engine);

      if (gen_neg < 50)
        M[i * N + i] = sum + 1;
      else
        M[i * N + i] = -1 * (sum + 1);

      R[i] = distr(engine);
      R[i] = round(100. * R[i]) / 100.;
    });
  });
}
// Function responsible for printing the matrix, called only for N < 10.
void print_matrix(std::vector<float> matrix, std::vector<real> results) {
  for (int i = 0; i < N; ++i) {
    std::cout << '[';
    for (int j = i * N; j < N * (i + 1); ++j) {
      std::cout << matrix[j] << " ";
    }
    std::cout << "][" << results[i] << "]\n";
  }

  for (int i = 0; i < N; ++i) {
    outfile << '[';
    for (int j = i * N; j < N * (i + 1); ++j) {
      outfile << matrix[j] << " ";
    }
    outfile << "][" << results[i] << "]\n";
  }
}
// Function responsible for printing the results.
void print_results(real *data, int N) {
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
bool check_if_equal(real *data, real *old_values) {
  int correct_result = 0;

  for (int i = 0; i < N; ++i) {
    if (fabs(data[i] - old_values[i]) < check_error) correct_result++;
  }

  return correct_result == N;
}

int main(int argc, char *argv[]) {
  for (int i = 0; i < argc; ++i) std::cout << argv[i] << std::endl;
  auto begin_runtime = std::chrono::high_resolution_clock::now();

  outfile.open("report.txt", std::ios_base::out);

  std::vector<float> matrix(N * N);
  std::vector<real> results(N);

  queue q(selector);

  std::cout << "Device : " << q.get_device().get_info<info::device::name>()
            << std::endl;
  outfile << "Device : " << q.get_device().get_info<info::device::name>()
          << std::endl;

  auto begin_matrix = std::chrono::high_resolution_clock::now();

  generate_matrix(matrix, results);

  buffer buf_mat(matrix);
  buffer buf_res(results);

  auto end_matrix = std::chrono::high_resolution_clock::now();
  auto elapsed_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_matrix - begin_matrix);

  std::cout << "\nMatrix generated, time elapsed: "
            << elapsed_matrix.count() * 1e-9 << " seconds.\n";
  outfile << "\nMatrix generated, time elapsed: "
          << elapsed_matrix.count() * 1e-9 << " seconds.\n";

  if (N < 10) print_matrix(matrix, results);

  auto begin_computations = std::chrono::high_resolution_clock::now();

  real *data = malloc_shared<real>(N, q);
  real *old_values = malloc_shared<real>(N, q);

  for (int i = 0; i < N; i++) data[i] = 0;

  bool is_equal = false;
  int sweeps = 0;

  // The main functionality of the Jacobi Solver. Every iteration
  // calculates new values until the difference between the values
  // calculatedthis iteration and the one before is less than the error.
  do {
    for (int i = 0; i < N; ++i) old_values[i] = data[i];
    q.submit([&](handler &h) {
       accessor M{buf_mat, h, read_only};
       accessor R{buf_res, h, read_only};
       h.parallel_for(range<1>(N), [=](id<1> id) {
         int i = id;
         int j = N * i;
         int it = N * i + i;

         data[i] = R[i];
         for (int z = 0; z < N; ++z) {
           if (z != i)
             data[i] = data[i] - (old_values[z] * static_cast<real>(M[j]));
           j = j + 1;
         }
         data[i] = data[i] / static_cast<real>(M[it]);
       });
     }).wait();

    ++sweeps;
    is_equal = check_if_equal(data, old_values);
  } while (!is_equal && sweeps < max_sweeps);

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

  std::vector<real> new_results(N, 0);

  // Calculating a new set of results from the calculated values.
  for (int i = 0; i < N * N; ++i) {
    new_results[i / N] += data[i % N] * static_cast<real>(matrix[i]);
  }

  bool *all_eq = malloc_shared<bool>(1, q);
  all_eq[0] = true;

  // Comparing the newly calculated results with the ones that were
  // given. If the difference is less than the error rate for each of
  // the elements, then all values have been calculated correctly.
  {
    buffer buf_new_res(new_results);

    q.submit([&](handler &h) {
      accessor R{buf_res, h, read_only};
      accessor NR{buf_new_res, h, read_only};
      h.parallel_for(range<1>(N), [=](id<1> id) {
        real diff = fabs(NR[id] - R[id]);
        if (diff > calculation_error) all_eq[0] = false;
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

  print_results(data, N);
  free(data, q);
  free(old_values, q);

  return 0;
}
