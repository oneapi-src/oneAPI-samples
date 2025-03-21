#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#define SIZE 10000000
#define NUM_BINS 2048
#define REAL float

void initialize(REAL *input, int size, int num_bins) {
  for (int i = 0; i < size; i++) {
    input[i] = rand() % num_bins;
  }
}

void validate(int *result_ref, int *result, int num_bins) {
  for (int i = 0; i < num_bins; i++)
    assert(result_ref[i] == result[i]);
}

int main(int argc, char **argv) {

  int size = SIZE;
  int num_bins = NUM_BINS;
  if (argc > 1)
    size = atoi(argv[1]);

  REAL *input = reinterpret_cast<REAL *>(malloc(size * sizeof(REAL)));
  int *result = reinterpret_cast<int *>(calloc(num_bins, sizeof(int)));

  initialize(input, size, num_bins);
  double total_time;

  // collect result on host for validation
  int *result_ref = (int *)calloc(num_bins, sizeof(int));
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    int type = input[i];
#pragma omp atomic update
    result_ref[type]++;
  }

  total_time = omp_get_wtime();
  // critical begin
#pragma omp target teams distribute parallel for map(to : input[0 : size])     \
    map(tofrom : result[0 : num_bins]) num_teams(1)
  for (int i = 0; i < size; i++) {
    int type = input[i];
#pragma omp critical
    result[type]++;
  }
  // critical end
  total_time = omp_get_wtime() - total_time;
  printf("Critical: %g ms\n", total_time * 1000);
  validate(result_ref, result, num_bins);
  memset(result, 0, sizeof(int) * num_bins);

  total_time = omp_get_wtime();
  // atomic relaxed begin
#pragma omp target teams distribute parallel for map(to : input[0 : size])     \
    map(tofrom : result[0 : num_bins])
  for (int i = 0; i < size; i++) {
    int type = input[i];
#pragma omp atomic update
    result[type]++;
  }
  // atomic relaxed end
  total_time = omp_get_wtime() - total_time;
  printf("Atomic relaxed: %g ms\n", total_time * 1000);
  validate(result_ref, result, num_bins);
  memset(result, 0, sizeof(int) * num_bins);

  total_time = omp_get_wtime();
  // atomic seq_cst begin
#pragma omp target teams distribute parallel for map(to : input[0 : size])     \
    map(tofrom : result[0 : num_bins])
  for (int i = 0; i < size; i++) {
    int type = input[i];
#pragma omp atomic update seq_cst
    result[type]++;
  }
  // atomic seq_cst end
  total_time = omp_get_wtime() - total_time;
  printf("Atomic seq_cst: %g ms\n", total_time * 1000);
  validate(result_ref, result, num_bins);
  memset(result, 0, sizeof(int) * num_bins);

  total_time = omp_get_wtime();
  // atomic relaxed using SLM begin
#pragma omp target teams map(to : input[0 : size])                             \
    map(tofrom : result[0 : num_bins])
  {
    // create a local histogram using SLM in the team
    int local_histogram[NUM_BINS] = {0};
    int num_local_histogram = omp_get_num_teams();
    int team_id = omp_get_team_num();
    int chunk_size = size / num_local_histogram;
    int leftover = size % num_local_histogram;
    int local_lb = team_id * chunk_size;
    int local_ub = (team_id + 1) * chunk_size;
    //  Add the leftover to last chunk.
    //  e.g. 18 iterations and 4 teams -> 4, 4, 4, 6 = 4(last chunk) +
    //  2(leftover)
    if (local_ub + chunk_size > size)
      local_ub += leftover;
    if (local_ub <= size) {
#pragma omp parallel for shared(local_histogram)
      for (int i = local_lb; i < local_ub; i++) {
        int type = input[i];
#pragma omp atomic update
        local_histogram[type]++;
      }

      // Combine local histograms
#pragma omp parallel for
      for (int i = 0; i < num_bins; i++) {
#pragma omp atomic update
        result[i] += local_histogram[i];
      }
    }
  }
  // atomic relaxed using SLM end
  total_time = omp_get_wtime() - total_time;
  printf("Atomic relaxed with SLM: %g ms\n", total_time * 1000);
  validate(result_ref, result, num_bins);
  memset(result, 0, sizeof(int) * num_bins);

  total_time = omp_get_wtime();
  // atomic seq_cst using SLM begin
#pragma omp target map(to : input[0 : size]) map(tofrom : result[0 : num_bins])
#pragma omp teams
  {
    // create a local histogram using SLM in the team
    int local_histogram[NUM_BINS] = {0};
    int num_local_histogram = omp_get_num_teams();
    int team_id = omp_get_team_num();
    int chunk_size = size / num_local_histogram;
    int leftover = size % num_local_histogram;
    int local_lb = team_id * chunk_size;
    int local_ub = (team_id + 1) * chunk_size;
    //  Add the leftover to last chunk.
    //  e.g. 18 iterations and 4 teams -> 4, 4, 4, 6 = 4(last chunk) +
    //  2(leftover)
    if (local_ub + chunk_size > size)
      local_ub += leftover;
    if (local_ub <= size) {
#pragma omp parallel for shared(local_histogram)
      for (int i = local_lb; i < local_ub; i++) {
        int type = input[i];
#pragma omp atomic update seq_cst
        local_histogram[type]++;
      }

      // Combine local histograms
#pragma omp parallel for
      for (int i = 0; i < num_bins; i++) {
#pragma omp atomic update seq_cst
        result[i] += local_histogram[i];
      }
    }
  }
  // atomic seq_cst using SLM end
  total_time = omp_get_wtime() - total_time;
  printf("Atomic seq_cst with SLM: %g ms\n", total_time * 1000);
  validate(result_ref, result, num_bins);
  memset(result, 0, sizeof(int) * num_bins);

  free(input);
  free(result);
  free(result_ref);

  return 0;
}
