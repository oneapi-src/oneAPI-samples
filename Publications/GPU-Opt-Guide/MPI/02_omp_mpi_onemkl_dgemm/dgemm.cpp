#include "mkl.h"
#include "mkl_omp_offload.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <mpi.h>
#include <omp.h>
#define FLOAT double
#define MPI_FLOAT_T MPI_DOUBLE
#define MKL_INT_T MKL_INT
#define index(i, j, ld) (((j) * (ld)) + (i))
#define RAND() ((FLOAT)rand() / (FLOAT)RAND_MAX * 2.0 - 1.0)
#define LD_ALIGN 256
#define LD_BIAS 8
#define HPL_PTR(ptr_, al_) ((((size_t)(ptr_) + (al_) - 1) / (al_)) * (al_))
static inline MKL_INT_T getld(MKL_INT_T x) {
  MKL_INT_T ld;
  ld = HPL_PTR(x, LD_ALIGN);
  if (ld - LD_BIAS >= x)
    ld -= LD_BIAS;
  else
    ld += LD_BIAS;
  return ld;
}
int main(int argc, char **argv) {
  if ((argc < 4) || (argc > 4 && argc < 8)) {
    printf("Performs a DGEMM test C = alpha*A*B + beta*C\n");
    printf("A matrix is MxK and B matrix is KxN\n");
    printf("All matrices are stored in column-major format\n");
    printf("Run as ./dgemm <M> <K> <N> [<alpha> <beta> <iterations>]\n");
    printf("Required inputs are:\n");
    printf("      M: number of rows of matrix A\n");
    printf("      K: number of cols of matrix A\n");
    printf("      N: number of cols of matrix B\n");
    printf("Optional inputs are (all must be provided if providing any):\n");
    printf("      alpha: scalar multiplier (default: 1.0)\n");
    printf("      beta:  scalar multiplier (default: 0.0)\n");
    printf("      iterations: number of blocking DGEMM calls to perform "
           "(default: 10)\n");
    return EXIT_FAILURE;
  }
  MKL_INT_T HA = (MKL_INT_T)(atoi(argv[1]));
  MKL_INT_T WA = (MKL_INT_T)(atoi(argv[2]));
  MKL_INT_T WB = (MKL_INT_T)(atoi(argv[3]));
  FLOAT alpha, beta;
  int niter;
  if (argc > 4) {
    sscanf(argv[4], "%lf", &alpha);
    sscanf(argv[5], "%lf", &beta);
    niter = atoi(argv[6]);
  } else {
    alpha = 1.0;
    beta = 0.0;
    niter = 10;
  }
  MKL_INT_T HB = WA;
  MKL_INT_T WC = WB;
  MKL_INT_T HC = HA;
  MKL_INT_T ldA = getld(HA);
  MKL_INT_T ldB = getld(HB);
  MKL_INT_T ldC = getld(HC);
  double tot_t = 0.0, best_t = std::numeric_limits<double>::max();
  FLOAT *A = new FLOAT[ldA * WA];
  FLOAT *B, *C, *local_B, *local_C;
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    B = new FLOAT[ldB * WB];
    C = new FLOAT[ldC * WC];
    srand(2864);
    for (int j = 0; j < WA; j++)
      for (int i = 0; i < HA; i++)
        A[index(i, j, ldA)] = RAND();
    for (int j = 0; j < WB; j++)
      for (int i = 0; i < HB; i++)
        B[index(i, j, ldB)] = RAND();
    if (beta != 0.0) {
      for (int j = 0; j < WC; j++)
        for (int i = 0; i < HC; i++)
          C[index(i, j, ldC)] = RAND();
    } else {
      for (int j = 0; j < WC; j++)
        for (int i = 0; i < HC; i++)
          C[index(i, j, ldC)] = 0.0;
    }
  }
  size_t sizea = (size_t)ldA * WA;
  size_t local_sizeb, local_sizec;
  int *displacements_b = new int[mpi_size];
  int *send_counts_b = new int[mpi_size];
  int *displacements_c = new int[mpi_size];
  int *send_counts_c = new int[mpi_size];
  int local_WB = WB / mpi_size;
  send_counts_b[0] = ldB * (local_WB + WB % mpi_size);
  send_counts_c[0] = ldC * (local_WB + WB % mpi_size);
  displacements_b[0] = 0;
  displacements_c[0] = 0;
  for (int i = 1; i < mpi_size; i++) {
    send_counts_b[i] = ldB * local_WB;
    send_counts_c[i] = ldC * local_WB;
    displacements_b[i] = displacements_b[i - 1] + send_counts_b[i - 1];
    displacements_c[i] = displacements_b[i - 1] + send_counts_c[i - 1];
  }
  if (mpi_rank == 0) {
    local_WB += WB % mpi_size;
  }
  local_sizeb = ldB * local_WB;
  local_sizec = ldC * local_WB;
  local_B = new FLOAT[local_sizeb];
  local_C = new FLOAT[local_sizec];
  MPI_Bcast(A, sizea, MPI_FLOAT_T, 0, MPI_COMM_WORLD);
  MPI_Scatterv(B, send_counts_b, displacements_b, MPI_FLOAT_T, local_B,
               local_sizeb, MPI_FLOAT_T, 0, MPI_COMM_WORLD);
  MPI_Scatterv(C, send_counts_c, displacements_c, MPI_FLOAT_T, local_C,
               local_sizec, MPI_FLOAT_T, 0, MPI_COMM_WORLD);
#if defined(OMP_AFFINITIZATION)
#if OMP_AFFINITIZATION == 1
  int ndev = omp_get_num_devices();
  int dnum = mpi_rank % ndev;
  omp_set_default_device(dnum);
#endif
#endif
#pragma omp target data map(to : A[0 : sizea], local_B[0 : local_sizeb])       \
    map(tofrom : local_C[0 : local_sizec])
  {
#pragma omp dispatch
    dgemm("N", "N", &HA, &local_WB, &WA, &alpha, A, &ldA, local_B, &ldB, &beta,
          local_C, &ldC);
    for (int i = 0; i < niter; i++) {
      auto start_t = std::chrono::high_resolution_clock::now();
#pragma omp dispatch
      dgemm("N", "N", &HA, &local_WB, &WA, &alpha, A, &ldA, local_B, &ldB,
            &beta, local_C, &ldC);
      MPI_Barrier(MPI_COMM_WORLD);
      auto end_t = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end_t - start_t;
      tot_t += diff.count();
      best_t = std::min(best_t, diff.count());
    }
  }
  MPI_Gatherv(local_C, local_sizec, MPI_FLOAT_T, C, send_counts_c,
              displacements_c, MPI_FLOAT_T, 0, MPI_COMM_WORLD);
  delete[] local_B;
  delete[] local_C;
  delete[] displacements_b;
  delete[] displacements_c;
  delete[] send_counts_b;
  delete[] send_counts_c;
  MPI_Allreduce(MPI_IN_PLACE, &tot_t, 1, MPI_FLOAT_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &best_t, 1, MPI_FLOAT_T, MPI_MAX, MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    double tflop_count = (double)2.0 * HA * WB * WA;
    if (beta != 0.0)
      tflop_count += (double)HA * WB;
    tflop_count *= 1.E-12;
    printf("Total runtime for %d iterations: %f seconds.\n", niter, tot_t);
    printf("Mean TFLOP/s: %f\n", (double)niter * tflop_count / tot_t);
    printf("Best TFLOP/s: %f\n", (double)tflop_count / best_t);
    delete[] B;
    delete[] C;
  }
  delete[] A;
  MPI_Finalize();
  return EXIT_SUCCESS;
}
