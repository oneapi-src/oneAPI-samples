/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Distributed Jacobian computation sample using OpenMP GPU offload and MPI-3 one-sided.
 */
#include "mpi.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int Nx = 16384; /* Grid size */
const int Ny = Nx;
const int Niter = 100; /* Nuber of algorithm iterations */
const int NormIteration = 10; /* Recaluculate norm after given number of iterations. 0 to disable norm calculation */
const int PrintTime = 1; /* Output overall time of compute/communication part */

struct subarray {
    int rank, comm_size;        /* MPI rank and communicator size */
    int x_size, y_size;         /* Subarray size excluding border rows and columns */
    MPI_Aint l_nbh_offt;        /* Offset predecessor data to update */
};

#define ROW_SIZE(S) ((S).x_size + 2)
#define XY_2_IDX(X,Y,S) (((Y)+1)*ROW_SIZE(S)+((X)+1))

/* Subroutine to create and initialize initial state of input subarrays */
void InitDeviceArrays(double **A_dev_1, double **A_dev_2, struct subarray *sub)
{
    size_t total_size = (sub->x_size + 2) * (sub->y_size + 2);

    int device_id = omp_get_default_device();
    int host_id = omp_get_initial_device();
    double *A = (double*) malloc(total_size * sizeof(double));

    *A_dev_1 = (double*) omp_target_alloc_device(total_size * sizeof(double), device_id);
    *A_dev_2 = (double*) omp_target_alloc_device(total_size * sizeof(double), device_id);

    for (int i = 0; i < (sub->y_size + 2); i++)
        for (int j = 0; j < (sub->x_size + 2); j++)
            A[i * (sub->x_size + 2) + j] = 0.0;

    if (sub->rank == 0) /* set top boundary */
        for (int i = 1; i <= sub->x_size; i++)
            A[i] = 1.0; /* set bottom boundary */
    if (sub->rank == (sub->comm_size - 1))
        for (int i = 1; i <= sub->x_size; i++)
            A[(sub->x_size + 2) * (sub->y_size + 1) + i] = 10.0;

    for (int i = 1; i <= sub->y_size; i++) {
        int row_offt = i * (sub->x_size + 2);
        A[row_offt] = 1.0;      /* set left boundary */
        A[row_offt + sub->x_size + 1] = 1.0;    /* set right boundary */
    }
    /* Move input arrays to device */
    omp_target_memcpy(*A_dev_1, A, sizeof(double) * total_size, 0, 0, device_id, host_id);
    omp_target_memcpy(*A_dev_2, A, sizeof(double) * total_size, 0, 0, device_id, host_id);
    free(A);
}

/* Setup subarray size and layout processed by current rank */
void GetMySubarray(struct subarray *sub)
{
    MPI_Comm_size(MPI_COMM_WORLD, &sub->comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &sub->rank);
    sub->y_size = Ny / sub->comm_size;
    sub->x_size = Nx;
    sub->l_nbh_offt = (sub->x_size + 2) * (sub->y_size + 1) + 1;

    int tail = sub->y_size % sub->comm_size;
    if (tail != 0) {
        if (sub->rank < tail)
            sub->y_size++;
        if ((sub->rank > 0) && ((sub->rank - 1) < tail))
            sub->l_nbh_offt += (sub->x_size + 2);
    }
}

int main(int argc, char *argv[])
{
    double t_start;
    struct subarray my_subarray = { };
    double *A_device1 = NULL;
    double *A_device2 = NULL;
    MPI_Win win[2] = { MPI_WIN_NULL, MPI_WIN_NULL };

    /* Initialization of runtime and initial state of data */
    MPI_Init(&argc, &argv);
    GetMySubarray(&my_subarray);
    InitDeviceArrays(&A_device1, &A_device2, &my_subarray);

    /* Create RMA window using device memory */
    MPI_Win_create(A_device1,
                   sizeof(double) * (my_subarray.x_size + 2) * (my_subarray.y_size + 2),
                   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win[0]);
    MPI_Win_create(A_device2,
                   sizeof(double) * (my_subarray.x_size + 2) * (my_subarray.y_size + 2),
                   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win[1]);

    /* Start RMA exposure epoch */
    MPI_Win_fence(0, win[0]);
    MPI_Win_fence(0, win[1]);

    if (PrintTime) {
        t_start = MPI_Wtime();
    }

#pragma omp target data map(to: Niter, my_subarray, win[0:2], NormIteration) use_device_ptr(A_device1, A_device2)
    {
        for (int i = 0; i < Niter; ++i) {
            MPI_Win cwin = win[(i + 1) % 2];
            double *a = (i % 2) ? A_device1 : A_device2;
            double *a_out = ((1 + i) % 2) ? A_device1 : A_device2;

            /* Offload compute loop to the device */
#pragma omp target teams distribute parallel for is_device_ptr(a, a_out)
            /* Calculate values on borders to initiate communications early */
            for (int _column = 0; _column < my_subarray.x_size; ++_column) {
                int idx = XY_2_IDX(_column, 0, my_subarray);
                a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] + a[idx - ROW_SIZE(my_subarray)] +
                                     a[idx + ROW_SIZE(my_subarray)]);

                idx = XY_2_IDX(_column, my_subarray.y_size - 1, my_subarray);
                a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] + a[idx - ROW_SIZE(my_subarray)] +
                                     a[idx + ROW_SIZE(my_subarray)]);
            }

            /* Perform halo-exchange with neighbours */
            if (my_subarray.rank != 0) {
                int idx = XY_2_IDX(0, 0, my_subarray);
                MPI_Put(&a_out[idx], my_subarray.x_size, MPI_DOUBLE,
                        my_subarray.rank - 1, my_subarray.l_nbh_offt,
                        my_subarray.x_size, MPI_DOUBLE, cwin);
            }

            if (my_subarray.rank != (my_subarray.comm_size - 1)) {
                int idx = XY_2_IDX(0, my_subarray.y_size - 1, my_subarray);
                MPI_Put(&a_out[idx], my_subarray.x_size, MPI_DOUBLE,
                        my_subarray.rank + 1, 1,
                        my_subarray.x_size, MPI_DOUBLE, cwin);
                }

            /* Offload compute loop to the device */
#pragma omp target teams distribute parallel for is_device_ptr(a, a_out) collapse(2)
            /* Recalculate internal points in parallel with communication */
            for (int row = 1; row < my_subarray.y_size - 1; ++row) {
                for (int column = 0; column < my_subarray.x_size; ++column) {
                    int idx = XY_2_IDX(column, row, my_subarray);
                    a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] + a[idx - ROW_SIZE(my_subarray)]
                                         + a[idx + ROW_SIZE(my_subarray)]);
                }
            }

            if ((NormIteration > 0) && ((NormIteration - 1) == i % NormIteration)) {
                    double result_norm = 0.0;
                    double norm = 0.0;

                    /* Offload compute loop to the device */
#pragma omp target teams distribute parallel for is_device_ptr(a, a_out) reduction(+:norm) collapse(2)
                    /* Calculate and report norm value after given number of iterations */
                    for (int row = 0; row < my_subarray.y_size; ++row) {
                        for (int column = 0; column < my_subarray.x_size; ++column) {
                            int idx = XY_2_IDX(column, row, my_subarray);
                            double diff = a_out[idx] - a[idx];
                            norm += diff*diff;
                        }
                    }

                    MPI_Reduce(&norm, &result_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    if (my_subarray.rank == 0) {
                        printf("NORM value on iteration %d: %f\n", i+1, sqrt(result_norm));
                    }
            }
            /* Ensure all communications are complete before next iteration */
            MPI_Win_fence(0, cwin);
        }
    }

    if (PrintTime) {
        double avg_time;
        double rank_time;
        rank_time = MPI_Wtime() - t_start;

        MPI_Reduce(&rank_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (my_subarray.rank == 0) {
            avg_time = avg_time/my_subarray.comm_size;
            printf("Average solver time: %f(sec)\n", avg_time);
        }
    }

    if (my_subarray.rank == 0) {
        printf("SUCCESS\n");
    }
    MPI_Win_free(&win[1]);
    MPI_Win_free(&win[0]);
    MPI_Finalize();

    omp_target_free((void*)A_device1, omp_get_default_device());
    omp_target_free((void*)A_device2, omp_get_default_device());

    return 0;
}
