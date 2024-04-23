/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */
/* Description:
 * Sending GPU buffer from <ACTIVE_RANK> to the host buffer of another rank.
 *
 * How to run:
 * mpiexec -n 2 -genv I_MPI_OFFLOAD=1 -genv LIBOMPTARGET_PLUGIN=level0 ./mpi_send_gpu_buf_omp
*/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#define ACTIVE_RANK 1

void VerifyResult(int *result, int *values, unsigned num_values, int rank)
{
    /* Validation */
    printf("[%d] result: ", rank);
    for (unsigned i = 0; i < num_values; ++i) {
        printf("%d ", result[i]);
        /* Signal an error if the result does not match the expected values */
        if (result[i] != values[i] * (ACTIVE_RANK + 1)) {
            printf("\n"); fflush(stdout);
            fprintf(stderr, "[%d] VALIDATION ERROR (expected %d)\n", rank, values[i] * (ACTIVE_RANK + 1));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    printf("\n");
}

int main(int argc, char **argv) {

    int nranks, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "run mpiexec with -n 2\n");
        MPI_Finalize();
        return 1;
    }

    const unsigned num_values = 10;
    int *values = (int *) malloc (num_values * sizeof(int));
    if (values == NULL) {
        fprintf(stderr, "[%d] could not allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    values[0] = 0;
    values[1] = 1;
    for (unsigned i = 2; i < num_values; i++) {
        values[i] = values[i - 2] + values[i - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == ACTIVE_RANK) {
        /* Copy `rank` and `values` from host to device */
        #pragma omp target data map(to: rank, values[0:num_values]) use_device_ptr(values)
        {
            /* Compute something on device */
            #pragma omp target parallel for is_device_ptr(values)
            for (unsigned i = 0; i < num_values; ++i) {
                values[i] *= (rank + 1);
            }

            /* Send device buffer to another rank (host recieve buffer) */
            printf("[%d] Sending GPU buffer %p to rank %d\n", rank, values, 1 - rank); fflush(stdout);
            MPI_Send(values, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD);

            /* Send device buffer to another rank (device recieve buffer) */
            printf("[%d] Sending GPU buffer %p to rank %d\n", rank, values, 1 - rank); fflush(stdout);
            MPI_Send(values, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD);
        }

        /* Send host buffer to another rank (device recieve buffer) */
        printf("[%d] Sending host buffer %p to rank %d\n", rank, values, 1 - rank); fflush(stdout);
        MPI_Send(values, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD);

    } else {
        int *result = (int *) malloc (num_values * sizeof(int));
        if (result == NULL) {
            fprintf(stderr, "[%d] could not allocate memory\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Receive values from device buffer and store it on the host buffer */
        printf("[%d] Receiving data from rank %d to host buffer\n", rank, 1 - rank); fflush(stdout);
        MPI_Recv(result, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        VerifyResult(result, values, num_values, rank);

        /* Copy `rank` and `values` from host to device and backward */
        #pragma omp target data map(to: rank, result[0:num_values]) map(from: result[0:num_values]) use_device_ptr(result)
        {
            /* Reset recieve buffer */
            #pragma omp target parallel for is_device_ptr(result)
            for (unsigned i = 0; i < num_values; ++i) {
                result[i] = 0;
            }

            /* Receive values from device buffer and store it in the device buffer */
            printf("[%d] Receiving data from rank %d to GPU buffer\n", rank, 1 - rank); fflush(stdout);
            MPI_Recv(result, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        VerifyResult(result, values, num_values, rank);

        /* Copy `rank` and `values` from host to device and backward */
        #pragma omp target data map(to: rank, result[0:num_values]) map(from: result[0:num_values]) use_device_ptr(result)
        {
            /* Reset recieve buffer */
            #pragma omp target parallel for is_device_ptr(result)
            for (unsigned i = 0; i < num_values; ++i) {
                result[i] = 0;
            }

            /* Receive values from host buffer and store it in the device buffer */
            printf("[%d] Receiving data from rank %d to GPU buffer\n", rank, 1 - rank); fflush(stdout);
            MPI_Recv(result, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        VerifyResult(result, values, num_values, rank);


        printf("[%d] SUCCESS\n", rank);
        free(result);
    }

    free(values);
    MPI_Finalize();
    return 0;
}
