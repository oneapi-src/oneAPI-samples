/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */
/* Description:
 * Sending GPU buffer from <ACTIVE_RANK> to the host buffer of another rank.
 *
 * How to run:
 * mpiexec -n 2 -genv I_MPI_OFFLOAD=1 -genv LIBOMPTARGET_PLUGIN=level0 ./mpi_send_gpu_buf_sycl
*/

#include <mpi.h>
#include <sycl.hpp>
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
    const unsigned num_values = 10;

    sycl::queue q(sycl::gpu_selector_v);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nranks != 2) {
        if (rank == 0) fprintf(stderr, "run mpiexec with -n 2\n");
        MPI_Finalize();
        return 1;
    }

    int values[num_values];
    values[0] = 0;
    values[1] = 1;
    for (unsigned i = 2; i < num_values; i++) {
        values[i] = values[i - 2] + values[i - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == ACTIVE_RANK) {
        int *device_values = sycl::malloc_device < int >(num_values, q);
        int *host_values = sycl::malloc_host < int >(num_values, q);

        q.memcpy(device_values, values, sizeof(values)).wait();
        q.submit([&](auto & h) {
                 h.parallel_for(sycl::range(num_values), [=] (auto index) {
                            device_values[index[0]] *= (rank + 1);
                         });
                 }).wait();
        q.memcpy(host_values, device_values, sizeof(values)).wait();

        /* Send device buffer to another rank (host recieve buffer) */
        printf("[%d] Sending GPU buffer %p to rank %d\n", rank, device_values, 1 - rank); fflush(stdout);
        MPI_Send(device_values, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD);

        /* Send device buffer to another rank (device recieve buffer) */
        printf("[%d] Sending GPU buffer %p to rank %d\n", rank, device_values, 1 - rank); fflush(stdout);
        MPI_Send(device_values, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD);

        /* Send host buffer to another rank (device recieve buffer) */
        printf("[%d] Sending host buffer %p to rank %d\n", rank, host_values, 1 - rank); fflush(stdout);
        MPI_Send(host_values, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD);
        sycl::free(device_values, q);
        sycl::free(host_values, q);
    } else {
        int result [num_values];
        int *device_result = sycl::malloc_device < int >(num_values, q);
        int *host_result = sycl::malloc_host < int >(num_values, q);

        /* Receive values from device buffer and store it in the host buffer */
        printf("[%d] Receiving data from rank %d to host buffer\n", rank, 1 - rank); fflush(stdout);
        MPI_Recv(host_result, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.memcpy(result, host_result, sizeof(result)).wait();
        VerifyResult(result, values, num_values, rank);

        /* Receive values from device buffer and store it in the device buffer */
        printf("[%d] Receiving data from rank %d to GPU buffer\n", rank, 1 - rank); fflush(stdout);
        MPI_Recv(device_result, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.memcpy(result, device_result, sizeof(result)).wait();
        VerifyResult(result, values, num_values, rank);

        /* Receive values from device buffer and store it in the device buffer */
        printf("[%d] Receiving data from rank %d to GPU buffer\n", rank, 1 - rank); fflush(stdout);
        MPI_Recv(device_result, num_values, MPI_INT, 1 - rank, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.memcpy(result, device_result, sizeof(result)).wait();
        VerifyResult(result, values, num_values, rank);

        printf("[%d] SUCCESS\n", rank);
        sycl::free(device_result, q);
        sycl::free(host_result, q);
    }

    MPI_Finalize();
    return 0;
}
