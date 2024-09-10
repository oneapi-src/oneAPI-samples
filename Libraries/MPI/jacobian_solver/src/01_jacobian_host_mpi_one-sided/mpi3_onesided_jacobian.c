/*==============================================================
 * Copyright © 2024 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Distributed Jacobian computation sample using MPI-3 one-sided communications and host memory.
 */

#include "../include/common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char *argv[])
{
    double t_start;
    struct subarray my_subarray = { };
    /* Here we uses double buffering to allow overlap of compute and communication phase.
     * Odd iterations use buffs[0] as input and buffs[1] as output and vice versa.
     * Same scheme is used for MPI_Win objects.
     */
    double *buffs[2] = { NULL, NULL };
    MPI_Win win[2] = { MPI_WIN_NULL, MPI_WIN_NULL };

    /* Initialization of runtime and initial state of data */
    MPI_Init(&argc, &argv);
    /* Initialize subarray owned by current process
     * and create RMA-windows for MPI-3 one-sided communications.
     *  - For this sample, we use host memory for buffers and windows.
     *  - Sample uses MPI_Win_fence for synchronization.
     */
    InitSubarryAndWindows(&my_subarray, buffs, win, "host", false);
    
    /* Start RMA exposure epoch */
    MPI_Win_fence(0, win[0]);
    MPI_Win_fence(0, win[1]);

    const int row_size = ROW_SIZE(my_subarray);
    /* Amount of iterations to perform between norm calculations */
    const int iterations_batch = (NormIteration <= 0) ? Niter : NormIteration;

    /* Timestamp start time to measure overall execution time */
    BEGIN_PROFILING
    /* Main computation loop */
    for (int passed_iters = 0; passed_iters < Niter; passed_iters += iterations_batch) {
        /* Perfrom a batch of iterations before checking norm */
        for (int k = 0; k < iterations_batch; ++k) {
            int i = passed_iters + k;
            double *in = buffs[i % 2];
            double *out = buffs[(1 + i) % 2];
            MPI_Win current_win = win[(i + 1) % 2];

            /* Calculate values on borders to initiate communications early */
            for (int column = 0; column < my_subarray.x_size; ++column) {
                RECALCULATE_POINT(out, in, column, 0, row_size);
                RECALCULATE_POINT(out, in, column, my_subarray.y_size - 1, row_size);
            }

            /* Perform 1D halo-exchange with neighbours */
            if (my_subarray.up_neighbour != MPI_PROC_NULL) {
                int idx = XY_2_IDX(0, 0, row_size);
                MPI_Put(&out[idx], my_subarray.x_size, MPI_DOUBLE,
                        my_subarray.up_neighbour, my_subarray.l_nbh_offt,
                        my_subarray.x_size, MPI_DOUBLE, current_win);
            }

            if (my_subarray.dn_neighbour != MPI_PROC_NULL) {
                int idx = XY_2_IDX(0, my_subarray.y_size - 1, row_size);
                MPI_Put(&out[idx], my_subarray.x_size, MPI_DOUBLE,
                        my_subarray.dn_neighbour, 1,
                        my_subarray.x_size, MPI_DOUBLE, current_win);
            }

            /* Recalculate internal points in parallel with communication */
            for (int row = 1; row < my_subarray.y_size - 1; ++row) {
                for (int column = 0; column < my_subarray.x_size; ++column) {
                    RECALCULATE_POINT(out, in, column, row, row_size);
                }
            }

            /* Ensure all communications are complete before next iteration */
            MPI_Win_fence(0, current_win);
        }

        /* Calculate norm value after given number of iterations */
        if (NormIteration > 0) {
            double result_norm = 0.0;
            double norm = 0.0;
            double *in = buffs[0];
            double *out = buffs[1];
            for (int row = 0; row < my_subarray.y_size; ++row) {
                for (int column = 0; column < my_subarray.x_size; ++column) {
                    int idx = XY_2_IDX(column, row, row_size);
                    double diff = out[idx] - in[idx];
                    norm += diff*diff;
                }
            }
            MPI_Reduce(&norm, &result_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (my_subarray.rank == 0) {
                printf("NORM value on iteration %d: %f\n", passed_iters+iterations_batch, sqrt(result_norm));
            }
        }
    }
    /* Timestamp end time to measure overall execution time and report average compute time */
    END_PROFILING

    /* Close RMA exposure epoch and free resources */
    MPI_Win_fence(0, win[0]);
    MPI_Win_fence(0, win[1]);
    MPI_Win_free(&win[1]);
    MPI_Win_free(&win[0]);

    if (my_subarray.rank == 0) {
        printf("SUCCESS\n");
    }
    MPI_Finalize();

    return 0;
}