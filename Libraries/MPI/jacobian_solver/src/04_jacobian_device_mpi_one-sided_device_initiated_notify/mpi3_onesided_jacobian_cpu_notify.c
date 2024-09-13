/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Distributed Jacobian computation sample using CPU computations and MPI-3 one-sided communication.
 * This sample also demonstrates notified RMA operations usage.
 */

#include "../include/common.h"
#ifndef MPI_ERR_INVALID_NOTIFICATION
/* For Intel MPI 2021.13/14 we have to use API compatibility layer */
#include "mpix_compat.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    double t_start;
    struct subarray my_subarray = { };
    /* Here we use double buffering to allow the overlap of the compute and communication phases.
     * Odd iterations use buffs[0] as input and buffs[1] as output, and vice versa.
     * The same scheme is used for MPI_Win objects.
     */
    double *buffs[2] = { NULL, NULL };
    MPI_Win win[2] = { MPI_WIN_NULL, MPI_WIN_NULL };

    /* Initialization of runtime and initial state of data */
    MPI_Init(&argc, &argv);

    /* Initialize the subarray owned by the current process
     * and create RMA windows for MPI-3 one-sided communications.
     *  - For this sample, we use GPU memory for buffers and windows.
     *  - This sample uses MPI_Win_lock* for synchronization.
     */
    InitSubarryAndWindows(&my_subarray, buffs, win, "host", true);

    /* Enable notification counters */
    MPI_Win_notify_set_num(win[0], MPI_INFO_NULL, 1);
    MPI_Win_notify_set_num(win[1], MPI_INFO_NULL, 1);
    /* Start the RMA exposure epoch */
    MPI_Win_lock_all(0, win[0]);
    MPI_Win_lock_all(0, win[1]);

    const int row_size = ROW_SIZE(my_subarray);
    /* Number of iterations to perform between norm calculations */
    const int iterations_batch = (NormIteration <= 0) ? Niter : NormIteration;
    /* Auxiliary variables used to let OMP capture pointers */
    double *b1 = buffs[0], *b2 = buffs[1];
    /* iter_counter_step defines a notification counter step per iteration */
    const MPI_Count iter_counter_step =
        ((my_subarray.up_neighbour != MPI_PROC_NULL) ? 1 : 0) +
        ((my_subarray.dn_neighbour != MPI_PROC_NULL) ? 1 : 0);

    /* Timestamp the start time to measure overall execution time */
    BEGIN_PROFILING
    for (int passed_iters = 0; passed_iters < Niter; passed_iters += iterations_batch) {
        /* Perform a batch of iterations before checking the norm */
        for (int k = 0; k < iterations_batch; ++k)
        {
            int i = passed_iters + k;
            MPI_Win current_win = win[(i + 1) % 2];
            double *in = buffs[i % 2];
            double *out = buffs[(1 + i) % 2];


            /* Calculate values on the borders to initiate communications early */
            for (int column = 0; column < my_subarray.x_size; column++) {
                RECALCULATE_POINT(out, in, column, 0, row_size);
                RECALCULATE_POINT(out, in, column, my_subarray.y_size - 1, row_size);
            }

            /* Perform 1D halo-exchange with neighbors.
             * Here we use extension primitives which allow notifying the remote process about data readiness.
             * This approach allows us to relax synchronization requirements between origin and target processes.
             * 
             * This code is executed outside of the parallel section but still on the device.
             * It is possible to use MPI_Put_notify in the parallel region, which may have better performance for
             * scale-up cases but would have additional overhead for scale-out cases.
             * Also, in this case, iter_counter_step should be adjusted.
             */
            if (my_subarray.up_neighbour != MPI_PROC_NULL) {
                int idx = XY_2_IDX(0, 0, row_size);
                MPI_Put_notify(&out[idx], my_subarray.x_size, MPI_DOUBLE,
                        my_subarray.up_neighbour, my_subarray.l_nbh_offt,
                        my_subarray.x_size, MPI_DOUBLE, 0, current_win);
            }

            if (my_subarray.dn_neighbour != MPI_PROC_NULL) {
                int idx = XY_2_IDX(0, my_subarray.y_size - 1, row_size);
                MPI_Put_notify(&out[idx], my_subarray.x_size, MPI_DOUBLE,
                        my_subarray.dn_neighbour, 1,
                        my_subarray.x_size, MPI_DOUBLE, 0, current_win);
            }

            /* Recalculate internal points in parallel with communication */
            for (int row = 1; row < my_subarray.y_size - 1; ++row) {
                for (int column = 0; column < my_subarray.x_size; ++column) {
                    RECALCULATE_POINT(out, in, column, row, row_size);
                }
            }

            /* Wait for the notification counter to reach the expected value:
             * here we check that communication operations issued by peers on the previous iteration are completed
             * and data is ready for the next iteration.
             * 
             * NOTE:
             * To be completely standard compliant, the application should check the memory model
             * and call MPI_Win_sync(prev_win) in case of MPI_WIN_SEPARATE mode after the notification has been received.
             * Although, IntelMPI uses the MPI_WIN_UNIFIED memory model, so this call could be omitted.
             */
            MPI_Count c = 0;
            MPI_Win_flush_all(current_win);
            while (c < iter_counter_step) {
                MPI_Win_notify_get_value(current_win, 0, &c);
            }
            MPI_Win_notify_set_value(current_win, 0, 0);
        }

        /* Calculate the norm value after the given number of iterations */
        if (NormIteration > 0) {
            double result_norm = 0.0;
            double norm = 0.0;

            for (int row = 0; row < my_subarray.y_size; ++row) {
                for (int column = 0; column < my_subarray.x_size; ++column) {
                    int idx = XY_2_IDX(column, row, row_size);
                    double diff = b1[idx] - b2[idx];
                    norm += diff * diff;
                }
            }
            MPI_Reduce(&norm, &result_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (my_subarray.rank == 0) {
                printf("NORM value on iteration %d: %f\n", passed_iters + iterations_batch, sqrt(result_norm));
            }
        }
    }
    /* Timestamp the end time to measure overall execution time and report average compute time */
    END_PROFILING

    /* Close the RMA exposure epoch and free resources */
    MPI_Win_unlock_all(win[1]);
    MPI_Win_unlock_all(win[0]);
    MPI_Win_free(&win[1]);
    MPI_Win_free(&win[0]);

    if (my_subarray.rank == 0) {
        printf("SUCCESS\n");
    }
    MPI_Finalize();

    return 0;
}
