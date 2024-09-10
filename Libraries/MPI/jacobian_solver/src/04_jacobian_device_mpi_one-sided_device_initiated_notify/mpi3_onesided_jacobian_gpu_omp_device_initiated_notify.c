/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Distributed Jacobian computation sample using OpenMP GPU offload and device-initiated MPI-3 one-sided communication.
 * This sample also demonstrates notified RMA operations usage.
 */

#include "../include/common.h"
#ifndef MPI_ERR_INVALID_NOTIFICATION
/*For Intel MPI 2021.13/14 we have to use API compatibility layer*/
#include "mpix_compat.h"
#endif

#include <omp.h>
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
    int provided;

    /* Initialization of runtime and initial state of data */
    /* MPI_THREAD_MULTIPLE is required for device-initiated communications */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE is required for this sample\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Initialize subarray owned by current process
     * and create RMA-windows for MPI-3 one-sided communications.
     *  - For this sample, we use GPU memory for buffers and windows.
     *  - Sample uses MPI_Win_lock* for synchronization.
     */
    InitSubarryAndWindows(&my_subarray, buffs, win, "device", true);

    /* Enable notification counters */
    MPI_Win_notify_set_num(win[0], MPI_INFO_NULL, 1);
    MPI_Win_notify_set_num(win[1], MPI_INFO_NULL, 1);
    /* Start RMA exposure epoch */
    MPI_Win_lock_all(0, win[0]);
    MPI_Win_lock_all(0, win[1]);

    const int row_size = ROW_SIZE(my_subarray);
    /* Amount of iterations to perform between norm calculations */
    const int iterations_batch = (NormIteration <= 0) ? Niter : NormIteration;
    /* Aux variables used to let OMP capture pointers */
    double *b1 = buffs[0], *b2 = buffs[1];
    /* iter_counter_step defines a notification counter step per iteration */
    const MPI_Count iter_counter_step =
        ((my_subarray.up_neighbour != MPI_PROC_NULL) ? 1 : 0) +
        ((my_subarray.dn_neighbour != MPI_PROC_NULL) ? 1 : 0);

    /* Timestamp start time to measure overall execution time */
    BEGIN_PROFILING
    /* Main computation loop offloaded to the device:
     * "#pragma omp target data" maps the data to the device memory for a following code region*/
    #pragma omp target data map(to: my_subarray, win[0:2], iterations_batch, iter_counter_step) use_device_ptr(b1, b2)
    {
        for (int passed_iters = 0; passed_iters < Niter; passed_iters += iterations_batch) {
            /* Offload compute loop to the device:
             * "#pragma omp target teams" start a target region with a single team 
             *
             * NOTE: For simplification and unification across samples we use single team
             *       to avoid extra syncronization across teams in the future */
            #pragma omp target thread_limit(1024)
            {
                for (int k = 0; k < iterations_batch; ++k)
                {
                    int i = passed_iters + k;
                    MPI_Win prev_win = win[i % 2];
                    MPI_Win current_win = win[(i + 1) % 2];
                    double *in = (i % 2) ? b1 : b2;
                    double *out = ((1 + i) % 2) ? b1 : b2;

                    /* Wait for notification counter to reach the expected value:
                     *  here we check that communication operations issued by peers on the previous iteration are completed
                     *  and data is ready for the next iteration.
                     * 
                     * NOTE:
                     *  To be completely standard compliant, application should check memory model
                     *  and call MPI_Win_sync(prev_win) in case of MPI_WIN_SEPARATE mode after notification has been recieved.
                     *  Although, IntelMPI uses MPI_WIN_UNIFIED memory model, so this call could be omitted.
                     */
                    MPI_Count c = 0;
                    MPI_Win_flush_local_all(current_win);
                    while (c < (iter_counter_step*i)) {
                        MPI_Win_notify_get_value(prev_win, 0, &c);
                    }

                    /* Start parallel loop on the device, to accelerate a calculation */
                    #pragma omp parallel for
                    /* Calculate values on borders to initiate communications early */
                    for (int column = 0; column < my_subarray.x_size;  column ++) {
                        RECALCULATE_POINT(out, in, column, 0, row_size);
                        RECALCULATE_POINT(out, in, column, my_subarray.y_size - 1, row_size);
                    }

                    /* Perform 1D halo-exchange with neighbours.
                     *    Here we uses extention primitives which allows to notify remote process about data readiness.
                     *    This approach allows us to relax syncronization requirement between origin and target processes.
                     * 
                     * This code is executed outside of parallel section, but still on the device.
                     * It is possible to use MPI_Put_notify in parallel region, which may have better performance for
                     * scale-up cases, but would have additional overhead for scale-out cases.
                     * Also, in this case iter_counter_step should be adjusted.
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


                    /* Start parallel loop on the device, to accelerate a calculation */
                    #pragma omp parallel for collapse(2)
                    /* Recalculate internal points in parallel with communication */
                    for (int row = 1; row < my_subarray.y_size - 1; ++row) {
                        for (int column = 0; column < my_subarray.x_size; ++column) {
                            RECALCULATE_POINT(out, in, column, row, row_size);
                        }
                    }
                }
            }

            /* Calculate norm value after given number of iterations */
            if (NormIteration > 0) {
                double result_norm = 0.0;
                double norm = 0.0;

                #pragma omp target teams distribute parallel for simd is_device_ptr(b1, b2) reduction(+:norm) collapse(2)
                for (int row = 0; row < my_subarray.y_size; ++row) {
                    for (int column = 0; column < my_subarray.x_size; ++column) {
                        int idx = XY_2_IDX(column, row, row_size);
                        double diff = b1[idx] - b2[idx];
                        norm += diff*diff;
                    }
                }
                MPI_Reduce(&norm, &result_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (my_subarray.rank == 0) {
                    printf("NORM value on iteration %d: %f\n", passed_iters+iterations_batch, sqrt(result_norm));
                }
            }
        }
    }
    /* Timestamp end time to measure overall execution time and report average compute time */
    END_PROFILING

    /* Close RMA exposure epoch and free resources */
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
