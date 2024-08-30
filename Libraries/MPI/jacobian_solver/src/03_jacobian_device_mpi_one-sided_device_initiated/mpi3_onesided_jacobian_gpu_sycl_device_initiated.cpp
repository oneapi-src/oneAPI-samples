/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Distributed Jacobian computation sample using SYCL GPU offload and device-initiated MPI-3 one-sided.
 */
#include "../include/common.h"
#include <sycl.hpp>
#include <vector>
#include <iostream>

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
     *  - Sample uses MPI_Win_fence for synchronization.
     */
    InitSubarryAndWindows(&my_subarray, buffs, win, "device", false);
    /* Create SYCL GPU queue */
    sycl::queue q(sycl::gpu_selector_v);
    /* NOTE: For simplification and unification across samples we use single workgroup
     *       to avoid extra syncronization across workgroups in the future.
     */
#ifdef GROUP_SIZE_DEFAULT
    int work_group_size = GROUP_SIZE_DEFAULT;
#else
    int work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
#endif

    /* Start RMA exposure epoch */
    MPI_Win_fence(0, win[0]);
    MPI_Win_fence(0, win[1]);
    
    const int row_size = ROW_SIZE(my_subarray);
    /* Amount of iterations to perform between norm calculations */
    const int iterations_batch = (NormIteration <= 0) ? Niter : NormIteration;

    /* Calculate amount of columns per workgroup */
    const int col_per_wg = my_subarray.x_size / work_group_size;
    const int tail = my_subarray.y_size % my_subarray.comm_size;

    /* Timestamp start time to measure overall execution time */
    BEGIN_PROFILING
    for (int passed_iters = 0; passed_iters < Niter; passed_iters += iterations_batch) {

        /* Submit a compute kernel to calculate next "iterations_batch" steps */
        q.submit([&](auto & h) {
            h.parallel_for(sycl::nd_range<1>(work_group_size, work_group_size),
                            [=](sycl::nd_item<1> item) {
                int id = item.get_global_id(0);
                const int my_x_lb = col_per_wg * id + ((id < tail) ? id : tail);

                for (int k = 0; k < iterations_batch; ++k)
                {
                    int i = passed_iters + k;
                    MPI_Win current_win = win[(i + 1) % 2];
                    double *in = buffs[i % 2];
                    double *out = buffs[(1 + i) % 2];
                    /* Calculate values on borders to initiate communications early */
                    int column;
                    for (column = 0; (column + work_group_size) < my_subarray.x_size; 
                                column += work_group_size) {
                        RECALCULATE_POINT(out, in, column+id, 0, row_size);
                        RECALCULATE_POINT(out, in, column+id, my_subarray.y_size - 1, row_size);
                    }

                    if (id < my_subarray.x_size % work_group_size) {
                        RECALCULATE_POINT(out, in, column+id, 0, row_size);
                        RECALCULATE_POINT(out, in, column+id, my_subarray.y_size - 1, row_size);
                    }

                    item.barrier(sycl::access::fence_space::global_space);
                    /* Perform 1D halo-exchange with neighbours */
                    if (my_subarray.up_neighbour != MPI_PROC_NULL) {
                        int idx = XY_2_IDX(my_x_lb, 0, row_size);
                        MPI_Put(&out[idx], col_per_wg, MPI_DOUBLE,
                            my_subarray.rank - 1, my_subarray.l_nbh_offt+my_x_lb,
                            col_per_wg, MPI_DOUBLE, current_win);
                    }

                    if (my_subarray.dn_neighbour != MPI_PROC_NULL)  {
                        int idx = XY_2_IDX(my_x_lb, my_subarray.y_size - 1, row_size);
                        MPI_Put(&out[idx], col_per_wg, MPI_DOUBLE,
                            my_subarray.rank + 1, 1+my_x_lb,
                            col_per_wg, MPI_DOUBLE, current_win);
                    }


                    /* Recalculate internal points in parallel with comunications */
                    for (int row = 1; row < my_subarray.y_size - 1; ++row) {
                        int column = 0;
                        for (; (column + work_group_size) < my_subarray.x_size; 
                                column += work_group_size) {
                            RECALCULATE_POINT(out, in, column+id, row, row_size);
                        }
                        if (id < my_subarray.x_size % work_group_size) {
                            RECALCULATE_POINT(out, in, column+id, row, row_size);
                        }
                    }

                    item.barrier(sycl::access::fence_space::global_space);
                    /* Ensure all communications complete before next iteration.
                     * Syncronization primitives called within a kernel have the same
                     * limitations as they have been called from regular threads.
                     * Because of that, we have to syncronizeall threads of a workgroup using a barriar,
                     * then call MPI_Win_fence from a single thread. 
                     */
                    if (id == 0) {
                        MPI_Win_fence(0, current_win);
                    }
                    item.barrier(sycl::access::fence_space::global_space);
                }
            });
        }).wait();


        /* Calculate norm value after given number of iterations */
        if (NormIteration > 0) {
            double result_norm = 0.0;
            double norm = 0.0;
            double *a = buffs[0];
            double *a_out = buffs[1];

            {
                sycl::buffer<double> norm_buf(&norm, 1);
                q.submit([&](auto & h) {
                    auto sumr = sycl::reduction(norm_buf, h, sycl::plus<>());
                    h.parallel_for(sycl::range(my_subarray.x_size, my_subarray.y_size), sumr, [=] (auto index, auto &v) {
                        int idx = XY_2_IDX(index[0], index[1], row_size);
                        double diff = a_out[idx] - a[idx];
                        v += (diff * diff);
                    });
                }).wait();
            }

            MPI_Reduce(&norm, &result_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (my_subarray.rank == 0) {
                printf("NORM value on iteration %d: %f\n", passed_iters + iterations_batch, sqrt(result_norm));
            }
        }
    }
    END_PROFILING

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
