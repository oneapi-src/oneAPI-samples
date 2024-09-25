#include "mpi.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

const int Nx = 16384; /* Grid size */
const int Ny = Nx;
const int Niter = 100; /* Nuber of algorithm iterations */
const int NormIteration = 10; /* Recaluculate norm after given number of iterations. 0 to disable norm calculation */
const int PrintTime = 1; /* Output overall time of compute/communication part */

struct subarray {
    int rank, comm_size;        /* MPI rank and communicator size */
    int x_size, y_size;         /* Subarray size excluding border rows and columns */
    MPI_Aint l_nbh_offt;        /* Offset predecessor data to update */
    int up_neighbour, dn_neighbour; /* Up and sown neighbour ranks */
};

#define ROW_SIZE(S) ((S).x_size + 2)
#define XY_2_IDX(X,Y,RS) (((Y)+1)*(RS)+((X)+1))

/* This macro recalculate single point of OUT array, as an avarage of 4(top, bottom,left and right)
 * neighbours of a point from IN array.
 *
 * RECALCULATE_POINT(_OUT, _IN, _C, _R, _row_size)
 * Params:
 *      @_OUT: output array
 *      @_IN:  input array
 *      @_C:   column index
 *      @_R:   row index
 *      @_row_size: real row size
 */
#define RECALCULATE_POINT(_OUT, _IN, _C, _R, _row_size) \
    {                                                   \
        int _idx = XY_2_IDX(_C, _R, _row_size);         \
        _OUT[_idx] = 0.25 * (_IN[_idx - 1] + _IN[_idx + 1]  \
                             + _IN[_idx - _row_size]      \
                             + _IN[_idx + _row_size]);    \
    }

/* =============================================================================================================
 *    Data layout description
 *    -----------------------
 *   The data layout is a 2D grid of size (Nx+2) x (Ny+2), distributed across MPI processes along the Y-axis.
 *   Where first and last row/column areconstant and used for boundary conditions.
 *   Each porcess handles Nx x (Ny/comm_size) subarray. 
 *
 *   Overall grid format:
 * 
 *             Left border                                Right border  
 *                  |                                            |
 *                  v                                            v
 *                 ------------------------------------------------
 * Top border ---> |X|                                          |X|
 *                 ------------------------------------------------
 *                 | |                /\                        | |
 *                 | |                 |                        | |
 *                 | |                 |                        | |
 *                 | |                 |                        | |                    ------------------------------------------------
 *                 | |                 |                        | |                    |X|                                          |X| <- Last row of of i-1 subarray from previous iterarion used for calculation
 *                 | |                 |                        | |....................------------------------------------------------
 *                 | |<--------- Nx x Ny array ---------------->| |                    | |                                          | |
 *                 | |                 |                        | |                    | |          i-th process subarray           | |
 *                 | |                 |                        | |                    | |            Nx x (Ny/comm_size)           | |
 *                 | |                 |                        | |                    | |                                          | |
 *                 | |                 |                        | |....................------------------------------------------------
 *                 | |                 |                        | |                    |X|                                          |X| <- First row of of i+1 subarray from previous iterarion used for calculation
 *                 | |                 V                        | |                    ------------------------------------------------
 *                 ------------------------------------------------
 * Bottom border-> |X|                                          |X|
 *                 ------------------------------------------------
 * 
 * =============================================================================================================
 */


/* Timing utils */
double t_start;

#define BEGIN_PROFILING        \
    if (PrintTime) {           \
        t_start = MPI_Wtime(); \
    }

#define END_PROFILING  \
    if (PrintTime) {                                \
        double rank_time = MPI_Wtime() - t_start;   \
        double avg_time;                            \
        int rank;                                   \
        int comm_size;                              \
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);  \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);       \
        MPI_Reduce(&rank_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); \
        if (rank == 0) {                            \
            avg_time = avg_time / comm_size;        \
            printf("Average solver time: %f(sec)\n", avg_time); \
        }                                           \
    }




/* InitSubarryAndWindows: Initialize subarray and windows for Jacobian solver
 *      @sub: Subarray structure to initialize. Defines part of grid process is rewsponsible for.
 *      @buffers: Array of pointers to buffers used for computation. Output. Must be size of 2.
 *      @wins: Array of windows used for RMA operations. Output. Must be size of 2.
 *      @alloc_type: "host" or "device". Defines RMA windows are allocated in host or device memory.
 *      @use_passive_target: Use passive target RMA operations. If false, MPI_Win_fence is used.
 */
static void InitSubarryAndWindows(struct subarray *sub, double **buffers, MPI_Win *wins,
                                  const char *alloc_type, bool use_passive_target)
{
    /* Get own process index and total amount of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &sub->comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &sub->rank);

    /* Partititon grid across processes */
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

    /* Check if process have neighbours */
    sub->up_neighbour = (sub->rank > 0) ? sub->rank - 1 : MPI_PROC_NULL;
    sub->dn_neighbour = (sub->rank < (sub->comm_size - 1)) ? sub->rank + 1 : MPI_PROC_NULL;

    size_t total_size = sizeof(double) * (sub->x_size + 2) * (sub->y_size + 2);
    
    {   /* Allocate RMA-Windows using requested memory allocation type */
        MPI_Info info;
        MPI_Info_create(&info);
        /* allocation type could be "host" or "device" */
        MPI_Info_set(info, "mem_type", alloc_type);

        MPI_Win_allocate( total_size, sizeof(double), info, MPI_COMM_WORLD, (void*) &buffers[0], &wins[0]);
        MPI_Win_allocate( total_size, sizeof(double), info, MPI_COMM_WORLD, (void*) &buffers[1], &wins[1]);
    }

    {
        /* Create a temporary buffer */
        double *A = (double*) malloc(total_size * sizeof(double));
        for (int i = 0; i < (sub->y_size + 2); i++)
            for (int j = 0; j < (sub->x_size + 2); j++)
                A[i * (sub->x_size + 2) + j] = 0.0;

        /* set top boundary values */
        if (sub->rank == 0)
            for (int i = 1; i <= sub->x_size; i++)
                A[i] = 1.0;

        /* set bottom boundary values */
        if (sub->rank == (sub->comm_size - 1))
            for (int i = 1; i <= sub->x_size; i++)
                A[(sub->x_size + 2) * (sub->y_size + 1) + i] = 10.0;

        for (int i = 1; i <= sub->y_size; i++) {
            int row_offt = i * (sub->x_size + 2);
            A[row_offt] = 1.0;      /* set left boundary values */
            A[row_offt + sub->x_size + 1] = 1.0;    /* set right boundary values */
        }

        /* Use MPI_put to self as a memory and runtime anostic method to copy data
           from temorary buffer to buffers used for the computation. */
        if (use_passive_target) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, sub->rank, 0, wins[0]);
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, sub->rank, 0, wins[1]);
        } else {
            MPI_Win_fence(0, wins[0]);
            MPI_Win_fence(0, wins[1]);
        }

        MPI_Put((void*) A, total_size, MPI_BYTE, sub->rank, 0, total_size, MPI_BYTE, wins[0]);
        MPI_Put((void*) A, total_size, MPI_BYTE, sub->rank, 0, total_size, MPI_BYTE, wins[1]);

        if (use_passive_target) {
            MPI_Win_unlock(sub->rank, wins[0]);
            MPI_Win_unlock(sub->rank, wins[1]);
        } else {
            MPI_Win_fence(0, wins[0]);
            MPI_Win_fence(0, wins[1]);
        }
        free(A);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return;
}