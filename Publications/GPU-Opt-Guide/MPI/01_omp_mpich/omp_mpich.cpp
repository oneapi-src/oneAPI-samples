//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>

#define N 1000000
#define STEPS 1000
#define ABS(x) (x) > 0 ? (x) : -(x)

int main (int argc, char **argv )
{
   int mpi_aware = 0;
   if ( argc > 1 ) {
       mpi_aware = 1;
       printf("MPI device aware path enabled\n");
   } // argc check

   MPI_Init(NULL, NULL);
   int rank,nranks;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);

   int next_rank = ( rank + 1 ) % nranks;
   int prev_rank = rank == 0 ? nranks-1 : rank-1;
   printf("rank=%d next=%d prev=%d\n",rank,next_rank,prev_rank);

#pragma omp target
   ;

   double buf1[N],buf2[N];
   double *curr,*next,*tmp;

   for ( int  i = 0; i < N; i++ ) {
      buf1[i] = 0;
      buf2[i] = 0;
   }

   MPI_Request psrq;
   double start = omp_get_wtime();
#pragma omp target data map(buf1,buf2)
{
  #pragma omp target data use_device_addr(buf1,buf2) if(mpi_aware)
  {
    curr = buf1;
    next = buf2;
  }
  printf("curr=%p next=%p\n",curr,next);

  for ( int step = 0; step < STEPS; step++ ) {
     if ( rank == 0 && step % 100 == 0 ) printf("step: %d\n",step);

#pragma omp target teams distribute parallel for
     for ( int i = 0; i < N; i++ ) curr[i]++;

     if ( nranks > 1 ) {
        #pragma omp target update from(curr[0:N]) if(!mpi_aware)
        MPI_Request srq;
        MPI_Isend(curr,N,MPI_DOUBLE,next_rank,0,MPI_COMM_WORLD,&srq);
	// we need to make sure that the MPI_Isend of the previous
	// iteration finished before doing the MPI_Recv of this
	// iteration
	if ( step > 0 ) MPI_Wait(&psrq,MPI_STATUS_IGNORE);
	psrq = srq;
        MPI_Recv(next,N,MPI_DOUBLE,prev_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        #pragma omp target update to(next[0:N]) if(!mpi_aware)
     } // nranks

     tmp = curr;
     curr = next;
     next = tmp;
   }
}

   MPI_Barrier(MPI_COMM_WORLD);
   double end = omp_get_wtime();
   printf("rank %d total_time=%g\n",rank, end-start);

   for ( int i = 0; i < N; i++ ) {
       if ( buf1[i] != STEPS ) {
         printf("Error in %d = %f\n",i,buf1[i]);
         break;
       }
   }

   return 0;
}
