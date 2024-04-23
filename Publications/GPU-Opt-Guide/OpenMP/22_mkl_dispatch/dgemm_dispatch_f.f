        !=============================================================
        ! Copyright © 2022 Intel Corporation
        !
        ! SPDX-License-Identifier: MIT
        !=============================================================
        ! Snippet begin
        include "mkl_omp_offload.f90"

        program DGEMM_MAIN

#if defined(MKL_ILP64)
        use onemkl_blas_omp_offload_ilp64
#else
        use onemkl_blas_omp_offload_lp64
#endif
        use omp_lib
        use iso_fortran_env
        implicit none

        integer, parameter :: m = 20
        integer, parameter :: k = 5
        integer, parameter :: n = 10
        double precision   a(m,k), b(k,n), c1(m,n), c2(m,n)
        double precision   alpha, beta
        integer            i, j

        print*
        print*,'   D G E M M  EXAMPLE PROGRAM'


        ! Initialize

        alpha = 1.025
        beta  = 0.75

        do i = 1, m
          do j = 1, k
            a(i,j) = (i-1) - (0.25 * k)
          end do
        end do

        do i = 1, k
          do j = 1, n
            b(i,j) = -((i-1) + j)
          end do
        end do

        do i = 1, m
          do j = 1, n
            c1(i,j) = 0.2 + i - j
            c2(i,j) = 0.2 + i - j
          end do
        end do


        ! Execute DGEMM on host.

        call DGEMM('N','N',m,n,k,alpha,a,m,b,k,beta,c1,m)

        print *
        print *, 'c1 - After DGEMM host execution'

        do i=1,m
           print 110, (c1(i,j),j=1,n)
        end do
        print*


        ! Execute DGEMM on device

!$omp target data map(to: a, b) map(tofrom: c2)

!$omp dispatch
        call DGEMM('N','N',m,n,k,alpha,a,m,b,k,beta,c2,m)

!$omp end target data

        print *
        print *, 'c2 - After DGEMM device execution'

        do i=1,m
           print 110, (c2(i,j),j=1,n)
        end do
        print *

 110    format(7x,10(f10.2,2x))

        end
        ! Snippet end
