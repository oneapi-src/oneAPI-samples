        !=============================================================
        ! Copyright © 2024 Intel Corporation
        !
        ! SPDX-License-Identifier: MIT
        !=============================================================
        include "mkl_omp_offload.f90"

        subroutine init_arrays (a, b, c, m, k, n)
        implicit none

        real, intent(inout) :: a(m, k), b(k,n), c(m,n)
        integer m, k, n, i, j

        do i = 1, m
          do j = 1, k
            a(i, j) = i
          end do
        end do

        do i = 1, k
          do j = 1, n
            b(i, j) = j - 1
          end do
        end do

        do i = 1, m
          do j = 1, n
            c(i, j) = 0.2 + i - j
          end do
        end do
        end subroutine init_arrays


        program main

#if defined(MKL_ILP64)
        use onemkl_blas_omp_offload_ilp64
#else
        use onemkl_blas_omp_offload_lp64
#endif
        use omp_lib
        use iso_fortran_env
        implicit none

        integer, parameter  :: m = 1024
        integer, parameter  :: k = 1024
        integer, parameter  :: n = 1024
        integer, parameter  :: iter = 2000
        real, allocatable   :: a(:, :), b(:, :), c(:, :)
        real                :: alpha, beta
        integer             :: i
        double precision    :: t0, t1

! Snippet begin 1
        allocate( a(1 : m, 1 : k) )
        allocate( b(1 : k, 1 : n) )
        allocate( c(1 : m, 1 : n) )
! Snippet end 1

        ! Initialize.

        alpha = 1.025
        beta  = 0.750
        call init_arrays (a, b, c, m, k, n)

        ! Compute sgemm on the device.

        t0 = omp_get_wtime()

! Snippet begin 2
        !$omp target data map(to: a, b) map(tofrom: c)

        do i = 1, iter
            ! Update arrays a and b on the host.
            a(:,:) = a(:,:) + 1
            b(:,:) = b(:,:) - 1

            ! Copy new values of a and b to the device.
            !$omp target update to (a, b)

            ! Compute sgemm on the device.
            !$omp dispatch
            call sgemm('n','n',m,n,k,alpha,a,m,b,k,beta,c,m)
        end do

        !$omp end target data
! Snippet end 2

        t1 = omp_get_wtime()

        print *, c(1,1), c(m/2,n/2), c(m,n)
        write (*, 120) " Number of iterations = ", iter
        write (*, 130) " Time = ", t1-t0, " seconds"
 120    format (A, I4)
 130    format (A, F10.3, A)

        ! Deallocate arrays.
        deallocate(a)
        deallocate(b)
        deallocate(c)

        end program main
