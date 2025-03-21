        !=============================================================
        ! Copyright © 2024 Intel Corporation
        !
        ! SPDX-License-Identifier: MIT
        !=============================================================
! Snippet begin
        include "mkl_omp_offload.f90"

        subroutine init (a, b, c, m, k, n)
        implicit none

        real :: a(m, k), b(k,n), c(m,n)
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
        end subroutine init


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
        real                :: alpha, beta, sum, total
        integer             :: i, j1, j2
        double precision    :: t0, t1

        !$omp allocators allocate(allocator(omp_target_shared_mem_alloc): a)
        allocate( a(1 : m, 1 : k) )

        !$omp allocators allocate(allocator(omp_target_shared_mem_alloc): b)
        allocate( b(1 : k, 1 : n) )

        !$omp allocators allocate(allocator(omp_target_shared_mem_alloc): c)
        allocate( c(1 : m, 1 : n) )

        ! Initialize.

        alpha = 1.025
        beta  = 1.0
        total = 0.0
        call init (a, b, c, m, k, n)

        ! Compute sgemm on the device.

        t0 = omp_get_wtime()

        do i = 1, iter
            ! Update arrays a and b.
            a(:,:) = a(:,:) + 1
            b(:,:) = b(:,:) - 1

            ! Compute sgemm on the device.
            !$omp dispatch
            call sgemm('n','n',m,n,k,alpha,a,m,b,k,beta,c,m)

            sum = 0.0
            !$omp target teams distribute parallel do collapse(2) reduction(+:sum)
            do j1 = 1, m
               do j2 = 1, n
                  sum = sum + c(j1,j2)
               enddo
            enddo
            !$omp end target teams distribute parallel do

            total = total + sum
        end do

        t1 = omp_get_wtime()

        print *, "total = ", total
        write (*, 120) " Number of iterations = ", iter
        write (*, 130) " Time = ", t1-t0, " seconds"
 120    format (A, I4)
 130    format (A, F10.3, A)

        ! Deallocate arrays.
        deallocate(a)
        deallocate(b)
        deallocate(c)

        end program main
! Snippet end
