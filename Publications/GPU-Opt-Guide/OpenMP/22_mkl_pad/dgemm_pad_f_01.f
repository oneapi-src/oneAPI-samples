        !=============================================================
        ! Copyright © 2022 Intel Corporation
        !
        ! SPDX-License-Identifier: MIT
        !=============================================================
        ! Snippet begin
        include "mkl_omp_offload.f90"

     ! This subroutine reads command line arguments m1, k1, and n1.
      subroutine get_arguments (m1, k1, n1)
        implicit none
        integer           :: m1, k1, n1
        character(len=32) :: m1_char, k1_char, n1_char

     ! First, make sure that the right number of command line arguments
     ! have been provided.
        if (command_argument_count() .ne. 3) then
          print *, "ERROR: Three command-line arguments expected; stopping."
          stop
        endif

     ! Get command line arguments.
        call get_command_argument(1, m1_char)
        call get_command_argument(2, k1_char)
        call get_command_argument(3, n1_char)

     ! Convert arguments to integers.
        read (m1_char,*) m1
        read (k1_char,*) k1
        read (n1_char,*) n1
      end subroutine get_arguments


     ! This function returns the smallest multiple of 8 that is >= n.
     ! Examples:
     ! if n =  3, then get_mul8 =  8
     ! if n =  9, then get_mul8 = 16
     ! if n = 30, then get_mul8 = 32
     ! if n = 80, then get_mul8 = 8
      integer function get_mul8 (n)
        implicit none
        integer :: n
        integer :: mod
        if (mod(n,8) .eq. 0) then
            get_mul8 = n
        else
            get_mul8 = ((n/8) + 1) * 8
        endif
      end function get_mul8


     ! This subroutine initializes matrices.
      subroutine init_matrix (m, k, n, a, b, c)
        implicit none
        integer          :: m, k, n
        double precision :: a(m,k), b(k,n), c(m,n)
        integer          :: i, j

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
            c(i,j) = 0.2 + i - j
          end do
        end do
      end subroutine init_matrix


      program DGEMM_MAIN

#if defined(MKL_ILP64)
        use onemkl_blas_omp_offload_ilp64
#else
        use onemkl_blas_omp_offload_lp64
#endif
        use omp_lib
        use iso_fortran_env
        implicit none

        interface
          integer function get_mul8 (n)
            implicit none
            integer :: n
          end function get_mul8
        end interface

        double precision :: alpha, beta
        integer :: m1, k1, n1, m2, k2, n2
        double precision, allocatable :: a1(:,:)
        double precision, allocatable :: b1(:,:)
        double precision, allocatable :: c1(:,:)

        double precision, allocatable :: a2(:,:)
        double precision, allocatable :: b2(:,:)
        double precision, allocatable :: c2(:,:)

        double precision :: start_t1, end_t1
        double precision :: start_t2, end_t2

     ! Read command line arguments m1, k1, and n1.

       call get_arguments (m1, k1, n1)

     !
     ! Initialize alpha, beta, and m2, k2, n2
     !

      alpha = 1.025
      beta  = 0.75

      m2 = get_mul8(m1)
      k2 = get_mul8(k1)
      n2 = get_mul8(n1)

     !
     ! Allocate and initialize matrices.
     !
      allocate( a1(1:m1,1:k1) )
      allocate( b1(1:k1,1:n1) )
      allocate( c1(1:m1,1:n1) )
      allocate( a2(1:m2,1:k2) )
      allocate( b2(1:k2,1:n2) )
      allocate( c2(1:m2,1:n1) )
      call init_matrix (m1, k1, n1, a1, b1, c1)
      call init_matrix (m2, k2, n2, a2, b2, c2)


     !$omp target data map(to: a1, b1, a2, b2) map(tofrom: c1, c2)

     ! Warm up run on device
        !$omp dispatch
        call DGEMM('N','N',m1,n1,k1,alpha,a1,m1,b1,k1,beta,c1,m1)

     !
     ! Run DGEMM on device (using matrices a1, b1, and c1)
     !
        start_t1 = omp_get_wtime()

        !$omp dispatch
        call DGEMM('N','N',m1,n1,k1,alpha,a1,m1,b1,k1,beta,c1,m1)

        end_t1 = omp_get_wtime()


     ! Warm up run on device
        !$omp dispatch
        call DGEMM('N','N',m2,n2,k2,alpha,a2,m2,b2,k2,beta,c2,m2)

     !
     ! Run DGEMM on device (using padded matrices a2, b2, and c2)
     !
        start_t2 = omp_get_wtime()

        !$omp dispatch
        call DGEMM('N','N',m2,n2,k2,alpha,a2,m2,b2,k2,beta,c2,m2)

        end_t2 = omp_get_wtime()

        !$omp end target data

        print 100, alpha, beta
        print *
        print 101, m1, n1, k1
        print 111, (end_t1 - start_t1)
        print *
        print 102, m2, n2, k2
        print 112, (end_t2 - start_t2)

 100    format(7x, "ALPHA =", f10.4, "  BETA =",f10.4)
 101    format(7x, "M1 =", i5,"  N1 =", i5, "  K1 =",i5)
 111    format(7x, "Time (non-padded arrays)  =", f10.4, " sec")
 102    format(7x, "M2 =", i5,"  N2 =", i5, "  K2 =",i5)
 112    format(7x, "Time (padded arrays) =", f10.4, " sec")

        end
        ! Snippet end
