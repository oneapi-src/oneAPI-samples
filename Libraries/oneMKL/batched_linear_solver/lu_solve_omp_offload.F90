!===============================================================================
! Copyright 2021-2022 Intel Corporation.
!
! This software and the related documents are Intel copyrighted  materials,  and
! your use of  them is  governed by the  express license  under which  they were
! provided to you (License).  Unless the License provides otherwise, you may not
! use, modify, copy, publish, distribute,  disclose or transmit this software or
! the related documents without Intel's prior written permission.
!
! This software and the related documents  are provided as  is,  with no express
! or implied  warranties,  other  than those  that are  expressly stated  in the
! License.
!===============================================================================
!
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL)
!      FORTRAN OpenMP offload examples for solving batched linear systems.
!
! Compile for CPU:
!     ifx -i8 -qmkl -free \
!         lu_solve_omp_offload_ex1_timer.F90 -o lu_solve_ex1_timer
!
! Compile for GPU:
!     ifx -i8 -DMKL_ILP64 -qopenmp -fopenmp-targets=spir64 -fsycl -free \
!         lu_solve_omp_offload_ex1_timer.F90 -o lu_solve_ex1_omp_timer \
!         -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 \
!         -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl
!
! Compile with -DSP to use single precision instead of double precision.
!
!*******************************************************************************

!$ include "mkl_omp_offload.f90"

program solve_batched_linear_systems

! Decide whether to use 32- or 64-bit integer type
#if defined(MKL_ILP64)
!$  use onemkl_lapack_omp_offload_ilp64   ! 64-bit
#else
!$  use onemkl_lapack_omp_offload_lp64    ! 32-bit
#endif

    implicit none

    integer              :: n = 64, batch_size = 4096, nrhs = 1, cycles = 5
    integer              :: lda, stride_a, stride_ipiv
    integer              :: ldb, stride_b
    integer, allocatable :: ipiv(:,:), info(:)

#if defined(SP)
    real (kind=4), allocatable :: a(:,:), b(:,:), a_orig(:,:), b_orig(:,:), x(:)
    real (kind=4)              :: residual, threshold = 1.0e-5
#else
    real (kind=8), allocatable :: a(:,:), b(:,:), a_orig(:,:), b_orig(:,:), x(:)
    real (kind=8)              :: residual, threshold = 1.0d-9
#endif

    integer (kind=8) :: start_time, end_time, clock_precision
    real    (kind=8) :: cycle_time, total_time = 0.0d0

    integer               :: i, j, c, allocstat, stat
    character (len = 132) :: allocmsg
    character (len =  32) :: arg1, arg2

    ! Simple command-line parser with no error checks
    do i = 1, command_argument_count(), 2
        call get_command_argument(i, arg1)
        select case (arg1)
            case ('-n')
                call get_command_argument(i+1, arg2)
                read(arg2, *, iostat=stat) n
            case ('-b')
                call get_command_argument(i+1, arg2)
                read(arg2, *, iostat=stat) batch_size
            case ('-r')
                call get_command_argument(i+1, arg2)
                read(arg2, *, iostat=stat) nrhs
            case ('-c')
                call get_command_argument(i+1, arg2)
                read(arg2, *, iostat=stat) cycles
            case default
                print *, 'Unrecognized command-line option:', arg1
                stop
        end select
    enddo
    print *, 'Matrix dimensions:', n
    print *, 'Batch size:', batch_size
    print *, 'Number of RHS:', nrhs
    print *, 'Number of test cycles:', cycles

    lda         = n
    stride_a    = n * lda
    stride_ipiv = n
    ldb         = n
    stride_b    = n * nrhs

    ! Allocate memory for linear algebra computations
    allocate (a(stride_a, batch_size), b(n, batch_size*nrhs),  &
              ipiv(stride_ipiv, batch_size), info(batch_size), &
              stat = allocstat, errmsg = allocmsg)
    if (allocstat > 0) stop trim(allocmsg)

    ! Allocate memory for error checking
    allocate (a_orig(stride_a, batch_size), b_orig(n, batch_size*nrhs), x(n), &
              stat = allocstat, errmsg = allocmsg)
    if (allocstat > 0) stop trim(allocmsg)

    call system_clock(count_rate = clock_precision)
    call random_seed()

    do c = 1, cycles
        ! Initialize the matrices with a random number in the interval (-0.5, 0.5)
        call random_number(a)
        a = 0.5 - a
        ! Make diagonal band values larger to ensure well-conditioned matrices
        do i = 1, n
            a(i+(i-1)*lda,:) = a(i+(i-1)*lda,:) + 50.0
            if (i .ne. 1) a(i+(i-1)*lda-1,:) = a(i+(i-1)*lda-1,:) + 20.0
            if (i .ne. n) a(i+(i-1)*lda+1,:) = a(i+(i-1)*lda+1,:) + 20.0
        enddo

        ! Initialize the RHS with a random number in the interval (-2.5, 2.5)
        call random_number(b)
        b = 2.5 - (5.0 * b)
        a_orig = a
        b_orig = b

        call system_clock(start_time)   ! Start timer

        ! Compute the LU factorizations and solve the linear systems using OpenMP offload.
        ! On entry, "a" contains the input matrices. On exit, it contains the factored matrices.
        !$omp target data map(tofrom:a) map(from:ipiv) map(from:info)
            !$omp dispatch
#if defined(SP)
            call sgetrf_batch_strided(n, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, info)
#else
            call dgetrf_batch_strided(n, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, info)
#endif
        !$omp end target data

        if (any(info .ne. 0)) then
            print *, 'Error: getrf_batch_strided returned with errors.'
            stop
        else
            ! Solving the linear systems. On exit, the solutions are stored in b.
            !$omp target data map(to:a) map(to:ipiv) map(tofrom: b) map(from:info)
                !$omp dispatch
#if defined(SP)
                call sgetrs_batch_strided('N', n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, &
                                                        b, ldb, stride_b, batch_size, info)
#else
                call dgetrs_batch_strided('N', n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, &
                                                        b, ldb, stride_b, batch_size, info)
#endif
            !$omp end target data

            call system_clock(end_time)   ! Stop timer

            if (any(info .ne. 0)) then
                print *, 'Error: getrs_batch_strided returned with errors.'
                stop
            else

                ! Compute a_orig*b and compare result to saved RHS
                do i = 1, batch_size
                    do j = 1, nrhs
                        x = 0.0
#if defined(SP)
                        call sgemv('N', n, n, 1.0, a_orig(:,i), lda, b(:,(i-1)*nrhs+j), 1, 0.0, x, 1)
#else
                        call dgemv('N', n, n, 1.0d0, a_orig(:,i), lda, b(:,(i-1)*nrhs+j), 1, 0.0d0, x, 1)
#endif

                        ! Check relative residual
                        residual = norm2(b_orig(:,(i-1)*nrhs+j) - x(:)) / norm2(b_orig(:,(i-1)*nrhs+j))
                        if (residual > threshold) then
                            print *, 'Warning: relative residual of ', residual
                        endif
                    enddo
                enddo

                cycle_time = dble(end_time - start_time) / dble(clock_precision)
                print *, 'Computation completed successfully', cycle_time, 'seconds'
                total_time = total_time + cycle_time
            endif
        endif
    enddo

    print *, 'Total time:', total_time, 'seconds'

    ! Clean up
    deallocate (a, b, a_orig, b_orig, x, ipiv, info)
end program solve_batched_linear_systems
