    !=============================================================
    ! Copyright © 2022 Intel Corporation
    !
    ! SPDX-License-Identifier: MIT
    !=============================================================
    program main
    use iso_fortran_env
    use omp_lib
    implicit none

    integer, parameter :: iterations=100
    integer, parameter :: length=64*1024*1024
    real(kind=REAL64), parameter ::  epsilon=1.D-8
    real(kind=REAL64), allocatable ::  A(:)
    real(kind=REAL64), allocatable ::  B(:)
    real(kind=REAL64), allocatable ::  C(:)
    real(kind=REAL64) :: scalar=3.0
    real(kind=REAL64) :: ar, br, cr, asum
    real(kind=REAL64) :: nstream_time, avgtime
    integer :: err, i, iter

    !
    ! Allocate arrays on the host using plain allocate

    allocate( A(length), stat=err )
    if (err .ne. 0) then
      print *, "Allocation of A returned ", err
      stop 1
    endif

    allocate( B(length), stat=err )
    if (err .ne. 0) then
      print *, "Allocation of B returned ", err
      stop 1
    endif

    allocate( C(length), stat=err )
    if (err .ne. 0) then
      print *, "Allocation of C returned ", err
      stop 1
    endif

    !
    ! Initialize the arrays

    !$omp parallel do
    do i = 1, length
       A(i) = 2.0
       B(i) = 2.0
       C(i) = 0.0
    end do

    !
    ! Perform the computation

    nstream_time = omp_get_wtime()
    !$omp target data  map(to: A, B) map(tofrom: C)

    do iter = 1, iterations
       !$omp target teams distribute parallel do
       do i = 1, length
          C(i) = C(i) + A(i) + scalar * B(i)
       end do
    end do

    !$omp end target data
    nstream_time = omp_get_wtime() - nstream_time

    !
    ! Validate and output results

    ar = 2.0
    br = 2.0
    cr = 0.0
    do iter = 1, iterations
       do i = 1, length
          cr = cr + ar + scalar * br
       end do
    end do

    asum = 0.0
    !$omp parallel do reduction(+:asum)
    do i = 1, length
       asum = asum + abs(C(i))
    end do

    if (abs(cr - asum)/asum > epsilon) then
       write(*,110) "Failed Validation on output array: Expected =", cr, ", Observed =", asum
    else
       avgtime = nstream_time/iterations
       write(*,120) "Solution validates: Checksum =", asum, ", Avg time (s) =",  avgtime
    endif

110 format (A, F20.6, A, F20.6)
120 format (A, F20.6, A, F10.6)

    deallocate(A)
    deallocate(B)
    deallocate(C)

    end program main
