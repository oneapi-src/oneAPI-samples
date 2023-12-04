  !=============================================================
  ! Copyright © 2022 Intel Corporation
  !
  ! SPDX-License-Identifier: MIT
  !=============================================================
program main
  use iso_fortran_env
  use omp_lib
  implicit none

  integer, parameter :: iterations=1000
  integer, parameter :: length=64*1024*1024
  real(kind=REAL64), parameter ::  epsilon=1.D-8
  real(kind=REAL64), allocatable ::  A(:)
  real(kind=REAL64), allocatable ::  B(:)
  real(kind=REAL64), allocatable ::  C(:)
  real(kind=REAL64) :: scalar=3.0
  real(kind=REAL64) :: ar, br, cr, asum
  real(kind=REAL64) :: nstream_time, avgtime
  integer :: i, iter

  !
  ! Allocate arrays in device memory

  !$omp allocators allocate(allocator(omp_target_device_mem_alloc): A)
  allocate(A(length))

  !$omp allocators allocate(allocator(omp_target_device_mem_alloc): B)
  allocate(B(length))

  !$omp allocators allocate(allocator(omp_target_device_mem_alloc): C)
  allocate(C(length))

  !
  ! Initialize the arrays

  !$omp target teams distribute parallel do has_device_addr(A, B, C)
  do i = 1, length
      A(i) = 2.0
      B(i) = 2.0
      C(i) = 0.0
  end do

  !
  ! Perform the computation

  nstream_time = omp_get_wtime()
  do iter = 1, iterations
     !$omp target teams distribute parallel do has_device_addr(A, B, C)
     do i = 1, length
        C(i) = C(i) + A(i) + scalar * B(i)
     end do
  end do
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
  !$omp target teams distribute parallel do reduction(+:asum) has_device_addr(C)
  do i = 1, length
     asum = asum + abs(C(i))
  end do

  if (abs(cr - asum)/asum > epsilon) then
     print *, "Failed Validation on output array:", "Expected =", cr, "Observed =", asum
  else
     avgtime = nstream_time/iterations
     print *, "Solution validates:", "Checksum =", asum, "Avg time (s) =",  avgtime
  endif

  deallocate(A)
  deallocate(B)
  deallocate(C)

end program main
