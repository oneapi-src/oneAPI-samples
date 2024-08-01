  !=============================================================
  ! Copyright © 2022 Intel Corporation
  !
  ! SPDX-License-Identifier: MIT
  !=============================================================
program target_use_device_addr

  use omp_lib
  use iso_fortran_env, only : real64
  implicit none

  integer, parameter :: N1 = 1024
  real(kind=real64), parameter :: aval = real(42, real64)
  real(kind=real64), allocatable :: array_d(:), array_h(:)
  integer :: i,err

  ! Allocate host data
  allocate(array_h(N1))

  !$omp target data map (from:array_h(1:N1)) map(alloc:array_d(1:N1))
  !$omp target data use_device_addr(array_d)
  !$omp target
      do i=1, N1
        array_d(i) = aval
        array_h(i) = array_d(i)
     end do
  !$omp end target
  !$omp end target data
  !$omp end target data

  ! Check result
  write (*,*) array_h(1), array_h(N1)
  if (any(array_h /= aval)) then
    err = 1
  else
    err = 0
  end if

  deallocate(array_h)
  if (err == 1) then
    stop 1
  else
    stop 0
  end if

end program target_use_device_addr
