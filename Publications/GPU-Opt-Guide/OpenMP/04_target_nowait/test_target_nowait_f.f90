      !=============================================================
      ! Copyright © 2022 Intel Corporation
      !
      ! SPDX-License-Identifier: MIT
      !=============================================================
      !
      ! This test is from OpenMP API 5.0.1 Examples (June 2020)
      ! https://www.openmp.org/wp-content/uploads/openmp-examples-5-0-1.pdf
      !(4.13.2 nowait Clause on target Construct)
      !

      subroutine init(n, v1, v2)
      integer :: i, n
      real :: v1(n), v2(n)

      do i = 1, n
         v1(i) = i * 0.25
         v2(i) = i - 1.25
      end do
      end subroutine init

      program test_target_nowait
      use omp_lib
      use iso_fortran_env
      implicit none

      integer, parameter :: NUM=100000 ! NUM must be even
      real :: v1(NUM), v2(NUM), vxv(NUM)
      integer :: n, i
      real(kind=REAL64) :: start, end

      n = NUM
      call init(n, v1, v2)

      ! Dummy parallel and target (nowait) regions, so as not to measure
      ! startup time.
      !$omp parallel
        !$omp master
          !$omp target nowait
          !$omp end target
        !$omp end master
      !$omp end parallel

      start=omp_get_wtime()

      !$omp parallel

        !$omp master
          !$omp target teams distribute parallel do nowait &
          !$omp& map(to: v1(1:n/2)) &
          !$omp& map(to: v2(1:n/2)) &
          !$omp& map(from: vxv(1:n/2))
          do i = 1, n/2
             vxv(i) = v1(i)*v2(i)
          end do
        !$omp end master

        !$omp do
        do i = n/2+1, n
           vxv(i) = v1(i)*v2(i)
        end do

      !$omp end parallel

      end=omp_get_wtime()

      write(*,110) "vxv(1)=", vxv(1), ", vxv(n-1)=", vxv(n-1), ", time=", end-start
110   format (A, F10.6, A, F17.6, A, F10.6)

      end program test_target_nowait
