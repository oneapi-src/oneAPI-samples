!==============================================================
! Copyright © Intel Corporation
!
! SPDX-License-Identifier: MIT
!=============================================================

PROGRAM test_omp_fortran
    use omp_lib
    implicit none

    print *, "Hello, OpenMP C World!"
    !$omp parallel
        print *, "  I am thread", omp_get_thread_num()
    !$omp end parallel

    print *, "All done, bye."
END PROGRAM test_omp_fortran

