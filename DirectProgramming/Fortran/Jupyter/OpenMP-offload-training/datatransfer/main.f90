!==============================================================
! Copyright © 2020 Intel Corporation
!
! SPDX-License-Identifier: MIT
! =============================================================
program main
        use omp_lib
        integer, parameter :: ARRAY_SIZE=256
        integer :: i, correct_count=0;
        logical :: is_cpu = .true.
        real(kind=4) :: x(ARRAY_SIZE), y(ARRAY_SIZE) 
        real(kind=4) :: a = 1.5, tolerance=0.01
        double precision :: tb, te

        !initialize data
        do i=1,ARRAY_SIZE
                x(i) = i;
                y(i) = i;
        end do

        tb = omp_get_wtime() 

        !Perform Saxpy Function
        INCLUDE 'lab/saxpy_func.f90'

        te = omp_get_wtime() 
        print '("Work took ",(f7.5)," seconds.")', te-tb

        if (is_cpu) then
                print *, "CPU"
        else
                print *, "GPU"
        end if

        !Check Results
        do i=1, ARRAY_SIZE
                if ( abs(y(i)-(a*i+i))<tolerance) correct_count=correct_count+1
        end do
        if (correct_count == ARRAY_SIZE) then
                print *, "Test: Passed"
        else
                print *, "Test: Failed"
        end if
end program main
