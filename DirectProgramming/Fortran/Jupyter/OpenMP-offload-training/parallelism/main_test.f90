!==============================================================
! Copyright © 2020 Intel Corporation
!
! SPDX-License-Identifier: MIT
! =============================================================
program main
        use omp_lib
        integer, parameter :: ARRAY_SIZE=256, NUM_BLOCKS=9
        integer :: i,ib,correct_count=0, num_teams
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
        INCLUDE 'saxpy_func_parallel_solution.f90'

        te = omp_get_wtime() 
        
        print *, "Number of OpenMP Device Available:", omp_get_num_devices()
        if (is_cpu) then
                print *, "Running on CPU"
        else
                print *, "Running on GPU"
        end if
        print '("Work took ",(f7.5)," seconds.")', te-tb
        print *, "Number of Teams Created: ", num_teams
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
