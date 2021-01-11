!==============================================================
! Copyright © 2020 Intel Corporation
!
! SPDX-License-Identifier: MIT
! =============================================================

subroutine init (x, n, val)
        integer, intent(in) :: n
        real(kind=4), intent(out), dimension(n) :: x
        real(kind=4), intent(in) :: val
        do i=1,n
                x(i)=val
        end do
end subroutine init

program main
        use omp_lib
        integer, parameter :: ARRAY_SIZE=256
        integer :: i, correct_count=0;
        real(kind=4) :: x(ARRAY_SIZE), y(ARRAY_SIZE) 
        real(kind=4) :: a = 1.5, tolerance=0.01
        double precision :: tb, te

        call init (x, ARRAY_SIZE, 1.0)
        call init (y, ARRAY_SIZE, 1.0)

        print *, "Number of OpenMP Devices: ", omp_get_num_devices()

        tb = omp_get_wtime() 

        ! Written by Jupyter Notebook
        INCLUDE 'lab/target_data_region.f90'

        te = omp_get_wtime() 

        print '("Work took ",(f7.5)," seconds.")', te-tb

        !Check Results
        do i=1, ARRAY_SIZE
                if (x(i)==4.0 ) correct_count=correct_count+1
        end do
        if (correct_count == ARRAY_SIZE) then
                print *, "Test: Passed"
        else
                print *, "Test: Failed"
        end if
end program main
