!==============================================================
! Copyright © 2020 Intel Corporation
!
! SPDX-License-Identifier: MIT
! =============================================================
program main
    use omp_lib
    integer, parameter :: N=16
    integer :: i, x(N)
    logical :: is_cpu = .true.
        
    do i=1,N
        x(i) = i
    end do
       
    !TODO Place the target directive here including the map(from:is_cpu) clause
    !$omp target map(tofrom: is_cpu)
    !$omp parallel do
    do i=1,N
        if ((i==1) .and. (.not.(omp_is_initial_device()))) is_cpu=.false.
        x(i) = x(i) * 2
    end do
    
    !TODO Place the end target directive here
    !$omp end target
        
    if (is_cpu) then
        print *, "Running on CPU"
    else
        print *, "Running on GPU"
    end if
        
    do i=1,N
        print *, x(i)
    end do
end program main
