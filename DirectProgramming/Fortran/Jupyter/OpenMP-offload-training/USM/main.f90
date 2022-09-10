!==============================================================
! Copyright © 2020 Intel Corporation
!
! SPDX-License-Identifier: MIT
! =============================================================
program main
    use omp_lib
    integer, parameter :: N=16
    integer :: i
    integer, allocatable :: x(:)
    logical :: is_cpu = .true.

    !$omp allocate allocator(omp_target_shared_mem_alloc)
    allocate(x(N)) 
    
    do i=1,N
        x(i) = i
    end do  

    !$omp target map(tofrom: is_cpu) has_device_addr(x)
    !$omp teams distribute parallel do
    do i=1,N
        if ((i==1) .and. (.not.(omp_is_initial_device()))) is_cpu=.false.
        x(i) = x(i) * 2
    end do
    !$omp end target
        
    if (is_cpu) then
        print *, "Running on CPU"
    else
        print *, "Running on GPU"
    end if
        
    do i=1,N
        print *, x(i)
    end do
    
    deallocate(x)
end program main
