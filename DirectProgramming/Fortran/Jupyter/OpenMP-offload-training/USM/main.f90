!==============================================================
! Copyright © 2020 Intel Corporation
!
! SPDX-License-Identifier: MIT
! =============================================================
program main
    use omp_lib
    implicit none
    integer, parameter :: N=16
    integer :: correct_count=0
    integer :: i
    integer, allocatable :: x(:), y(:)
    double precision :: te, tb

    !$omp allocate allocator(omp_target_shared_mem_alloc)
    allocate(x(N),y(N)) 
    
    print *,'Number of OpenMP Devices ',omp_get_num_devices()
    
    tb = omp_get_wtime()
    
    do i=1,N
        x(i) = 1
    end do
    
    do i=1,N
        y(i) = 1
    end do

    !$omp target map(tofrom: is_cpu) has_device_addr(x)
    !$omp target map(tofrom: is_cpu) has_device_addr(y)
    !$omp teams distribute parallel do
    do i=1,N
        x(i) = x(i) + y(i)
    end do
    !$omp end target
    
    do i=1,N
        y(i) = 2
    end do
        
    !$omp target map(tofrom: is_cpu) has_device_addr(y)
    !$omp teams distribute parallel do
    do i=1,N
        x(i) = x(i) + y(i)
    end do
    !$omp end target

    te = omp_get_wtime()
    print *,'Time of kernel ',te-tb,' seconds'
    
    do i=1,N
        if (x(i)==4) then
              correct_count = correct_count + 1
        end if
    end do
    
    if (correct_count==N) then
         print *, 'Test: PASSED'
    else
          print *, 'Test: Failed'
    endif
    
    deallocate(x,y)
end program main
