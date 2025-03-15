! Snippet begin0
program do_concurrent
    use omp_lib
    implicit none

    integer :: i, outer
    integer, dimension(10000) :: x, y, z
    double precision    :: t0, t1, time
    x = 1
    y = 0
    z = 0
   ! Dummy offload to warm up the device
   !$omp target
   !$omp end target

    t0 = omp_get_wtime()
    do outer = 1, 24000
      !call do_work_on_host_updating_x(x,...)
      do concurrent (i = 1:10000)
          y(i) = x(i) + 1
      enddo

      do concurrent (i = 1:10000)
          z(i) = y(i) + 1
      enddo
    !call do_work_on_host_using_z(z,...)
    enddo
    t1 = omp_get_wtime()
    time = t1-t0
    print *, time
end program do_concurrent
! Snippet end
