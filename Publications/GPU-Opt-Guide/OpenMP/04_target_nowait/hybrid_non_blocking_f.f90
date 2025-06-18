program hybrid_non_blocking
  use omp_lib
  implicit none

  integer, parameter :: N_A = 1000000, N_B = 500000, NUM_ITERATIONS = 10000
  real(4), allocatable :: a(:), b(:), res(:)
  real(4) :: sum
  integer :: count, i, j
  real(8) :: start, mid, end

  allocate(a(N_A), b(N_B), res(N_A))
  sum = 0.0
  count = 0

  do i = 1, N_A
    a(i) = i * 0.0001
  end do

  do i = 1, N_B
    b(i) = mod(i, 1000) * 0.001
  end do

  ! Dummy target region to warm up GPU
  !$omp target
  !$omp end target

  call omp_set_default_device(0)
  call cpu_time(start)

  do j = 1, NUM_ITERATIONS
    !$omp target teams distribute parallel do nowait map(to: a(1:N_A)) map(from: res(1:N_A))
    do i = 1, N_A
      res(i) = a(i)
      res(i) = res(i) + sin(res(i)) * exp(res(i))
      res(i) = res(i) * 1.01
    end do

    do i = 1, N_B
      if (b(i) > 0.5) then
        count = count + 1
      end if
      sum = sum + b(i) * 0.1
    end do

    !$omp taskwait
  end do

  call cpu_time(end)

  print *, "Hybrid Non-blocking: ", end - start, " seconds"
  deallocate(a, b, res)

end program hybrid_non_blocking
