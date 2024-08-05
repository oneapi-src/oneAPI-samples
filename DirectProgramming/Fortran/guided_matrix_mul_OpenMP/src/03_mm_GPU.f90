program matrix_multiply
   use omp_lib
   implicit none
   integer :: i, j, k, myid, m, n
   real, allocatable, dimension(:,:) :: a, b, c, c_serial
! 
! Different Intel GPUs have varying amounts of memory. If the program
! fails at runtime, try decreasing the value of "n".
!
   n = 2600

    myid = OMP_GET_THREAD_NUM()
    if (myid .eq. 0) then
      print *, 'matrix size ', n
      print *, 'Number of CPU procs is ', OMP_GET_NUM_THREADS()
      print *, 'Number of OpenMP Device Available:', omp_get_num_devices()
!$omp target 
      if (OMP_IS_INITIAL_DEVICE()) then
        print *, ' Running on CPU'
        else
        print *, ' Running on GPU'
      endif
!$omp end target 
    endif

      allocate( a(n,n), b(n,n), c(n,n), c_serial(n,n))

! Initialize matrices
      do j=1,n
         do i=1,n
            a(i,j) = i + j - 1
            b(i,j) = i - j + 1
         enddo
      enddo
      c = 0.0
      c_serial = 0.0

!$omp target teams map(to: a, b) map(tofrom: c)
!$omp distribute parallel do SIMD private(j, i, k)
! parallel compute matrix multiplication.
      do j=1,n
         do i=1,n
            do k=1,n
                c(i,j) = c(i,j) + a(i,k) * b(k,j)
            enddo
         enddo
      enddo
!$omp end target teams

! serial compute matrix multiplication
      do j=1,n
         do i=1,n
            do k=1,n
                c_serial(i,j) = c_serial(i,j) + a(i,k) * b(k,j)
            enddo
         enddo
      enddo

! verify result
      do j=1,n
         do i=1,n
            if (c_serial(i,j) .ne. c(i,j)) then
               print *,'FAILED, i, j, c_serial(i,j), c(i,j) ', i, j, c_serial(i,j), c(i,j)
            exit
            endif
         enddo
      enddo

      print *,'PASSED'


end program matrix_multiply
