program matrix_multiply
   implicit none
   integer :: i, j, k, myid, m, n
   real, allocatable, dimension(:,:) :: a, b, c, c_serial

   n = 2600
   allocate( a(n,n), b(n,n), c(n,n), c_serial(n,n))

   print *, 'matrix size ', n
  
! Initialize matrices
   do j=1,n
     do i=1,n
       a(i,j) = i + j - 1
       b(i,j) = i - j + 1
     enddo
   enddo
   c = 0.0
   c_serial = 0.0

! parallel compute matrix multiplication.
   do j=1,n
     do i=1,n
       do k=1,n
         c(i,j) = c(i,j) + a(i,k) * b(k,j)
       enddo
    enddo
   enddo

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
         print *,'FAILED'
         exit
       endif
     enddo
   enddo

   print *,'PASSED'

end program matrix_multiply

