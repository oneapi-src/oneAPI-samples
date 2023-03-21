#define CACHE_CLEAN_SIZE 100000000
#define ITERATIONS 100
#define ARRAYLEN1 4096
#define ARRAYLEN2 32768
c snippet-begin
#define WORKGROUP_SIZE 1024
#define PREFETCH_HINT 4     ! 4 = prefetch to L1 and L3;  2 = prefetch to L3
#define TILE_SIZE 64

      subroutine nbody_1d_gpu(c, a, b, n1, n2)
      implicit none
      integer n1, n2
      real a(0:n1-1), b(0:n2-1), c(0:n1-1)
      real dx, bb(0:TILE_SIZE-1), delta, r2, s0, s1, f
      integer i,j,u,next
      real ma0, ma1, ma2, ma3, ma4, ma5, eps
      parameter (ma0=0.269327, ma1=-0.0750978, ma2=0.0114808)
      parameter (ma3=-0.00109313, ma4=0.0000605491, ma5=-0.00000147177)
      parameter (eps=0.01)

!$omp target teams distribute parallel do thread_limit(WORKGROUP_SIZE)
!$omp& private(i,dx,j,u,bb,next,delta,r2,s0,s1,f)
      do i = 0, n1-1
        dx = 0.0
        do j = 0, n2-1, TILE_SIZE
          ! load tile from b
          do u = 0, TILE_SIZE-1
            bb(u) = b(j+u)
#ifdef PREFETCH
            next = j + TILE_SIZE + u
            if (mod(next,16).eq.0) then
!$omp prefetch data(PREFETCH_HINT:b(next:next))if(next<n2)
            endif
#endif
          enddo
          ! compute
          !DIR$ unroll(TILE_SIZE)
          do u = 0, TILE_SIZE-1
            delta = bb(u) - a(i)
            r2 = delta*delta
            s0 = r2 + eps
            s1 = 1.0 / sqrt(s0)
            f = (s1*s1*s1)-(ma0+r2*(ma1+r2*(ma2+r2*(ma3+r2*(ma4+ma5)))))
            dx = dx + f*delta
          enddo
        enddo
        c(i) = dx*0.23
      enddo
      end subroutine
c snippet-end

      subroutine nbody_1d_cpu(c, a, b, n1, n2)
      implicit none
      integer n1, n2
      real a(0:n1), b(0:n2), c(0:n1)
      real dx, bb(0:TILE_SIZE), delta, r2, s0, s1, f
      integer i,j
      real ma0, ma1, ma2, ma3, ma4, ma5, eps
      parameter (ma0=0.269327, ma1=-0.0750978, ma2=0.0114808)
      parameter (ma3=-0.00109313, ma4=0.0000605491, ma5=-0.00000147177)
      parameter (eps=0.01)
      do i = 0, n1-1
        dx = 0.0
        do j = 0, n2-1
          ! compute
          delta = b(j) - a(i)
          r2 = delta*delta
          s0 = r2 + eps
          s1 = 1.0 / sqrt(s0)
          f = (s1*s1*s1)-(ma0+r2*(ma1+r2*(ma2+r2*(ma3+r2*(ma4+ma5)))))
          dx = dx + f*delta
        enddo
        c(i) = dx*0.23
      enddo
      end subroutine


      subroutine clean_cache_gpu(d,n)
      implicit none
      real d(1)
      integer n, i
!$omp target teams distribute parallel do thread_limit(1024)
      do i=1,n
         d(i) = i
      end do
!$omp end target teams distribute parallel do
      end subroutine

      program nbody
      implicit none
      include 'omp_lib.h'

      real, dimension(:), allocatable :: a,b,c,d
      double precision sum1
      real dx,dummy
      integer i
      integer N1, N2
      double precision t1, t2, elapsed_s
      parameter (N1=ARRAYLEN1, N2=ARRAYLEN2)

      allocate(a(N1),b(N2),c(N1))
      allocate(d(CACHE_CLEAN_SIZE))

      !initialize
      dx = 1.0/N2
      b(1) = 0.0
      do i = 2, N2
        b(i) = b(i-1) + dx
      enddo
      do i = 1, N1
        a(i) = b(i)
        c(i) = 0.0
      enddo

      call omp_set_default_device(0)

!$omp target
      dummy = 0.0
!$omp end target

      elapsed_s = 0.0
!$omp target enter data map(alloc:a(1:N1),b(1:N2),c(1:N1))
!$omp target enter data map(alloc:d(1:CACHE_CLEAN_SIZE))
!$omp target update to(a(1:N1),b(1:N2))
      do i = 1, ITERATIONS
         call clean_cache_gpu(d,CACHE_CLEAN_SIZE)
         t1 = omp_get_wtime()
         call nbody_1d_gpu(c,a,b,N1,N2)
         t2 = omp_get_wtime()
         elapsed_s = elapsed_s + (t2 - t1)
      enddo
!$omp target update from(c(1:N1))

      sum1=0.0
      do i = 1,N1
        sum1 = sum1 + c(i)
      enddo
      print '(/,4x,"Obtained output = ",f15.3)',sum1

      do i = 1,N1
        c(i) = 0.0
      enddo
      call nbody_1d_cpu(c,a,b,N1,N2)

      sum1=0.0
      do i = 1,N1
        sum1 = sum1 + c(i)
      enddo
      print '(/,4x,"Expected output = ",f15.3)',sum1

      print '(//,4x,"Total time = ",f8.1," milliseconds")',
     +  elapsed_s*1000

!$omp target exit data map(delete:a(1:N1),b(1:N2),c(1:N1))
!$omp target exit data map(delete:d(1:CACHE_CLEAN_SIZE))

      deallocate(a,b,c,d)

      end program nbody
