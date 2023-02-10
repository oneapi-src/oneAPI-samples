        !=============================================================
        ! Copyright © 2022 Intel Corporation
        !
        ! SPDX-License-Identifier: MIT
        !=============================================================
      include "mkl_omp_offload.f90"

      subroutine omp_fbody(f1n,f2n,eorb,                          &
                           ncor,nocc,nvir, emp4,emp5,a,i,j,k,klo, &
                           Jia, Kia, Tia, Xia, Tkj, Kkj,          &
                           t1v1,t1v2)

        use omp_lib
        use onemkl_blas_omp_offload_lp64
        use iso_fortran_env
        implicit none

        real, intent(inout) :: emp4,emp5
        integer, intent(in) :: ncor,nocc,nvir
        integer, intent(in) :: a,i,j,k, klo
        real, intent(inout) :: f1n(nvir,nvir)
        real, intent(inout) :: f2n(nvir,nvir)
        real, intent(in)    :: eorb(*)
        real, intent(in)    :: Jia(*), Kia(*), Tia(*), Xia(*)
        real, intent(in)    :: Tkj(*), Kkj(*)
        real, intent(in)    :: t1v1(nvir),t1v2(nvir)
        real    :: emp4i,emp5i
        real    :: eaijk,denom
        integer :: lnov,lnvv
        integer :: b,c
        real    :: f1nbc,f1ncb,f2nbc,f2ncb
        real    :: t1v1b,t1v2b

        lnov=nocc*nvir
        lnvv=nvir*nvir
        emp4i = 0.0
        emp5i = 0.0

        !$omp dispatch
        call sgemm('n','t',nvir,nvir,nvir,1.0,Jia,nvir,       &
                   Tkj(1+(k-klo)*lnvv),nvir,1.0,f1n,nvir)

        !$omp dispatch
        call sgemm('n','t',nvir,nvir,nvir,1.0,Kia,nvir,       &
                   Tkj(1+(k-klo)*lnvv),nvir,1.0, f2n,nvir)

        !$omp dispatch
        call sgemm('n','n',nvir,nvir,nocc,-1.0,Tia,nvir,      &
                   Kkj(1+(k-klo)*lnov),nocc,1.0, f1n,nvir)

        !$omp dispatch
        call sgemm('n','n',nvir,nvir,nocc,-1.0,Xia,nvir,      &
                   Kkj(1+(k-klo)*lnov),nocc,1.0, f2n,nvir)

        eaijk = eorb(a) - ( eorb(ncor+i)+eorb(ncor+j)+eorb(ncor+k) )

        !$omp target teams distribute parallel do collapse(2)       &
        !$omp     reduction(+:emp4i,emp5i)                          &
        !$omp     private(f1nbc,f1ncb,f2nbc,f2ncb)                  &
        !$omp     private(t1v1b,t1v2b)                              &
        !$omp     private(denom) firstprivate(eaijk,nvir,ncor,nocc)
        do b=1,nvir
           do c=1,nvir
              denom=-1.0/( eorb(ncor+nocc+b)+eorb(ncor+nocc+c)+eaijk )

              f1nbc = f1n(b,c);
              f1ncb = f1n(c,b);
              f2nbc = f2n(b,c);
              f2ncb = f2n(c,b);
              t1v1b = t1v1(b);
              t1v2b = t1v2(b);

              emp4i = emp4i + (denom*t1v1b*f1nbc) + (denom*2*f1ncb)
              emp5i = emp5i + (denom*t1v2b*f2nbc) + (denom*3*f2ncb)
          enddo
        enddo
        !$omp end target teams distribute parallel do

        emp4 = emp4 + emp4i
        emp5 = emp5 + emp5i

        end


      subroutine init_array_1(arr, m)
        implicit none
        real, intent(inout) :: arr(m)
        integer m, i

        do i = 1, m
           arr(i) = 1.0/(100.0 + i-1)
        end do
        end subroutine init_array_1


      subroutine init_array_2(arr, m, n)
        implicit none
        real, intent(inout) :: arr(m, n)
        integer m, n, i, j

        !$omp target teams distribute parallel do
        do i = 1, m
           do j = 1, n
              arr(i,j) = 1.0/(100.0 + ((i-1) * n) + j)
           end do
        end do
      end subroutine init_array_2


      program main

        use omp_lib
        use iso_fortran_env
        implicit none

        interface
           subroutine omp_fbody(f1n,f2n,eorb,                  &
                        ncor,nocc,nvir, emp4,emp5,a,i,j,k,klo, &
                        Jia, Kia, Tia, Xia, Tkj, Kkj,          &
                        t1v1,t1v2)
           real, intent(inout) :: emp4,emp5
           integer, intent(in) :: ncor,nocc,nvir
           integer, intent(in) :: a,i,j,k, klo
           real, intent(inout) :: f1n(nvir,nvir)
           real, intent(inout) :: f2n(nvir,nvir)
           real, intent(in)    :: eorb(*)
           real, intent(in)    :: Jia(*), Kia(*), Tia(*), Xia(*), Tkj(*), Kkj(*)
           real, intent(in)    :: t1v1(nvir),t1v2(nvir)
           end subroutine omp_fbody
        end interface

        integer :: ncor, nocc, nvir, maxiter, nkpass
        integer :: nbf, lnvv, lnov, kchunk
        real, allocatable :: eorb(:)
        real, allocatable :: f1n(:,:)
        real, allocatable :: f2n(:,:)

        real, allocatable :: Jia(:)
        real, allocatable :: Kia(:)
        real, allocatable :: Tia(:)
        real, allocatable :: Xia(:)
        real, allocatable :: Tkj(:)
        real, allocatable :: Kkj(:)

        real, allocatable :: t1v1(:),t1v2(:)
        real emp4, emp5
        integer :: a, b, c, i, j, k
        integer :: klo, khi, iter
        double precision, allocatable :: timers(:)
        double precision :: t0, t1, tsum, tmax, tmin, tavg

!       Run parameters
        nocc = 256
        nvir = 2048
        maxiter = 50
        nkpass = 1
        ncor = 0

        print *, "Run parameters:"
        print *, "nocc    =", nocc
        print *, "nvir    =", nvir
        print *, "maxiter =", maxiter
        print *, "nkpass  =", nkpass
        print *, "ncor    =", ncor
        print *, " "

!       Allocate and initialize arrays.

        nbf = ncor + nocc + nvir
        lnvv = nvir * nvir
        lnov = nocc * nvir
        kchunk = (nocc - 1)/nkpass + 1

        !$omp allocate allocator(omp_target_device_mem_alloc)
        allocate( f1n(1:nvir,1:nvir) )

        !$omp allocate allocator(omp_target_device_mem_alloc)
        allocate( f2n(1:nvir,1:nvir) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( eorb(1:nbf) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( Jia(1:lnvv) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( Kia(1:lnvv) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( Tia(1:lnov*nocc) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( Xia(1:lnov*nocc))

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( Tkj(1:kchunk*lnvv) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( Kkj(1:kchunk*lnvv) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( t1v1(1:lnvv) )

        !$omp allocate allocator(omp_target_host_mem_alloc)
        allocate( t1v2(1:lnvv) )
!
        call init_array_1(eorb, nbf)
        call init_array_1(Jia, lnvv)
        call init_array_1(Kia, lnvv)
        call init_array_1(Tia, lnov*nocc)
        call init_array_1(Xia, lnov*nocc)
        call init_array_1(Tkj, kchunk*lnvv)
        call init_array_1(Kkj, kchunk*lnov)
        call init_array_1(t1v1, lnvv)
        call init_array_1(t1v2, lnvv)

        call init_array_2(f1n, nvir, nvir)
        call init_array_2(f2n, nvir, nvir)

        print *, "End of initialization"

        allocate (timers(1:maxiter))

        emp4=0.0
        emp5=0.0
        a=1
        iter=1

        !$omp target data                   &
           map(to: eorb)                    &
           map(to: Jia, Kia, Tia, Xia)      &
           map(to: Tkj, Kkj)                &
           map(to: t1v1, t1v2)

        do klo = 1, nocc, kchunk
           khi = MIN(nocc, klo+kchunk-1)
           do j = 1, nocc

#if defined(DO_UPDATE_ARRAYS)
!             Update elements of Tkj and KKj.
              Tkj((khi-klo+1)*lnvv) = Tkj((khi-klo+1)*lnvv) + 1.0
              Kkj((khi-klo+1)*lnov) = Kkj((khi-klo+1)*lnov) + 1.0

              !$omp target update to (Tkj, Kkj)
#endif

              do i = 1, nocc

#if defined(DO_UPDATE_ARRAYS)
!                Update elements of Jia, Kia, Tia, Xia arrays.
                 Jia(lnvv) = Jia(lnvv) + 1.0
                 Kia(lnvv) = Kia(lnvv) + 1.0
                 Tia(lnov) = Tia(lnov) + 1.0
                 Xia(lnov) = Xia(lnov) + 1.0

                 !$omp target update to (Jia, Kia, Tia, Xia)
#endif

                 do k = klo, MIN(khi,i)

#if defined(DO_UPDATE_ARRAYS)
!                   Update elements of t1v1 array.
                    t1v1(:) = Tkj(lnvv-nvir+1:lnvv)

                    !$omp target update to (t1v1)
#endif

                    t0 = omp_get_wtime()

                    call omp_fbody(f1n,f2n,eorb,                     &
                              ncor,nocc,nvir, emp4,emp5,a,i,j,k,klo, &
                              Jia, Kia, Tia, Xia, Tkj, Kkj,          &
                              t1v1,t1v2)

                    t1 = omp_get_wtime()
                    timers(iter) = (t1-t0)
                    if (iter .eq. maxiter) then
                        print *, "Stopping after ", iter, "iterations"
                        print *, " "
                        goto 150
                    endif

!                   Prevent NAN for large maxiter...
                    if (emp4 >  1000.0) then
                        emp4 = emp4 - 1000.0
                    endif
                    if (emp4 < -1000.0) then
                        emp4 = emp4 + 1000.0
                    endif
                    if (emp5 >  1000.0) then
                        emp5 = emp5 - 1000.0
                    endif
                    if (emp5 < -1000.0) then
                        emp5 = emp5 + 1000.0
                     endif

                    iter = iter + 1

                 end do ! k = klo, MIN(khi,i)
              end do ! do i = 1, nocc
           end do ! do j = 1, nocc
        end do ! do klo = 1, nocc, kchunk

 150    CONTINUE
        !$omp end target data

        tsum =  0.0
        tmax = -1.0e10
        tmin =  1.0e10
        do i = 2, iter
           tsum = tsum + timers(i)
           tmax = MAX(tmax,timers(i))
           tmin = MIN(tmin,timers(i))
        end do

        tavg = tsum / (iter - 1)
        print *, "TOTAL ITER: ", iter
        write(*, 110) " TIMING: min=", tmin, ", max=", tmax, ", avg of iters after first=", tavg, " seconds"
 110    format (A, F9.6, A, F9.6, A, F9.6, A)

        write(*, 120) " emp4 = ", emp4, " emp5 =", emp5
 120    format (A, F15.3, A, F15.3)

        print *, "END"

        deallocate (f1n)
        deallocate (f2n)
        deallocate (eorb)
        deallocate (Jia)
        deallocate (Kia)
        deallocate (Tia)
        deallocate (Xia)
        deallocate (Tkj)
        deallocate (Kkj)

        deallocate (t1v1)
        deallocate (t1v2)
        deallocate (timers)

        end program
