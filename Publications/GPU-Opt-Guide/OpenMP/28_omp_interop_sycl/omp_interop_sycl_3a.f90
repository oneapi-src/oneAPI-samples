! Snippet begin0
module subs

  interface

    subroutine foo_gpu(c, v1, n, iop1)  !! variant function
      use iso_c_binding
      integer, intent(in)  :: c
      integer, intent(in)  :: n
      integer, intent(out) :: v1(10)
      type(c_ptr), intent(in):: iop1
    end subroutine foo_gpu

    subroutine foo(c, v1, n)  !! base function
      import foo_gpu          ! Need to add this statement
      integer, intent(in)  :: c
      integer, intent(in)  :: n
      integer, intent(out) :: v1(10)
      !$omp declare variant(foo:foo_gpu) &
      !$omp& match(construct={dispatch}) append_args(interop(targetsync))
    end subroutine foo

  end interface

end module subs
! Snippet end0

! Snippet begin1
program p
  use subs
  use omp_lib
  integer c
  integer v1(10)
  integer i, n, d
  integer (kind=omp_interop_kind) :: iop1

  c = 2
  n = 10
  do i = 1, n
    v1(i) = i
  enddo

  d = omp_get_default_device()

  !$omp interop init(prefer_type(omp_ifr_sycl), targetsync:iop1) device(d)

  !$omp dispatch device(d) interop(iop1)
  call foo(c, v1, n)

  !$omp interop destroy(iop1)

  print *, "v1(1) = ", v1(1), " (2), v1(10) = ", v1(10), " (20)"
end program
! Snippet end1
