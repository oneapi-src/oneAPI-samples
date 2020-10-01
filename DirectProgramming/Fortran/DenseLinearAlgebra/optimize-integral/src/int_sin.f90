 ! ==============================================================
 ! Copyright © 2020 Intel Corporation
 !
 ! SPDX-License-Identifier: MIT
 ! =============================================================
 !
 ! [DESCRIPTION]
 ! This program computes the integral (area under the curve) of a user-supplied
 ! function over an interval in a stepwise fashion. The interval is split into 
 ! segments, and at each segment position the area of a rectangle is computed 
 ! whose height is the value of sine at that point and the width is the segment
 ! width.  The areas of the rectangles are then summed.
 !
 ! The process is repeated with smaller and smaller width rectangles, more 
 ! closely approximating the true value.
 !
 ! The source for this program also demonstrates recommended Fortran
 ! coding practices.
 !
 ! Compile the sample several times using different optimization options.
 !
 ! Read the Intel(R) Fortran Compiler Documentation for more information about these options.
 ! 
 ! Some of these automatic optimizations use features and options
 ! that can restrict program execution to specific architectures. 
 ! 
 ! [COMPILE] 
 ! Use the one of the following compiler options: 
 !
 ! Windows*: /O1, /O2, /O3
 !
 ! Linux* and macOS*: -O1, -O2, -O3
 !

program int_sin
implicit none

! Create a value DP that is the "kind" number of a double precision value
! We will use this value in our declarations and constants.
integer, parameter :: DP = kind(0.0D0)

! Declare a named constant for pi, specifying the kind type
real(DP), parameter :: pi = 3.141592653589793238_DP 

! Declare interval begin and end
real(DP), parameter :: interval_begin = 0.0_DP
real(DP), parameter :: interval_end   = 2.0_DP * pi

real(DP) :: step, sum, x_i
integer :: N, i, j
real clock_start, clock_finish

write (*,'(A)') "  "
write (*,'(A)') "    Number of    | Computed Integral |"
write (*,'(A)') " Interior Points |                   |"
call cpu_time (clock_start)

do j=2,26
  write (*,'(A)') "--------------------------------------"
  N = 2**j
  ! Compute stepsize for N-1 internal rectangles 
  step = (interval_end - interval_begin) / real(N,DP);
  
  ! Approximate 1/2 area in first rectangle: f(x0) * (step/2) 
  sum = INTEG_FUNC(interval_begin) * (step / 2.0_DP)
  
  do i=1,N-1
    x_i = real(i,DP) * step
    ! Apply midpoint rule:
    ! Given length = f(x), compute the area of the
    ! rectangle of width step
    sum = sum + (INTEG_FUNC(x_i) * step)
    end do
    
  ! Add approximate area in last rectangle for f(xN) * (step/2) 
  sum = sum + (INTEG_FUNC(interval_end) * (step / 2.0_DP))
  
  write (*,'(T5,I10,T18,"|",2X,1P,E14.7,T38,"|")') N, sum
  end do

call cpu_time(clock_finish)
write (*,'(A)') "--------------------------------------"
write (*,'(A)') "  "
write (*,*) "CPU Time = ",(clock_finish - clock_start), " seconds"

contains

! Function to integrate
real(DP) function INTEG_FUNC (x)
  real(DP), intent(IN) :: x

  INTEG_FUNC = abs(sin(x))
  return
end function INTEG_FUNC

end program int_sin
