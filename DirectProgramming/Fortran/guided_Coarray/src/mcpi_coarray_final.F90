! ==============================================================
! Copyright © 2020 Intel Corporation
!
! SPDX-License-Identifier: MIT
! =============================================================
!
! Part of the Coarray Tutorial. For information, please read
! Tutorial: Using Fortran Coarrays
! Getting Started Tutorials document

program mcpi

! This program demonstrates using Fortran coarrays to implement the classic
! method of computing the mathematical value pi using a Monte Carlo technique.
!
! Compiler options: /Qcoarray
!                   -coarray 
!

implicit none

! Declare kind values for large integers, single and double precision
integer, parameter :: K_BIGINT = selected_int_kind(15)
integer, parameter :: K_DOUBLE = selected_real_kind(15,300)

! Number of trials per image. The bigger this is, the better the result
! This value must be evenly divisible by the number of images.
integer(K_BIGINT), parameter :: num_trials = 600000000_K_BIGINT

! Actual value of PI to 18 digits for comparison
real(K_DOUBLE), parameter :: actual_pi = 3.141592653589793238_K_DOUBLE

! Declare scalar coarray that will exist on each image
 integer(K_BIGINT) :: total[*] ! Per-image subtotal

! Local variables
real(K_DOUBLE) :: x,y
real(K_DOUBLE) :: computed_pi
integer :: i
integer(K_BIGINT) :: bigi
integer(K_BIGINT) :: clock_start,clock_end,clock_rate
integer, allocatable :: seed_array(:)
integer :: seed_size

! Image 1 initialization
if (THIS_IMAGE() == 1) then
    ! Make sure that num_trials is divisible by the number of images
    if (MOD(num_trials,INT(NUM_IMAGES(),K_BIGINT)) /= 0_K_BIGINT) &
        error stop "Number of trials not evenly divisible by number of images!"
    print '(A,I0,A,I0,A)', "Computing pi using ",num_trials," trials across ",NUM_IMAGES()," images"
    call SYSTEM_CLOCK(clock_start)
end if

! Set the initial random number seed to an unpredictable value, with a different
! sequence on each image. 
call RANDOM_INIT(.FALSE.,.TRUE.) 

! Initialize our subtotal
total = 0_K_BIGINT

! Run the trials, with each image doing its share of the trials.
!
! Get a random X and Y and see if the position is within a circle of radius 1. 
! If it is, add one to the subtotal
do bigi=1_K_BIGINT,num_trials/int(NUM_IMAGES(),K_BIGINT)
    call RANDOM_NUMBER(x)
    call RANDOM_NUMBER(y)
    if ((x*x)+(y*y) <= 1.0_K_DOUBLE) total = total + 1_K_BIGINT
end do

! Wait for everyone
sync all

! Image 1 end processing
if (this_image() == 1) then
    ! Sum subtotals of all images
    do i=2,num_images()
        total = total + total[i]
    end do

    ! total/num_trials is an approximation of pi/4
    computed_pi = 4.0_K_DOUBLE*(REAL(total,K_DOUBLE)/REAL(num_trials,K_DOUBLE))
    print '(A,G0.8,A,G0.3)', "Computed value of pi is ", computed_pi, &
        ", Relative Error: ",ABS((computed_pi-actual_pi)/actual_pi)
    ! Show elapsed time
    call SYSTEM_CLOCK(clock_end,clock_rate)
    print '(A,G0.3,A)', "Elapsed time is ", &
        REAL(clock_end-clock_start)/REAL(clock_rate)," seconds"
end if
    
end program mcpi
