!==============================================================
! Copyright © Intel Corporation
!
! SPDX-License-Identifier: MIT
!=============================================================

PROGRAM mandelbrot
    use iso_fortran_env
    use omp_lib
    use pnm
    implicit none

    complex, parameter   :: c_min = (-0.8, -0.2)
    complex, parameter   :: c_max = (-0.7, -0.1)
    integer              :: side = 4000
    real                 :: x_scale
    real                 :: y_scale
    integer, parameter   :: iter_max = 255
    real, parameter      :: threshold = 4.
    integer              :: x_idx, y_idx
    integer              :: iter
    complex              :: c
    complex              :: z_sum
    integer(kind=int8), allocatable :: image(:, :)
    double precision     :: start, end

    call parse_args(side)

    print *, "Thinking hard..."
    allocate(image(side,side))
    start = omp_get_wtime()
    x_scale = (real(c_max) - real(c_min)) / side
    y_scale = (imag(c_max) - imag(c_min)) / side

    !$omp target teams distribute parallel do map(from:image) private(c, z_sum, iter)
    do x_idx = 1, side
        !$omp SIMD private(c, z_sum, iter)
        do y_idx = 1, side
            c = cmplx(x_idx * x_scale, y_idx * y_scale) + c_min
            z_sum = (0.0, 0.0)
            iter = 0
            do while (iter < iter_max .and. abs(z_sum) < threshold)
                z_sum = z_sum * z_sum + c
                iter = iter + 1
            enddo
            image(x_idx, y_idx) = iter
        enddo
    enddo
    end = omp_get_wtime()

    call write_pgm("mandelbrot.pgm", image)

    print "(A,F6.2,A)", "Done in ", end - start, " seconds"

contains
    subroutine parse_args(side)
        integer, intent(inout)  :: side
        integer                 :: nargs
        character(len=32)       :: option_arg

        nargs = command_argument_count()
        if (nargs > 1) then
            print *, "ignoring extra command line arguments"
            call usage()
        end if

        if (nargs > 0) then
            call get_command_argument(1, option_arg)
            read (option_arg, "(I)") side
        end if

    end subroutine parse_args

    subroutine usage()
        character(len=4096)  :: prog

        call get_command_argument(0, prog)
        print *, "usage: ", trim(prog), " [ side ]"
        print *, ""
        print *, "Generate a side x side pixel mandelbrot image"
    end subroutine usage
END PROGRAM mandelbrot

