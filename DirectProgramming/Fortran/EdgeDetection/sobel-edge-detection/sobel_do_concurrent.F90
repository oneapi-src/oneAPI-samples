!===============================================================================
!
! Content:
!     Implement Sobel edge detection using Fortran DO CONCURRENT
!
! Compile for CPU (sequential):
!     ifx ppm_image_io.F90 sobel_do_concurrent.F90 -o sobel_do_conc_cpu_seq
!
! Compile for CPU (parallel):
!     ifx ppm_image_io.F90 sobel_do_concurrent.F90 -o sobel_do_conc_cpu_par \
!         -qopenmp
!
! Compile for GPU using the OpenMP backend:
!     ifx ppm_image_io.F90 sobel_do_concurrent.F90 -o sobel_do_conc_gpu \
!         -qopenmp -fopenmp-targets=spir64 -fopenmp-target-do-concurrent
!
! Compile for GPU using the OpenMP backend and AOT compilation for PVC:
!     ifx ppm_image_io.F90 sobel_do_concurrent.F90 -o sobel_do_conc_gpu_aot \
!         -qopenmp -fopenmp-targets=spir64_gen -fopenmp-target-do-concurrent \
!         -Xopenmp-target-backend "-device pvc"
!
! Compile for Nvidia {A|H}100 GPUs:
!     nvfortran ppm_image_io.F90 sobel_do_concurrent.F90 -o sobel_dc_nv_gpu \
!         -fast -stdpar=gpu -target=gpu -gpu=managed -Minfo=all -gpu=cc{80|90}
!
! Compile for multicore CPUs using the Nvidia compiler:
!     nvfortran ppm_image_io.F90 sobel_do_concurrent.F90 -o sobel_dc_nv_cpu \
!         -fast -stdpar=multicore -target=multicore -Minfo=all
!
!===============================================================================
program sobel_do_conc
    use ppm_image_io
    implicit none

    integer              :: c, r, stat
    character(len = 128) :: img_infile = '', img_outfile = ''

    ! Sobel horizontal and vertical filter masks
    integer                 :: max_gradient = 0
    real(kind = 4)          :: sh, sv
    integer, dimension(3,3) :: gh, gv, smooth

    integer (kind=8) :: start_time, end_time, clock_precision
    real    (kind=8) :: total_time

    call process_command_line()
    call system_clock(count_rate = clock_precision)

    stat = open_img_file(img_infile)
    if (stat /= 0) then
        stop 'Error in file open. Stopping.'
    endif
    call read_img_data()
    call close_img_file()

    ! Apply Sobel filter
    gh     = reshape([-1,  0,  1, -2,  0,  2, -1,  0,  1], [3, 3])
    gv     = reshape([-1, -2, -1,  0,  0,  0,  1,  2,  1], [3, 3])
    smooth = reshape([ 1,  2,  1,  2,  4,  2,  1,  2,  1], [3, 3])

    ! Dummy target region to avoid measuring startup time
    !$omp target
    !$omp end target

    call system_clock(start_time)   ! Start timer

    do concurrent (c = 2:img_width - 1, r = 2:img_height - 1)
        image_soa%red(r, c) = sum(image_soa%blue(r-1:r+1, c-1:c+1) * smooth) / 16
    enddo

    do concurrent (c = 2:img_width - 1, r = 2:img_height - 1) reduce(max: max_gradient)
        image_soa%green(r, c) = abs(sum(image_soa%red(r-1:r+1, c-1:c+1) * gh)) + &
                                abs(sum(image_soa%red(r-1:r+1, c-1:c+1) * gv))
        max_gradient = max(max_gradient, image_soa%green(r, c))
    enddo

    do concurrent (c = 1:img_width, r = 1:img_height)
        if (image_soa%green(r, c) >= 0.75 * max_gradient) then
            image_soa%green(r, c) = 0
        elseif (image_soa%green(r, c) >= 0.50 * max_gradient .and. &
                image_soa%green(r, c) <  0.75 * max_gradient) then
            image_soa%green(r, c) = 150
        elseif (image_soa%green(r, c) >= 0.10 * max_gradient .and. &
                image_soa%green(r, c) <  0.50 * max_gradient) then
            image_soa%green(r, c) = 225
        else
            image_soa%green(r, c) = 255
        endif
        image_soa%red(r, c)  = image_soa%green(r, c)
        image_soa%blue(r, c) = image_soa%green(r, c)
    enddo

    call system_clock(end_time)   ! Stop timer
    total_time = dble(end_time - start_time) / dble(clock_precision)
    print *, 'Total time:', total_time, 'seconds'

    call write_img_data(img_outfile)

contains
    subroutine process_command_line()
        integer             :: j
        character(len = 32) :: arg1, arg2

        if (command_argument_count() == 0) then
            call print_help()
            stop
        endif

        j = 1
        do while (j <= command_argument_count())
            call get_command_argument(j, arg1)
            select case (arg1)
                case ('-i')
                    call get_command_argument(j+1, arg2)
                    read(arg2, *, iostat=stat) img_infile
                    j = j + 2
                case ('-o')
                    call get_command_argument(j+1, arg2)
                    read(arg2, *, iostat=stat) img_outfile
                    j = j + 2
                case ('-h')
                    call print_help()
                    stop
                case default
                    print *, 'Unrecognized command-line option: ', arg1
                    call print_help()
                    stop
            end select
        enddo

        if (trim(img_infile) == '' .or. trim(img_outfile) == '') then
            call print_help()
            stop
        else
            print *, 'Name of input image: ', trim(img_infile)
            print *, 'Name of output image: ', trim(img_outfile)
        endif
    end subroutine process_command_line

    subroutine print_help()
        print '(a,/)', 'Command-line options:'
        print '(a)', '   -i <filename>   input PPM image (required)'
        print '(a)', '   -o <filename>   output PPM image (required)'
    end subroutine print_help
end program sobel_do_conc
