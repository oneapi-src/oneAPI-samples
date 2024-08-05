!===============================================================================
!
! Content:
!     Implement edge detection on simple binary images using a standard Fortran
!     DO CONCURRENT loop. The compiler will offload the loop to a GPU using the
!     OpenMP runtime.
!
! Compile for CPU (sequential):
!     ifx img_seg_do_concurrent.F90 -o img_seg_do_conc_cpu_seq
!
! Compile for CPU (parallel):
!     ifx img_seg_do_concurrent.F90 -o img_seg_do_conc_cpu_par -qopenmp
!
! Compile for GPU using the OpenMP backend:
!     ifx img_seg_do_concurrent.F90 -o img_seg_do_conc_gpu -qopenmp \
!         -fopenmp-targets=spir64 -fopenmp-target-do-concurrent
!
!===============================================================================
program img_seg_do_conc_example
    implicit none

    integer :: n = 8, objects = 3, images = 1
    logical :: display = .false.
    integer :: i, j, img_i, allocstat, stat

    integer, allocatable :: image(:,:)
    logical, allocatable :: edge_mask(:,:)

    character (len = 132) :: allocmsg
    character (len =  32) :: arg1, arg2

    integer (kind=8) :: start_time, end_time, clock_precision
    real    (kind=8) :: cycle_time, total_time = 0.0d0

    call process_command_line()
    call system_clock(count_rate = clock_precision)

    ! Allocate image and edge mask
    allocate (image(n, n), source = 0, stat = allocstat, errmsg = allocmsg)
    if (allocstat > 0) stop trim(allocmsg)

    allocate (edge_mask(n, n), source = .false., stat = allocstat, errmsg = allocmsg)
    if (allocstat > 0) stop trim(allocmsg)

    ! Process images
    do img_i = 1, images
        call initialize_image()
        if (display) call display_image()

        call system_clock(start_time)   ! Start timer

        ! Outline the objects in the binary image
        do concurrent (j = 1:n, i = 1:n, image(i, j) /= 0)
            if (i == 1 .or. i == n .or. &
                j == 1 .or. j == n) then
                edge_mask(i, j) = .true.
            else
                if (any(image(i-1:i+1, j-1:j+1) == 0)) edge_mask(i, j) = .true.
            endif
        enddo

        call system_clock(end_time)   ! Stop timer
        cycle_time = dble(end_time - start_time) / dble(clock_precision)

        if (display) call display_edge_mask()

        print *, 'Image', img_i, 'took', cycle_time, 'seconds'
        if (img_i /= 1) total_time = total_time + cycle_time

        edge_mask = .false.   ! Reset edge mask
    enddo
    print *, 'Total time (not including first iteration):', total_time, 'seconds'

    deallocate(image, edge_mask)

contains
    subroutine initialize_image()
        integer x, x_min, x_max, y, y_min, y_max, d
        real :: rn(3)

        image = 0

        ! Create random regions of interest in the image
        call random_seed()
        do i = 1, objects
            call random_number(rn)
            d = 1 + floor(2 * rn(1))

            x_min = d + 1
            x_max = n - d
            x = x_min + (x_max - x_min) * rn(2)

            y_min = d + 1
            y_max = n - d
            y = y_min + (y_max - y_min) * rn(3)

            image(x-d:x+d, y-d:y+d) = 1
        enddo
    end subroutine initialize_image

    subroutine display_image()
        print *
        print *, 'Binary image:'
        do j = 1, n
            do i = 1, n
                write(6, advance='no', fmt="(i3)") image(i, j)
            enddo
            print *
        enddo
    end subroutine display_image

    subroutine display_edge_mask()
        print *
        print *, 'Edge mask:'
        do j = 1, n
            do i = 1, n
                if (edge_mask(i, j)) then
                    write(6, advance='no', fmt="(l3)") edge_mask(i, j)
                else
                    write(6, advance='no', fmt="(a3)") '-'
                endif
            enddo
            print *
        enddo
    end subroutine display_edge_mask

    subroutine process_command_line()
        j = 1
        do while (j <= command_argument_count())
            call get_command_argument(j, arg1)
            select case (arg1)
                case ('-n')
                    call get_command_argument(j+1, arg2)
                    read(arg2, *, iostat=stat) n
                    j = j + 2
                case ('-o')
                    call get_command_argument(j+1, arg2)
                    read(arg2, *, iostat=stat) objects
                    j = j + 2
                case ('-i')
                    call get_command_argument(j+1, arg2)
                    read(arg2, *, iostat=stat) images
                    j = j + 2
                case ('-d')
                    display = .true.
                    j = j + 1
                case ('-h')
                    call print_help()
                    stop
                case default
                    print *, 'Unrecognized command-line option: ', arg1
                    call print_help()
                    stop
            end select
        enddo
        print *, 'Grid dimensions:', n
        print *, 'Number of images to process:', images
        print *, 'Number of objects in each image:', objects
    end subroutine process_command_line

    subroutine print_help()
        print '(a,/)', 'Command-line options:'
        print '(a)', '   -n #   image dimensions (integer)'
        print '(a)', '   -o #   number of objects in image (integer), objects may overlap'
        print '(a)', '   -i #   number of images to process (integer)'
        print '(a)', '   -d     display image and object edge mask'
    end subroutine print_help
end program img_seg_do_conc_example
