module ppm_image_io
    use iso_fortran_env
    implicit none

    integer :: img_width     ! Number of triplets in a scan line or row 
    integer :: img_height    ! Number of rows or scan lines of pixels

    private
        integer :: img_unit
        integer :: max_range

        character(:), allocatable :: input_file, output_file
        character(len = 2)        :: img_format

        type p3_pixel
            integer :: red
            integer :: green
            integer :: blue
        end type p3_pixel
        type (p3_pixel), dimension(:,:), allocatable :: image_aos

        type image_soa_type
            integer, allocatable :: red(:,:)
            integer, allocatable :: green(:,:)
            integer, allocatable :: blue(:,:)
        end type image_soa_type
        type (image_soa_type) :: image_soa

    public :: open_img_file    ! User method to open an image file
    public :: read_img_data    ! User method to read the image file
    public :: write_img_data   ! User method to write a new image file
    public :: close_img_file
    public :: image_aos
    public :: image_soa
    public :: img_width, img_height

contains
    function open_img_file(filename) result(stat)
        implicit none

        integer      :: stat
        character(*) :: filename
 
        open(newunit = img_unit, file = trim(filename), &
             access = 'sequential', form = 'formatted', &
             status = 'old', iostat = stat)
  
        if (stat /= 0) then
            write(*,*) 'Error in file open, iostat is ', stat
            return
        endif

        input_file = filename   ! Save filename in private module data
    end function open_img_file

    subroutine close_img_file()
        implicit none
        close(unit = img_unit)
    end subroutine close_img_file

    subroutine write_img_data(filename)
        implicit none

        integer      :: stat, r, c
        character(*) :: filename

        open(newunit = img_unit, file = trim(filename), &
             access = 'sequential', form = 'formatted', &
             status = 'unknown', iostat = stat)
  
        if (stat /= 0) then
            write(*,*) 'Error in file open, iostat is ', stat
            return
        endif

        write(img_unit, '(A2)') img_format
        write(img_unit, '(2(i0,1x))') img_width, img_height
        write(img_unit, '(i3)') max_range

        do r = 1, img_height
            write(img_unit, '(15(i0,:,1x))') ((image_soa%red(r, c),   &
                                               image_soa%green(r, c), &
                                               image_soa%blue(r, c)), c = 1, img_width)
        enddo
        close(unit = img_unit)
    end subroutine write_img_data

    subroutine read_img_data
        implicit none

        integer,        parameter :: LINE_MAX_CHARS = 132
        integer,        parameter :: TRIPLET_CHARS = 12
        integer                   :: rows, cols
        integer                   :: triplet, triplets_this_line, chars_read
        character(:), allocatable :: iomess
        character(LINE_MAX_CHARS) :: line
        integer :: errstat

        ! Read format first two characters on first line of file
        read(img_unit, '(A2)') img_format

        ! Read line 2, two integers space separated: WIDTH HEIGHT
        read(img_unit, *) img_width, img_height

        ! Read line 3, one integer which is max pixel component value, typically 255 for 8bit RGB 
        read(img_unit, *) max_range

        !allocate the new image
        call allocate_new_image()
    
        !--------- read in the data to the AOS 'image' array -----------
        do rows = 1, img_height
            triplet = 1
            do while (triplet <= img_width)
                line(1:LINE_MAX_CHARS) = ' '
                read(img_unit, '(A)', iostat = errstat) line
                if (errstat == IOSTAT_END) then
                   write(*,*) "End of file"
                   write(*,*) "row ", rows
                   write(*,*) "triplet ", triplet
                   write(*,*) "img_width", img_width
                   stop
                endif          
                line = ' '//line

                ! Read values off of a line
                errstat = 0
                do while (len(trim(line)) > 0)
                    read(line, *, iostat = errstat) image_aos(rows, triplet)%red, &
                                                    image_aos(rows, triplet)%green, &
                                                    image_aos(rows, triplet)%blue
                    if (errstat == 0) then
                        ! Shift line left removing the three integers just read
                        line = adjustl(line)
                        line = line(scan(line,' '):)   ! Remove first red
                        line = adjustl(line)
                        line = line(scan(line,' '):)   ! Remove green
                        line = adjustl(line)
                        line = line(scan(line,' '):)   ! Remove blue
                        triplet = triplet + 1
                    endif
                enddo
            enddo
        enddo

        !--------- copy the data to the SOA 'image_soa' array -----------
        do rows = 1, img_height
            do cols = 1, img_width
                image_soa%red(rows, cols)   = image_aos(rows, cols)%red
                image_soa%green(rows, cols) = image_aos(rows, cols)%green
                image_soa%blue(rows, cols)  = image_aos(rows, cols)%blue
            enddo
        enddo
    end subroutine read_img_data

    subroutine allocate_new_image()
        implicit none
        integer                   :: errstat
        character(:), allocatable :: errmess

        ! Allocate the AOS image array
        if (allocated(image_aos)) deallocate(image_aos)
        allocate(image_aos(img_height, img_width), stat = errstat, errmsg = errmess)
        if (errstat /= 0) stop errmess

        ! Allocate the SOA image red array
        if (allocated(image_soa%red)) deallocate(image_soa%red)
        allocate(image_soa%red(img_height, img_width), stat = errstat, errmsg = errmess)
        if (errstat /= 0) stop errmess

        ! Allocate the SOA image green array
        if (allocated(image_soa%green)) deallocate(image_soa%green)
        allocate(image_soa%green(img_height, img_width), stat = errstat, errmsg = errmess)
        if (errstat /= 0) stop errmess

        ! Allocate the SOA image blue array
        if (allocated(image_soa%blue)) deallocate(image_soa%blue)
        allocate(image_soa%blue(img_height, img_width), stat = errstat, errmsg = errmess)
        if (errstat /= 0) stop errmess
    end subroutine allocate_new_image
end module ppm_image_io
