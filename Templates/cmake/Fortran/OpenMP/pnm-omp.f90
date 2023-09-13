!==============================================================
! Copyright © Intel Corporation
!
! SPDX-License-Identifier: MIT
!=============================================================

module PNM
    use iso_fortran_env
    implicit none

contains
    subroutine write_pgm(name, data)
        character(len=*), intent(in)    :: name
        integer(kind=int8), intent(in)  :: data(:,:)
        integer                         :: image_shape(2)
        integer                         :: unit
        character(len=128)              :: header
        
        open(file=name, access='stream', form='unformatted', action='write', newunit=unit)

        image_shape = shape(data)
        ! header is in ASCII and needs formatting
        write (header, "(A,A)") "P5", new_line(header)
        write (unit) trim(header)
        write (header, "(I0, ' ',I0,A)") image_shape(1), image_shape(2), new_line(header)
        write (unit) trim(header)
        write (header, "(I0,A)") 255, new_line(header)
        write (unit) trim(header)
        ! write raw image data
        write (unit) data
    end subroutine write_pgm
end module PNM
