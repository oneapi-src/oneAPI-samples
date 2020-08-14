# Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
# SPDX-License-Identifier: BSD-2-Clause-Patent


# Allow this CMakeLists.txt to be used a a top-level file
cmake_minimum_required(VERSION 3.7)
project(uxdbgapp LANGUAGES C)
if(APPLE) # Use crosscompiler toolchain

    set(CMAKE_C_COMPILER x86_64-elf-gcc)
    set(CMAKE_CXX_COMPILER x86_64-elf-g++)

    # check for crosscompilers on Mac OS
    find_program(AR x86_64-elf-ar)
    find_program(RANLIB x86_64-elf-ranlib)

    if(NOT AR OR NOT RANLIB)
        message(FATAL_ERROR "elf tools not found! Please install x86_64-elf-binutils")
    endif()

    SET(CMAKE_AR ${AR})
    SET(CMAKE_RANLIB ${RANLIB})
    SET(MAKE_PREFIX x86_64-elf-)

    set(CREATE_ELF_HEADER ln -f -s /usr/local/include/elf.h .)
    set(CMAKE_C_COMPILER_WORKS 1)
    set(CMAKE_CXX_COMPILER_WORKS 1)
    set(CMAKE_C_COMPILER_ID GNU)
    set(CMAKE_CXX_COMPILER_ID GNU)
endif()

if (NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif ()

if (NOT DEFINED TOOLS_INSTALL_PREFIX)
    set(TOOLS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif ()

set(EFI_APP_NAME ${CMAKE_PROJECT_NAME})
# Installation folder for the EFI application binaries
set(EFI_APP_INSTALL_DIR ${TOOLS_INSTALL_PREFIX}/efi)

set(HAVE_EFI FALSE)
if (NOT DEFINED EDK2_ROOT)
    set(EDK2_ROOT ${CMAKE_SOURCE_DIR}/edk2)
endif ()

# Required for Windows (MSVC is multi-config but Ninja is not)
get_property(MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if (CMAKE_HOST_UNIX)
    set(EFI_APP_AR  ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${EFI_APP_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(EFI_APP_SO  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EFI_APP_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(EFI_APP_EFI ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EFI_APP_NAME}.efi)
    set(EFI_APP_DBG ${EFI_APP_SO})

    set(EFI_ARCH x86_64)

    set(EFI_DIR ${CMAKE_SOURCE_DIR}/gnu-efi-code)
    set(EDK_INC_DIR ${EDK2_ROOT}/MdePkg/Include)
    set(EFI_INC_DIR  ${EFI_DIR}/inc)
    set(EFI_INC_DIRS ${EDK_INC_DIR} ${EDK_INC_DIR}/X64 ${EFI_INC_DIR} ${EFI_INC_DIR}/${EFI_ARCH})
    if (EXISTS ${EFI_INC_DIR} AND IS_DIRECTORY ${EFI_INC_DIR})
        set(HAVE_EFI TRUE)
        else()
        message(FATAL_ERROR "Please clone gnuefi submodule")
    endif()

    set(EFI_LIB_DIR ${EFI_DIR}/${EFI_ARCH}/lib)
    set(EFI_LIB_DIR_GNU ${EFI_DIR}/gnuefi)

    set(EFI_CRT ${EFI_LIB_DIR_GNU}/crt0-efi-${EFI_ARCH}.o)

    set(libefifile ${EFI_LIB_DIR_GNU}/libefi.a)
    add_custom_command(OUTPUT ${libefifile}
            COMMAND ${CREATE_ELF_HEADER}
            COMMAND make prefix=${MAKE_PREFIX} -j
            WORKING_DIRECTORY ${EFI_DIR})
    add_custom_target(libefi DEPENDS ${libefifile})

    set(libgnuefifile ${EFI_LIB_DIR}libgnuefi.a)
    add_custom_command(OUTPUT ${libgnuefifile}
            COMMAND ${CREATE_ELF_HEADER}
            COMMAND make prefix=${MAKE_PREFIX}
            WORKING_DIRECTORY ${EFI_LIB_DIR_GNU})
    add_custom_target(libgnuefi DEPENDS ${libgnuefifile})
    if (CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(EFI_CFLAGS -debug -ffreestanding -O0 -c -fPIC -fno-stack-protector -fshort-wchar -mno-red-zone )
    else ()
        set(EFI_CFLAGS -g -ffreestanding -Og -fPIC -fno-stack-protector -fshort-wchar -mno-red-zone)
    endif()
    if (CMAKE_C_COMPILER_ID MATCHES "Clang")
        # setjmp is a builtin for Clang but present in the EFI lib headers...
        set(EFI_CFLAGS ${EFI_CFLAGS} -Wno-error=incompatible-library-redeclaration)
    endif ()
    set(EFI_APP_DEFS -DGNU_EFI -DEDK2_LIN)
    set(EFI_LDS ${EFI_DIR}/gnuefi/elf_${EFI_ARCH}_efi.lds)
    set(EFI_LDFLAGS -nostdlib -znocombreloc -T${EFI_LDS} -shared -Bsymbolic -L${EFI_LIB_DIR} -L${EFI_LIB_DIR_GNU})

    set(EFI_LDLIBS -lefi -lgnuefi)

    set(OBJCOPY_OPTS_EFI --target=efi-app-${EFI_ARCH} -j .text -j .sdata -j .data
                            -j .dynamic -j .dynsym  -j .rel -j .rela -j .reloc -j .eh_frame)

elseif (CMAKE_HOST_WIN32)

    if (NOT CMAKE_GENERATOR_PLATFORM MATCHES X64)
        message(FATAL_ERROR "ERROR: Only 64 bit builds are supported. Please build with -AX64 ${CMAKE_GENERATOR_PLATFORM}" )
    endif()
     set(EFI_ARCH X64)

    if (NOT DEFINED EDK2_TOOLS)
        set(EDK2_TOOLS ${CMAKE_SOURCE_DIR}/edk2/BaseTools/Win32)
    endif ()

    set(EFI_APP_DEFS -DEDK2)
    # Name of the app in its TE (Terse Executable incarnation)
    set(EFI_APP_TE_NAME ${EFI_APP_NAME}-te)

    if (MULTI_CONFIG)
        set(EFI_APP_AR  ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/$<CONFIG>/${EFI_APP_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})

        set(EFI_APP_SO  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/${EFI_APP_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(EFI_APP_DBG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/${EFI_APP_NAME}.pdb)
        set(EFI_APP_EFI ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/${EFI_APP_NAME}.efi)

        set(EFI_APP_TE_SO  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/${EFI_APP_TE_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(EFI_APP_TE_DBG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/${EFI_APP_TE_NAME}.pdb)
        set(EFI_APP_TE_EFI ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/${EFI_APP_TE_NAME}.efi)
    else ()
        set(EFI_APP_AR  ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${EFI_APP_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})

        set(EFI_APP_SO  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${EFI_APP_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(EFI_APP_DBG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${EFI_APP_NAME}.pdb)
        set(EFI_APP_EFI ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${EFI_APP_NAME}.efi)

        set(EFI_APP_TE_SO  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${EFI_APP_TE_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(EFI_APP_TE_DBG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${EFI_APP_TE_NAME}.pdb)
        set(EFI_APP_TE_EFI ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${EFI_APP_TE_NAME}.efi)
    endif ()

    set(GENFW GenFw.exe)
    file(TO_NATIVE_PATH ${EFI_APP_EFI} EFI_NATIVE_PATH)
    file(TO_NATIVE_PATH ${EFI_APP_SO} SO_NATIVE_PATH)
    file(TO_NATIVE_PATH ${EFI_APP_TE_EFI} TE_EFI_NATIVE_PATH)
    file(TO_NATIVE_PATH ${EFI_APP_TE_SO} TE_SO_NATIVE_PATH)

    set(EFI_INC_DIR ${EDK2_ROOT}/MdePkg/Include)
    set(EFI_INC_DIRS ${EFI_INC_DIR} ${EFI_INC_DIR}/${EFI_ARCH})

    if (NOT (EXISTS ${EFI_INC_DIR} AND IS_DIRECTORY ${EFI_INC_DIR}))
        message(FATAL_ERROR "ERROR: ${EFI_INC_DIR} folder not found, please set EDK2_ROOT.")
    elseif (NOT EXISTS ${EDK2_TOOLS}/${GENFW})
        message(FATAL_ERROR "ERROR: ${GENFW} not found, please set EDK2_TOOLS.")
    else ()
        set(HAVE_EFI TRUE)
        # Remove C runtime library support, enable debug info, disable optimizations and buffer security
        # disable cf guard, as this plays havoc without runtime library
        set(EFI_CFLAGS /Zl /Zi /Od /GS- /Oy- /guard:cf-)
        set(EFI_LDFLAGS /machine:${EFI_ARCH} /entry:efi_main /dll /ignore:4001 /debug /nodefaultlib)

        #ignore edk2 errors when headers are used in cpp context
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4804 /wd4005")
        string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
     string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")


    endif ()

endif()

if (HAVE_EFI)
    option(BAD "Set if the efi application should be badly behaved" OFF)


    if (BAD)
        set(EFI_APP_DEFS  ${EFI_APP_DEFS} -DBADLY_BEHAVED)
    endif()

    add_subdirectory(${CMAKE_SOURCE_DIR}/src)


endif ()
