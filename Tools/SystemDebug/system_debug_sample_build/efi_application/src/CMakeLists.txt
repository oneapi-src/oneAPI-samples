# Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
# SPDX-License-Identifier: BSD-2-Clause-Patent


file(GLOB EFI_APP_SOURCES *.c)

if (CMAKE_HOST_WIN32)
    enable_language(ASM_MASM)
    set(ASMUTILS_OBJ "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/asmutils.obj")
    set(ASMUTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/asmutils.asm")
    add_custom_command(OUTPUT ${ASMUTILS_OBJ}
    COMMAND ${CMAKE_ASM_MASM_COMPILER} /nologo /Fo"${ASMUTILS_OBJ}" /c /w "${ASMUTILS_SRC}")
endif(CMAKE_HOST_WIN32)

file(TO_NATIVE_PATH ${EFI_APP_EFI} EFI_NATIVE_PATH)
# HACK: On Unix, because we cannot choose the linker invoked by CMake (which by default invokes
# the CMAKE_C_COMPILER and not ld), we need an extra step. We first create a STATIC library
# to make sure we gathered/compiled all the sources/objects and then we call the ld manually
# with the right parameters. On Windows, for consistency and readability, we do the same.
add_library(${EFI_APP_NAME} STATIC ${EFI_APP_SOURCES} ${ASMUTILS_OBJ})

set_target_properties(${EFI_APP_NAME} PROPERTIES PREFIX "")
target_include_directories(${EFI_APP_NAME} PRIVATE ${EFI_INC_DIRS})

target_compile_options(${EFI_APP_NAME} PRIVATE ${EFI_APP_DEFS} ${EFI_CFLAGS})

# Prevent the GenFw from erasing the .pdata / .rdata (containing the unwind info)
set(GENFW_OPTS    --keepexceptiontable --keepoptionalheader -e APPLICATION)
set(GENFW_TE_OPTS --keepexceptiontable --keepoptionalheader -t)
if (CMAKE_HOST_UNIX)

    if (APPLE)
        find_program(LINKER x86_64-elf-ld DOC "The elf linker")
        find_program(OBJCOPY x86_64-elf-objcopy DOC "elf objcopy")
        if(NOT LINKER OR NOT OBJCOPY)
            message(FATAL_ERROR "elf tools not found! Please install x86_64-elf-binutils")
        endif()
    else()
        find_program(LINKER ld DOC "The elf linker")
        find_program(OBJCOPY objcopy DOC "elf objcopy")
        if(NOT LINKER)
            message(FATAL_ERROR "ld not found! Please install it")
        endif()
        if(NOT OBJCOPY)
            message(FATAL_ERROR "objcopy not found! Please install it")
        endif()
    endif(APPLE)
    add_custom_command(TARGET ${EFI_APP_NAME} POST_BUILD
    COMMAND ${LINKER} ${EFI_LDFLAGS} -o ${EFI_APP_SO} ${EFI_CRT}
                            --whole-archive ${EFI_APP_AR} ${EFI_LDLIBS}
    COMMAND ${OBJCOPY} ${OBJCOPY_OPTS_EFI} ${EFI_APP_SO} ${EFI_APP_EFI}
    DEPENDS ${EFI_APP_AR})
    add_dependencies(${EFI_APP_NAME} libefi libgnuefi)
elseif (CMAKE_HOST_WIN32)
    add_custom_command(TARGET ${EFI_APP_NAME} POST_BUILD
                        COMMAND link ${EFI_LDFLAGS} /out:${EFI_APP_SO} ${EFI_APP_AR}
                        COMMAND link ${EFI_LDFLAGS} /out:${EFI_APP_TE_SO} /filealign:4096 ${EFI_APP_AR}
                        COMMAND ${CMAKE_COMMAND} -E copy ${EDK2_TOOLS}/${GENFW} ${GENFW}
                        COMMAND ${GENFW} ${GENFW_OPTS} -o ${EFI_NATIVE_PATH} ${SO_NATIVE_PATH}
                        COMMAND ${GENFW} ${GENFW_TE_OPTS} -o ${TE_EFI_NATIVE_PATH} ${TE_SO_NATIVE_PATH}
                        DEPENDS ${EFI_APP_AR})
endif()