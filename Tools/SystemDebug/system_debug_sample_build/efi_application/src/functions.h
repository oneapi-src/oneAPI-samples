/**
 * @file    main.c
 * @brief   Main file (entry point) for the debugger EFI test app (inferior).
Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
SPDX-License-Identifier: BSD-2-Clause-Patent
 */
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "asmutils.h"
#include "dbgapp.h"
#include <stdbool.h>



#if defined(__clang__)
#define TOOLCHAIN   L"LLVM Clang"
#elif defined(__GNUC__)
#define TOOLCHAIN   L"GCC"
#elif defined(_MSC_VER)
#define TOOLCHAIN   L"Microsoft Visual C++"
// Needed for Visual Studio linker
#endif

#if defined(GNU_EFI)
#define EFI_FRAMEWORK "GNU EFI"
#elif defined(EDK2)
#define EFI_FRAMEWORK "EDK2"
#else
#define EFI_FRAMEWORK "UNKNOWN"
#endif

#ifndef PAGE_SIZE
#define PAGE_SIZE   (1 << 12)
#endif

#define IA32_SYSENTER_EIP 0x176


extern const EFI_SYSTEM_TABLE *g_st;

inline unsigned size2pages(unsigned size) {
    return (size / PAGE_SIZE) + ((size % PAGE_SIZE) ? 1 : 0);
}

/**
 * Allocate a memory buffer that can be also used to store UEFI app code.
 * @param size      Size of the buffer in bytes.
 * @return          A pointer to the newly allocated buffer (virtual address).
 */
inline void * efi_malloc(unsigned size) {
    EFI_PHYSICAL_ADDRESS phys = 0;
    EFI_STATUS ret;
    const unsigned pages = size2pages(size);

    ret = uefi(g_st->BootServices->AllocatePages, 4,
               AllocateAnyPages, EfiLoaderCode, pages, &phys);

    /* We are running in with identity mapping (virt == phys). */
    return ret == EFI_SUCCESS ? (void *)phys : NULL;
}


/**
 * Print a single (wide) string on the console. default ends with NewLine
 */
static inline void print_str_out(CHAR16 *str, int endWithNewLine) {
    SIMPLE_TEXT_OUTPUT_INTERFACE *out = g_st->ConOut;
    static CHAR16 crlf[] = L"\r\n";

    if (!out) {
        return;
    }
    uefi(out->OutputString, 2, out, str);

    if (endWithNewLine) {
        uefi(out->OutputString, 2, out, crlf);
    }
}

/**
 * Runs an infinite deadloop for the cpu
 */
unsigned int cpu_dead_loop();

/**
 * Print a single (wide) string on the console
 */
void print_str(CHAR16 *str);

/**
 * Print a sing unsigned int to the console
 */
void print_uint64(uint64_t x);

/**
 * Get the image base virtual address.
 */
uint64_t get_image_base(EFI_HANDLE img);


/**
 * Write a value to r11, and notify the debugger by writing to io port 0x90
 * @param to_dbg value to write to r1q1
 */
uint64_t notify_debugger(uint64_t to_dbg);


/**
 * Print the banner to the efi console
 *
 */
void print_banner(uint64_t base);

/**
 * Function to simulate callstack behaviour
 *
 */
int
callstack(int x, int y, int z);

/**
 * Triggers smm entry using an io port write
 *
 */
void trigger_smm();

/**
 * Function to simulate callstack behaviour
 *
 */
UINT32
EFIAPI
MmioReadEfi (UINTN Address);

#endif