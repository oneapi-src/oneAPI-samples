
/** @file
dbgapp library functions.


Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
SPDX-License-Identifier: BSD-2-Clause-Patent
**/

#ifndef DBGAPP_H
#define DBGAPP_H
#include <stdint.h>


#if defined(EDK2)
#include <Uefi.h>
#include <ProcessorBind.h>
#include <Protocol/LoadedImage.h>
#elif defined(GNU_EFI)
#include <efi.h>
#endif

/* On a native MSABI compiler, the wrapper is not needed. */
#ifdef uefi_call_wrapper
#define uefi(func, count, ...) uefi_call_wrapper(func, count, __VA_ARGS__)
#else
#define uefi(func, count, ...) func(__VA_ARGS__)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))
#endif


typedef void (* test_func)();

/**
 * Notify the debugger via a probe mode redirect and exchange data with it.
 * @param to_dbg    Data to be "sent" to the debugger.
 * @return          Data "received" from the debugger.
 */
uint64_t notify_debugger(uint64_t to_dbg);
void print_str(CHAR16 *str);

#endif /* DBGAPP_H */
