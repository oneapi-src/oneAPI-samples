/**
 * @file    main.c
 * @brief   Main file (entry point) for the debugger EFI test app (inferior).
 *
 *
Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
SPDX-License-Identifier: BSD-2-Clause-Patent
 */

#include "asmutils.h"
#include "dbgapp.h"
#include "functions.h"
#include "debugchecker.h"
#include <stdbool.h>

const EFI_SYSTEM_TABLE *g_st;

int var = 0;

EFI_STATUS EFIAPI efi_main(EFI_HANDLE image, EFI_SYSTEM_TABLE *st) {
    static const uint32_t msr_addr = IA32_SYSENTER_EIP;

    volatile unsigned app_size = 0;
    void *app_buf = NULL;
    uint64_t base;
    if (!st) {
        return EFI_INVALID_PARAMETER;
    }
    g_st = st;


    print_str(L"Welcome to the test application");
    base = get_image_base(image);
    print_banner(base);
    getDbgInfo();



#ifdef BADLY_BEHAVED
    notify_debugger(15);
#else
    notify_debugger(12);
#endif

    int x = var, y = 3, z = 5;
    callstack(x, y, z);
    cpu_dead_loop();

    return EFI_SUCCESS;
}
