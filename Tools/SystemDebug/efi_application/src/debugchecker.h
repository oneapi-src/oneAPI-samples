/** @file
debugchecker library functions.


Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
SPDX-License-Identifier: BSD-2-Clause-Patent
**/

#ifndef DEBUG_CHECKER_H
#define DEBUG_CHECKER_H

#if defined(EDK2)
#include <Uefi.h>

#include <Library/BaseLib.h>
#include <Library/UefiLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/IoLib.h>
#include <Protocol/LoadedImage.h>
#include <Library/UefiBootServicesTableLib.h>

#elif defined(GNU_EFI)
#include <efi.h>
#endif

#if defined(EDK2_LIN) || defined(EDK2)
#include <Register/Intel/Cpuid.h>
#endif

// Bit 4 of DCI Control Register
#define BIT_MASK_HDCIEN        (1 << 4)

#define DEBUG_PRIVACY_MSR    (0xC80)
#define IA32_RTIT_CTL        (0x570)
#define THREE_STRIKE_DISABLE (0x1A4)

// CPU ID information
extern UINT32 gCPU_Family;
extern UINT32 gCPU_Model;

// Call-to-action variables
extern UINT8 u8Run_Control;
extern UINT8 u8Processor_Trace;
extern UINT8 u8Three_Strike;
extern UINT16 u16SelectedIndex;

void getDbgInfo();

#endif // DEBUG_CHECKER_H_
