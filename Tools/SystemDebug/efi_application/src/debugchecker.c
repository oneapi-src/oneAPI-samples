/**
 * @file    debugchecker.c
 * @brief   functinos to check if debugging is enabled
 *
 *
Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
SPDX-License-Identifier: BSD-2-Clause-Patent
 */

#include "debugchecker.h"
#include "dbgapp.h"
#include "asmutils.h"
#include "functions.h"


UINT32 gCPU_Family;
UINT32 gCPU_Model;

// Call-to-action variables
UINT8 u8Run_Control;
UINT8 u8Processor_Trace;
UINT8 u8Three_Strike;
UINT16 u16SelectedIndex;

/**
// Display Processor Version Information
*/
static void ProcessorVersionInfo() {
  CPUID_VERSION_INFO_EAX Eax;
  _cpuid(CPUID_VERSION_INFO, &Eax.Uint32, 0, 0,
           0);

  gCPU_Family = Eax.Bits.FamilyId;
  gCPU_Model = Eax.Bits.Model;
  if (Eax.Bits.FamilyId == 0x06 || Eax.Bits.FamilyId == 0x0F) {
    gCPU_Model |= (Eax.Bits.ExtendedModelId << 4);
  }

    print_str(L"Family:");
    print_uint64((uint64_t) gCPU_Family);
    print_str(L"Model:");
    print_uint64((uint64_t) gCPU_Model);
    print_str(L"Stepping:");
    print_uint64((uint64_t) Eax.Bits.SteppingId);

}

static void CheckRunControl() {
  UINT64 u64Debug = read_msr(DEBUG_PRIVACY_MSR);
  u8Run_Control = u64Debug & 0x1;

  print_str(L"Phase-2: Platform Debug Enabling checking" );

  if (u8Run_Control) {
    print_str(L"Run Control is enabled");
  } else {
    print_str(L"Run Control is disabled");
  }
}

static void CheckProcessorTrace() {
  UINT64 u64Debug = read_msr(IA32_RTIT_CTL);
  u8Processor_Trace = u64Debug & 0x1;

  print_str(L"Phase-3: Intel(r) Process Trace Enabling checking" );

  if (u8Processor_Trace) {
    print_str(L"Intel(R) Processor Trace is enabled");
  } else {
    print_str(L"Intel(R) Processor Trace is disabled");
  }
}

VOID CheckThreeStrike() {
  UINT64 u64Debug = read_msr(THREE_STRIKE_DISABLE);
  u8Three_Strike = (u64Debug & 0x0080) ? 1 : 0; // 0000 1000 0000 0000 - bit 11

  print_str(L"Phase-4: Crash Log configuration checking" );
  if (u8Three_Strike) {
      print_str(L"Three Strike is enabled");
  } else {
      print_str(L"Three Strike is disabled");
  }
}

static void checkDCIStatus(UINT32 u32Enabling_Status) {

  print_str(L"Phase-1: Host-Target connectivity checking" );

  if (u32Enabling_Status) {
    print_str(L"Platform Debug Consent is enabled.");
  } else {
    print_str(L"Platform Debug Consent is disabled.");
  }
}

void getDbgInfo() {

    UINT32 u32ECTRL_register;
    UINT32 u32ECTRL_register_address;

    // First, gather the information about this processor.
    ProcessorVersionInfo();

    // With processor info, different models have the ECTRL register at different
    // addresses.
    if (gCPU_Model > 0x86) {
        u32ECTRL_register_address =  0xFDB80004;
    } else {
        u32ECTRL_register_address =  0xD0A80004;
    }

    u32ECTRL_register = MmioReadEfi(u32ECTRL_register_address);

    UINT32 u32Enabling_Status = (u32ECTRL_register & BIT_MASK_HDCIEN) >> 4;
    checkDCIStatus(u32Enabling_Status);
    CheckRunControl();
    CheckProcessorTrace();
    CheckThreeStrike();

}