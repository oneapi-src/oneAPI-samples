/**
 * @file    asmutils.h
 * @brief   Assembly-related utilities.
 *
 Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
SPDX-License-Identifier: BSD-2-Clause-Patent
 */

#ifndef ASMUTILS_H
#define ASMUTILS_H

#include "dbgapp.h"

#ifdef _MSC_VER

/**
 * Write to the R10 register.
 */
extern void write_r10(uint64_t val);

/**
 * Write to the R11 register.
 */
extern void write_r11(uint64_t val);

/**
 * Read the R11 register.
 */
extern uint64_t read_r11(void);

/**
 * Read the MSR at the specified address
 */
extern uint64_t read_msr(uint32_t addr);

/**
 * Trigger a probe mode redirect (halt the current HW thread).
 */
extern void probe_mode_redirect(void);

/**
 * run the cpuid command
 */
extern void _cpuid(uint32_t index, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx);

/**
 * Write a value to the IO port at the specified address
 */
extern void write_io(uint16_t val, uint16_t addr);

/**
 * Execute the HLT (halt) instruction
 */
extern void _hlt();

/**
 * Execute the MWAIT instruction.
 */
extern void _mwait();

#elif defined(__GNUC__)

static inline void write_r10(uint64_t val) {
    __asm__("movq %0, %%r10" :: "r"(val));
}

static inline void write_r11(uint64_t val) {
    __asm__("movq %0, %%r11" :: "r"(val));
}

static inline uint64_t read_r11(void) {
    uint64_t ret;
    __asm__("movq %%r11, %0" : "=r"(ret));
    return ret;
}

static inline uint64_t read_msr(uint32_t addr) {
    uint64_t ret;
    __asm__(
        "movl %1, %%ecx \n"
        "rdmsr \n"
        "shl $32, %%rdx \n"
        "or %%rdx, %%rax \n"
        "movq %%rax, %0"

        : "=r"(ret)
        : "r"(addr)
    );
    return ret;
}



static inline void write_io(uint16_t addr, uint16_t val) {
    __asm__(
        "mov %0, %%ax \n"
        "mov %1, %%dx \n"
        "out %%ax, %%dx"

        :: "r"(val), "r"(addr)
    );
}

static inline void _cpuid(uint32_t index, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
__asm__ volatile(
  "      movl      %0, %%eax\n"
  "      cpuid\n"
  "      mov     %%rcx, %%r9\n"
  "      movq      %1, %%rcx\n"
  "      jrcxz    1f\n"
  "      movq      %%rax, (%%rcx)\n"
  "1:    \n"
  "      movq      %2, %%rcx\n"
  "      jecxz    2f\n"
  "      movq      %%rbx, (%%rcx)\n"
  "2:    \n"
  "      mov      %%r9,%%rax\n"
  "      movq      %3, %%rcx\n"
  "      jrcxz    3f\n"
  "      movq      %%rax, (%%rcx)\n"
  "3:    \n"
  "      movq      %4, %%rcx\n"
  "      jrcxz    4f\n"
  "      movq      %%rdx, (%%rcx)\n"
  "4:    \n"
  "   movl %0, %%eax"
  :
  :"m"(index), "m"(eax), "m"(ebx), "m"(ecx), "m"(edx)
  :"memory","rsi", "rdi", "rcx"

);
}

static inline void probe_mode_redirect(void) {
    __asm__(".byte 0xf1" ::);
}

static inline void _hlt() {
    __asm__("hlt" ::);
}

static inline void _mwait() {
    __asm__("mwait" ::);
}

#endif

/**
 * Write a POST code to the 0x80 IO port
 */
static inline void post(uint16_t val) {
    write_io(0x80, val);
}

#endif /* ASMUTILS_H */
