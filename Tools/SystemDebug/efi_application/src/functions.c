
 /**
  *
  * @file    functions.c
  * @brief   helper functions for efi application
Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
SPDX-License-Identifier: BSD-2-Clause-Patent
  */

#include "functions.h"

static unsigned fibo(unsigned n) {                      /* step_into_result marker */
    unsigned ret;

    if (n <= 1) {
        return 1;                                       /* calls_test_breakpoint marker */
    }

    ret = fibo(n - 1) + fibo(n - 2);

    return ret;
}


void trigger_smm() {
    write_io(0xd2, 1);
}

static int test_print_count = 0;
static void test_print() {
    if (test_print_count<5) {
        print_str(L"test print !!! ");
    }
    if (test_print_count > 30000) { test_print_count = 0; }
    test_print_count++;
}

inline void test_inline() {
    /* Check callstack:
   -> The stackframe on thread 0 should display the following:
   [dbgapp.efi] test_inline():...
   [dbgapp.efi] efi_main():...
   */

    /* Check local variables: Open "Variables" view and step into the instructions below
    -> Variables are declared with proper types and values, values of a change accordingly to instructions
    */
    unsigned int a = 5;
    char c = 'c';
    a++;
    a--;
    c++;
    c--;

    /* Step Into: Click on "Step Into (F5)". Perform multiple Step Into to return inside this function
    -> pointer moves inside the print_str() function
    */
    print_str(L"inline test ! ");

    /* Step Over: Click on "Step Over (F6)"
    -> pointer moves to line below without entering the function below
    */
    print_str(L"Step over this");

    /* Run until caller: Click on "Run until caller (F7)"
    -> Pointer exits this function and comes back to efi_main() function
    */
}

static void test_primitive_types(unsigned int int1, char char1) {
    unsigned char uc = 'u';
    char c = 'c';
    signed char sc = -2;
    unsigned int ui = 2;
    signed int si =-3;
    int t[4]={0,1,3,3};
    bool b = true;
    float  f = 3.14e+0f;
    double  d = 3.14444466666;
    volatile int vi = 5;
    uc++;                                               /* variable_test1_breakpoint marker */
    uc--;
    c = (char) (char1 + 10);
    sc = (char) (c + sc);
    t[2]=(int)ui;
    ui = (unsigned int)uc + int1;
    si *= t[2];
    b = t[0] && b;
    f = (float) (f*5);
    d *= vi;
}                                                       /* variable_test2_breakpoint marker */


/**
  Used to serialize load and store operations.

  All loads and stores that proceed calls to this function are guaranteed to be
  globally visible when this function returns.

**/
VOID
EFIAPI
MemoryFence (
  VOID
  )
{
  return;
}



UINT32
EFIAPI
MmioReadEfi (UINTN Address)
{
  UINT32                            Value;
  MemoryFence ();
  Value = *(volatile UINT32*)Address;
  MemoryFence ();

  return Value;
}



/* Print a 64-bit value in hexadecimal. */
void print_uint64(uint64_t x) {
    CHAR16 str[] = L"0000000000000000";

    for (unsigned i = 15; x && (i < ARRAY_SIZE(str)); i--) {
        const int digit = x & 0xf;
        str[i] = (CHAR16)(digit > 9 ? (L'a' + digit - 0xa) : (L'0' + digit));
        x >>= 4;
    }

    print_str(str);
}

/**
 * Get the image base virtual address.
 */
uint64_t get_image_base(EFI_HANDLE img) {
    EFI_LOADED_IMAGE *loaded = NULL;
    EFI_STATUS ret = EFI_SUCCESS;
    EFI_GUID prot = LOADED_IMAGE_PROTOCOL;
    uint64_t base;

    ret = uefi(g_st->BootServices->HandleProtocol, 3, img, &prot, &loaded);
    if (ret != EFI_SUCCESS) {
        print_str(L"Failed to get the image base address");
        return 0;
    }

    if (!loaded) {
        print_str(L"LOADED_IMAGE_PROTOCOL returned NULL");
        return 0;
    }

    base = (uint64_t)loaded->ImageBase;
    return base;
}

uint64_t notify_debugger(uint64_t to_dbg) {
    uint64_t r11;
    static const uint16_t contract_io_port  = 0x90;
    static const uint16_t contract_io_value = 0xcafe;

    /* Write the data to be consumed by the debugger. */
    write_r11(to_dbg);
    /* Give control to the debugger. */
#ifdef BADLY_BEHAVED
    probe_mode_redirect();
#else
    write_io(contract_io_port, contract_io_value);
    /* Get the data from the debugger. */
#endif
    r11 = read_r11();

    return r11;
}

void print_banner(uint64_t base) {
    print_str(L"\r\nIntel System Studio test application\r\n");
    print_str(L"Built on " __DATE__ " at " __TIME__);
    print_str(L"Toolchain: " TOOLCHAIN);
    print_str(L"Framework: " EFI_FRAMEWORK);
    print_str(L"EFI image base address:");
    print_uint64(base);
}

void print_str(CHAR16 *str) {
    print_str_out(str, 1);
}

/**
 * Free a buffer allocated with malloc().
 */
inline void efi_free(void *buf, unsigned size) {
    if (!buf)
        return;

    const unsigned pages = size2pages(size);

    uefi(g_st->BootServices->FreePages, 2, (EFI_PHYSICAL_ADDRESS)buf, pages);
}

/* Check if an address is in the memory range of a given buffer. */
inline int in_buffer(EFI_VIRTUAL_ADDRESS addr, const void *buf, unsigned size) {
    const EFI_VIRTUAL_ADDRESS b = (EFI_VIRTUAL_ADDRESS)buf;

    return (b <= addr) && (addr < (b + size));
}


unsigned int cpu_dead_loop()
{
    post(0xdead);

    volatile unsigned int wait = 1;
    volatile unsigned long dummy = 0;
    while (wait) {                                       /* hardware_breakpoint marker */
        dummy += 0xcafe;
    }

    return wait;
}


int
d(void)
{
        return 4711;
}

int
c(int z)
{
        return z * d();
}

int
b(int y, int z)
{
        return y * c(z);
}

int
callstack(int x, int y, int z)
{
        return x * b(y, z);
}

