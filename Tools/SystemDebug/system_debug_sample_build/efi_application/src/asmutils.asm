; Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
; SPDX-License-Identifier: BSD-2-Clause-Patent

_TEXT SEGMENT

PUBLIC write_r10
PUBLIC write_r11
PUBLIC read_r11
PUBLIC probe_mode_redirect
PUBLIC _mwait
PUBLIC _hlt

write_r10 PROC
    mov  r10, rcx
    ret
write_r10 ENDP


write_r11 PROC
    mov  r11, rcx
    ret
write_r11 ENDP


read_r11 PROC
    mov  rax, r11
    ret
read_r11 ENDP

read_msr PROC
    rdmsr
    shl rdx, 32
    or rax, rdx
    ret
read_msr ENDP



write_io PROC
    mov ax, dx
    mov dx, cx
    out dx, ax
    ret
write_io ENDP

probe_mode_redirect PROC
    byte 0f1h
    ret
probe_mode_redirect ENDP

_cpuid PROC
    push    rbx
    mov     eax, ecx
    push    rax                         ; save Index on stack
    push    rdx
    cpuid
    test    r9, r9
    jz      one
    mov     [r9], ecx
one:
    pop     rcx
    jrcxz   two
    mov     [rcx], eax
two:
    mov     rcx, r8
    jrcxz   three
    mov     [rcx], ebx
three:
    mov     rcx, [rsp+38h]
    jrcxz   four
    mov     [rcx], edx
four:
    pop     rax                         ; restore Index to rax as return value
    pop     rbx
    ret
_cpuid ENDP

_mwait PROC
    ; opcode of MWAIT is 0f 01 c9
    db 15
    db 1
    db 201
    ret
_mwait ENDP

_hlt PROC
    hlt
    ret
_hlt ENDP

_TEXT ENDS

END
