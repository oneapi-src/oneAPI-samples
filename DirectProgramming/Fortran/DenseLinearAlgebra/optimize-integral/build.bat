: =============================================================
: Copyright © 2020 Intel Corporation
:
: SPDX-License-Identifier: MIT
: =============================================================
:
:
:******************************************************************************
: Content:
:
:  Build for int_sin
:******************************************************************************

del *.obj *.exe

 ifx /Od src/int_sin.f90 /o int_sin.exe
: ifx /O1 src/int_sin.f90 /o int_sin.exe
: ifx /O2 src/int_sin.f90 /o int_sin.exe
: ifx /O3 src/int_sin.f90 /o int_sin.exe