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
:  Build for openmp_sample
:******************************************************************************

del *.obj *.exe 

ifx /O2 /Qopenmp src/openmp_sample.f90 /o openmp_sample.exe

ifx /Od /Qopenmp src/openmp_sample.f90 /o openmp_sample_dbg.exe



