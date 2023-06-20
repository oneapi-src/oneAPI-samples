//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "multiply.h"
void matvec(int size1, int size2, FTYPE a[][size2], FTYPE b[], FTYPE x[])
{
    int i, j;

    for (i = 0; i < size1; i++) {
        b[i] = 0;
        for (j = 0;j < size2; j++) {
            b[i] += a[i][j] * x[j];
        }
    }
}
