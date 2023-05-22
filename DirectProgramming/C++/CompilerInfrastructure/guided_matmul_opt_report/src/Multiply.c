/*
 * SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
 * http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
 *
 * Copyright (C) Intel Corporation
 *
 * THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
 *
 */

#include "Multiply.h"
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
