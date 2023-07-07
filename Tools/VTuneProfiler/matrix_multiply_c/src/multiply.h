/*
 *
 *
 * Copyright (C) 2005 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials, and your use of them
 * is governed by the express license under which they were provided to you ("License"). Unless
 * the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose
 * or transmit this software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
*/

#ifdef __MIC__
#define MAXTHREADS 240
#define NUM 3840
#define MATRIX_BLOCK_SIZE 16
#else
#define MAXTHREADS 16
#define NUM 2048
#define MATRIX_BLOCK_SIZE 64
#endif

typedef double TYPE;
typedef TYPE array[NUM];

// Select which multiply kernel to use via the following macro so that the
// kernel being used can be reported when the test is run.
#ifdef USE_MKL
#define MULTIPLY multiply5
#else
#define MULTIPLY multiply1
#endif

extern void multiply0(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply3(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply4(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply5(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);

void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
void GetModelParams(int* nthreads, int* msize, int print);
