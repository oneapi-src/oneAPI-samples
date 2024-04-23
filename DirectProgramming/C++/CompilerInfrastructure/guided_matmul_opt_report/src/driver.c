//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#define ROW 101
#define COL 101
#define COLBUF 0
#define COLWIDTH COL+COLBUF
#define REPEATNTIMES 1000000
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include "multiply.h"

/* routine to initialize an array with data */

void init_matrix(int row, int col, FTYPE off, FTYPE  a[][COLWIDTH])
{
    int i,j;

    for (i=0; i< row;i++) {
        for (j=0; j< col;j++) {
            a[i][j] = fmod(i*j+off,10.0);
        }
    }
    if (COLBUF>0)
        for  (i=0;i<row;i++)
            for (j=col;j<COLWIDTH;j++)
                a[i][j]=0.0;
}

void init_array(int length, FTYPE off, FTYPE a[])
{
    int i;

    for (i=0; i< length;i++)
        a[i] = fmod(i+off,10.0);
    if (COLBUF>0)
        for  (i=length;i<COLWIDTH;i++)
                a[i]=0.0;
}

void printsum(int length, FTYPE ans[]) {
    /* Doesn't print the whole matrix - Just a very simple Checksum */
    int i;
    double sum=0.0;

    for (i=0;i<length;i++) sum+=ans[i];

    printf("Sum of result = %f\n", sum);
}



double clock_it(void)
{
    double duration = 0.0;
    struct timeval start;

    gettimeofday(&start, NULL);
    duration = (double)(start.tv_sec + start.tv_usec/1000000.0);
    return duration;
}



int main()
{
    double execTime = 0.0;
    double startTime, endTime;

    int k, size1, size2;

    FTYPE a[ROW][COLWIDTH];
    FTYPE b[ROW];
    FTYPE x[COLWIDTH];
    size1 = ROW;
    size2 = COLWIDTH;

    printf("\nROW:%d COL: %d\n",ROW,COLWIDTH);

    /* initialize the arrays with data */
    init_matrix(ROW,COL,1,a);
    init_array(COL,3,x);

    /* start timing the matrix multiply code */
    startTime = clock_it();
    for (k = 0;k < REPEATNTIMES;k++) {
#ifdef NOFUNCCALL
        int i, j;
        for (i = 0; i < size1; i++) {
            b[i] = 0;
            for (j = 0;j < size2; j++) {
                b[i] += a[i][j] * x[j];
            }
        }
#else
        matvec(size1,size2,a,b,x);
#endif
        x[0] = x[0] + 0.000001;
    }
    endTime = clock_it();
    execTime = endTime - startTime;

    printf("Execution time is %2.3f seconds\n", execTime);
    printf("GigaFlops = %f\n", (((double)REPEATNTIMES * (double)COL * (double)ROW * 2.0) / (double)(execTime))/1000000000.0);
    printsum(COL,b);

    return 0;
}
