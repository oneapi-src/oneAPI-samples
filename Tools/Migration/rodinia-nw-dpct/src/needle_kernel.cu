
#include "needle.h"
#include <stdio.h>


#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

__device__ __host__ int
maximum( int a,
         int b,
         int c){

int k;
if( a <= b )
k = b;
else
k = a;

if( k <=c )
return(c);
else
return(k);

}

__global__ void
needle_cuda_shared_1(   int* referrence,
                        int* matrix_cuda,
                        int cols,
                        int penalty,
                        int i,
                        int block_width)
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];

   if (tx == 0)
          temp[tx][0] = matrix_cuda[index_nw];


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  __syncthreads();

  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = matrix_cuda[index_n];

  __syncthreads();


  for( int m = 0 ; m < BLOCK_SIZE ; m++){

      if ( tx <= m ){

          int t_index_x =  tx + 1;
          int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                                                temp[t_index_y-1][t_index_x]  - penalty);



      }

      __syncthreads();

    }

 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

      if ( tx <= m){

          int t_index_x =  tx + BLOCK_SIZE - m ;
          int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                                                temp[t_index_y-1][t_index_x]  - penalty);

      }

      __syncthreads();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}


__global__ void
needle_cuda_shared_2(   int* referrence,
                        int* matrix_cuda,
                        int cols,
                        int penalty,
                        int i,
                        int block_width)
{

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int b_index_x = bx + block_width - i  ;
    int b_index_y = block_width - bx -1;

    int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
    int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
    int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
    int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

    __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
    __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];

    for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
    ref[ty][tx] = referrence[index + cols * ty];

    __syncthreads();

    if (tx == 0)
        temp[tx][0] = matrix_cuda[index_nw];

    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

    __syncthreads();

    temp[0][tx + 1] = matrix_cuda[index_n];

    __syncthreads();

    for( int m = 0 ; m < BLOCK_SIZE ; m++){
        if ( tx <= m ){
            int t_index_x =  tx + 1;
            int t_index_y =  m - tx + 1;
            temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                  temp[t_index_y][t_index_x-1]  - penalty,
                                                  temp[t_index_y-1][t_index_x]  - penalty);
        }
    __syncthreads();
    }

    for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
        if ( tx <= m){
            int t_index_x =  tx + BLOCK_SIZE - m ;
            int t_index_y =  BLOCK_SIZE - tx;
            temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                  temp[t_index_y][t_index_x-1]  - penalty,
                                                  temp[t_index_y-1][t_index_x]  - penalty);
        }
        __syncthreads();
    }

    for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
    matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];
}
