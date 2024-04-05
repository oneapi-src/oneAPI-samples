#ifndef __SVD_TESTBENCH_TOOL_HPP__
#define __SVD_TESTBENCH_TOOL_HPP__

#include <iostream>
#include <iomanip>
#include <vector>

#define EPSILON 2E-6

namespace svd_testbench_tool {  // not for kernel code


template <typename TT>
void soft_transpose(std::vector<std::vector<TT>> &origin, 
        unsigned rows, unsigned cols, 
        std::vector<std::vector<TT>> &transposed)
{   
    // just swap row and col
    for (unsigned row = 0; row < rows; row ++)
    {
        for (unsigned col = 0; col < cols; col ++)
        {
            transposed[col][row] = origin[row][col];
        }
    }
}

// for col major matrix
template <typename TT>
void soft_transpose(std::vector<TT> &mat_A,
                    unsigned rows, unsigned cols,
                    std::vector<TT> &mat_At)
{
    for (int i = 0; i < (rows*cols); i ++)
    {   
        int cur_col = int(i / rows);
        int cur_row = i % rows;
        mat_At[cur_row * cols + cur_col] = mat_A[i];
    }
}

template <typename TT>
void soft_matmult(std::vector<std::vector<TT>> &mat_A,
            unsigned rows_A, unsigned cols_A,
            std::vector<std::vector<TT>> &mat_B,
            unsigned rows_B, unsigned cols_B,
            std::vector<std::vector<TT>> &mat_AB)
{
    assert((cols_A == rows_B) && "Mat_Mult with illegal matrix sizes");
    // Initializing AB to 0s
    for(unsigned row = 0; row < rows_A; row ++)
    {
        for(unsigned col = 0; col < cols_B; col++)
        {
            mat_AB[row][col] = 0.0;
        }
    }
    // Multiplying matrix A and B and storing in AB.
    for(unsigned row = 0; row < rows_A; row ++)
    {
        for(unsigned col = 0; col < cols_B; col ++)
        {
            for(unsigned item = 0; item < cols_A; item ++)
            {
                mat_AB[row][col] += mat_A[row][item] * mat_B[item][col];
            }
        }
    }
}

// for col major matrix
template <typename TT>
void soft_matmult(std::vector<TT> &mat_A,
            unsigned rows_A, unsigned cols_A,
            std::vector<TT> &mat_B,
            unsigned rows_B, unsigned cols_B,
            std::vector<TT> &mat_AB)
{
    std::vector<std::vector<TT>> A_2d(rows_A , std::vector<TT> (cols_A, 0));
    std::vector<std::vector<TT>> B_2d(rows_B , std::vector<TT> (cols_B, 0));
    std::vector<std::vector<TT>> AB_2d(rows_A , std::vector<TT> (cols_B, 0));

    // turn A vertical
    for (int i = 0; i < (rows_A*cols_A); i ++)
    {   
        A_2d[i % rows_A][int(i / rows_A)] = mat_A[i];
    }

    // turn B vertical
    for (int i = 0; i < (rows_B*cols_B); i ++)
    {   
        B_2d[i % rows_B][int(i / rows_B)] = mat_B[i];
    }

    soft_matmult<TT>(A_2d, rows_A, cols_A, B_2d, rows_B, cols_B, AB_2d);

    for (int c = 0; c < cols_B; c ++)
    {
        for (int r = 0; r < rows_A; r ++){
            mat_AB[c * rows_A + r] = AB_2d[r][c];
        }
    }
}

} // end of name space 


#endif /* __SVD_TESTBENCH_TOOL_HPP__ */