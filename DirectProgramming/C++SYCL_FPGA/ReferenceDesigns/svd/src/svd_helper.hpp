#ifndef __SVD_HELPER_HPP__
#define __SVD_HELPER_HPP__

#include "dpc_common.hpp"
#include "tuple.hpp"
#include "constexpr_math.hpp"
#include "unrolled_loop.hpp"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

#include <iostream>
#include <iomanip>
#include <vector>

#define EPSILON 2E-6

namespace svd_testbench_tool {  // not for kernel code

template <typename TT>
void print_matrix(std::vector<std::vector<TT>> mat_A)
{
    // if its a floating point type number, fixed the print precision
    if (typeid(TT) == typeid(float) || 
            typeid(TT) == typeid(double))
    {
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
    }

    for (unsigned row = 0; row < mat_A.size(); row ++)
    {
        std::cout << "[\t";
        for (unsigned col = 0; col < mat_A[0].size(); col ++)
        {
            std::cout << mat_A[row][col] << ",\t";
        }
        std::cout << "]\n";
    }
}

// for stream matrix
template <typename TT>
void print_matrix(std::vector<TT> mat_A, int rows, int cols, bool col_maj=true)
{
    if (typeid(TT) == typeid(float) || 
            typeid(TT) == typeid(double))
    {
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
    }

    if (!col_maj) 
    {
        for (unsigned i = 0; i < mat_A.size(); i ++)
        {   
            // if its the start of a row
            if (i % cols == 0) std::cout << "[\t";

            std::cout << mat_A[i] << ",\t";

            // or if its the end of a row
            if ((i+1) % cols == 0) std::cout << "]\n";
        }
    }
    else
    {
        std::vector<std::vector<TT>> temp_mat(rows, std::vector<TT>(cols));
        for (unsigned i = 0; i < mat_A.size(); i ++)
        {
            int cur_col = i / rows;
            int cur_row = i % rows;
            temp_mat[cur_row][cur_col] = mat_A[i];
        }
        print_matrix<TT>(temp_mat);
    }
}

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
            // std::cout << "Paddind AB: " << row << ", " << col << std::endl;
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
                // std::cout << "About to mult: " << row << ", " << col << ", " << item << std::endl;
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


#endif /* __SVD_HELPER_HPP__ */