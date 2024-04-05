#ifndef __PRINT_MATRIX_HPP__
#define __PRINT_MATRIX_HPP__

#include <iostream>
#include <vector>

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

} // end of name space 

#endif // __PRINT_MATRIX_HPP__