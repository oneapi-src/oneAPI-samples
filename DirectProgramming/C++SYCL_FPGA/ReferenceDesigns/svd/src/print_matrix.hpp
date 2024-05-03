#ifndef __PRINT_MATRIX_HPP__
#define __PRINT_MATRIX_HPP__

#include <iostream>
#include <vector>

namespace svd_testbench_tool {  // not for kernel code

// Convert to string, if input is a float convert with a fixed digits
template <typename T>
std::string ToFixedString(T value, int digits) {
    std::string stringValue = std::to_string(value);

    if (typeid(T) == typeid(float) || 
            typeid(T) == typeid(double)) {
        // Find the position of the decimal point
        size_t decimalPos = stringValue.find('.');
        if (decimalPos != std::string::npos) {
            // Extract the substring up to the specified significant digits
            size_t endIndex = decimalPos + digits + 1; // +1 for the decimal point
            if (endIndex < stringValue.length()) {
                stringValue = stringValue.substr(0, endIndex);
            }
        }
    }

    return stringValue;
}

template <typename T>
void PrintMatrix(std::vector<std::vector<T>> mat_A)
{
    for (unsigned row = 0; row < mat_A.size(); row ++)
    {
        std::cout << "[\t";
        for (unsigned col = 0; col < mat_A[0].size(); col ++)
        {
            std::cout << ToFixedString<T>(mat_A[row][col], 2) << ",\t";
        }
        std::cout << "]\n";
    }
}

// for stream matrix
template <typename T>
void PrintMatrix(std::vector<T> mat_A, int rows, int cols, bool col_maj=true)
{

    if (!col_maj) 
    {
        for (unsigned i = 0; i < mat_A.size(); i ++)
        {   
            // if its the start of a row
            if (i % cols == 0) std::cout << "[\t";

            std::cout << ToFixedString<T>(mat_A[i], 2) << ",\t";

            // or if its the end of a row
            if ((i+1) % cols == 0) std::cout << "]\n";
        }
    }
    else
    {
        std::vector<std::vector<T>> temp_mat(rows, std::vector<T>(cols));
        for (unsigned i = 0; i < mat_A.size(); i ++)
        {
            int cur_col = i / rows;
            int cur_row = i % rows;
            temp_mat[cur_row][cur_col] = mat_A[i];
        }
        PrintMatrix<T>(temp_mat);
    }
}

} // end of name space 

#endif // __PRINT_MATRIX_HPP__