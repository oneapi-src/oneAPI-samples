#pragma once

#include "Utils.hpp"

/*
  QRInversionDim is a helper class to compute the difference constexpr 
  expressions required by the different kernel of the QR Inversion.
  It aggregates all the calculus in one single place rather than having
  to recompute these values in each kernel.
*/
template< bool isComplex,     // Type of QR inversion: complex vs real
          int rows,           // Number of rows in the matrix
          int columns,        // Number of columns in the matrix
          int RAWLatency = -1 // The RAW latency for the triangular loop
                              // optimization in the QRD kernel
                              // Other kernels don't need to provide this value 
        >
class QRInversionDim {
  public:
  // Number of elements in the matrix
  static constexpr int AMatrixSize = columns * rows;
  static constexpr int InverseMatrixSize = columns * rows;

  // Sizes of allocated memories for input and output matrix
  // Both the input matrix and Q are full matrices of complex elements
  // R only contains columns + 
  //                 (columns - 1) + 
  //                 (columns - 2) +
  //                 (columns - 3) + 
  //                 etc.
  // So R contains columns * (columns + 1) / 2 complex elements.
  static constexpr int RMatrixSize = columns * (columns + 1) / 2;
  static constexpr int QMatrixSize = AMatrixSize;
  static constexpr int QRMatrixSize = QMatrixSize + RMatrixSize;

  // Constants related to the memory configuration of the kernel's local
  // memories
  // We want 8 floating-point values in each memory bank
  static constexpr int NumElementsPerBank = isComplex ? 4 : 8;

  // Set the bankwidth in bytes
  static constexpr int BankWidth = NumElementsPerBank * 8;
  static constexpr bool NonCompleteBank = rows%NumElementsPerBank != 0;
  static constexpr int ExtraBank = NonCompleteBank ? 1 : 0;
  static constexpr int NumBanks = rows / NumElementsPerBank + ExtraBank;
  static constexpr int NumBanksNextPow2 = Pow2(CeilLog2<NumBanks>());

  // Number of load and store iterations for a single matrix given the size
  // of the input matrices and the number of elements per bank
  static constexpr bool NonCompleteIter = rows%NumElementsPerBank != 0;
  static constexpr int ExtraIter = NonCompleteIter ? 1 : 0;
  static constexpr int LoadIter = ((rows/NumElementsPerBank) + ExtraIter) 
                                                                  * columns;
  static constexpr int StoreIter = LoadIter; 

  // Number of bits required by the loop counters for the load/store iterators
  static constexpr int LoadIterBitSize = BitsForMaxValue<LoadIter + 1>();
  static constexpr int StoreIterBitSize = BitsForMaxValue<StoreIter + 1>();
  
  // Number of loads from DDR to read a full column of an input matrix
  static constexpr int LoadItersPerColumn = rows/NumElementsPerBank + ExtraIter;

  // Number of banks required to hold all the data from the R matrix
  static constexpr int NumRBanks = RMatrixSize/NumElementsPerBank + ExtraBank;

  // The indexes kLoadIter and kStoreIter iterators are being divided 
  // by kNumBanks. So we precompute the size of the output.
  static constexpr int LiNumBankBitSize = LoadIterBitSize - Log2(NumBanks);
  static constexpr int SiNumBankBitSize = StoreIterBitSize - Log2(NumBanks);

  static constexpr int NValue = columns;

  // Number of iterations performed without any dummy work added for the 
  // triangular loop optimization
  static constexpr int VariableIterations = NValue - RAWLatency;
  // Total number of dummy iterations
  static constexpr int DummyIterations = RAWLatency > columns ?
              (columns - 1) * columns / 2 + (RAWLatency - columns) * columns :
              RAWLatency * (RAWLatency - 1) / 2;

  // Total number of iterations (including dummy iterations)
  static constexpr int Iterations = columns + columns * (columns+1) / 2 +  
                                                              DummyIterations;

  // Sizes in bits for the triangular loop indexes
  // i starts from -1 and goes up to rows
  // So we need:
  // -> enough bits to encode rows+1 for the positive iterations and 
  //    the exit condition
  // -> one extra bit for the -1
  static constexpr int IBitSize = BitsForMaxValue<rows + 1>() + 1;

  
  // j starts from i, so from -1 and goes up to columns
  // So we need:
  // -> enough bits to encode columns+1 for the positive iterations and 
  //    the exit condition
  // -> one extra bit for the -1
  // But j may start below -1 if we perform more dummy iterations than the 
  // number of columns in the matrix.
  // In that case, we need:
  // -> enough bits to encode columns+1 for the positive iterations and 
  //    the exit condition
  // -> enough bits to encode the maximum number of negative iterations
  static constexpr int JNegativeIterations = 
                          VariableIterations < 0 ? -VariableIterations : 1;

  static constexpr int JBitSize = BitsForMaxValue<columns + 1>() 
                                    + BitsForMaxValue<JNegativeIterations>();

};