// This is a specification for the following computation:
//    C = alpha * op(A) * op(B) + beta * C
// where A, B  and C might be general, symmetric or Hermitian matrices, and op(X) can be X, transpose of X, or conjugate transpose of X.

#include "Halide.h"

// Constant parameters and data types of the systolic array
#include "./parameters.h"
using namespace Halide;

int main()
{
    // Loop indices. Loop iii, ii, and i iterate the output matrix's rows; loop jjj, jj, and j iterate the output matrix's columns;
    // loop kkk, kk, and k iterate the reduction dimension. The outermost loop j and i iterate the output matrix in "tiles".
    #define P                  kkk,      jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kkk_minus_1      kkk-1,    jjj,  iii,  jj, ii, kk,     k,  j,i
    #define P_kk_minus_1       kkk+KKK-1,jjj,  iii,  jj, ii, kk-1,   k,  j,i
    #define P_k_minus_1        kkk+KKK-1,jjj,  iii,  jj, ii, kk+KK-1,k-1,j,i
    #define P_jjj_minus_1      kkk,      jjj-1,iii,  jj, ii, kk,     k,  j,i
    #define P_iii_minus_1      kkk,      jjj,  iii-1,jj, ii, kk,     k,  j,i
    #define P_reduced                    jjj,  iii,  jj, ii,             j,i // The order of the resulting data after reduction is done
    #define P_reorder                    jjj,        jj, ii, iii,        j,i // The order of the resulting data when writing to the device DRAM

    // The location of the C element under reduction, and the step of the reduction
    #define C_row_idx          (iii + III * ii + III * II * i)
    #define C_col_idx          (jjj + JJJ * jj + JJJ * JJ * j)
    #define reduction_idx      (kkk + KKK * kk + KKK * KK * k)

    // Matrices' dimensions. In Halide style, column is the first dimension, and row the second. It is up to the user to ensure that the matrices' dimensions match,
    // for example, when there is no transpose, matrix A's rows should equal to matrix C's rows, matrix A's columns should equal to matrix B's rows, and matrix B's
    // columns should equal to matrix C's columns.
    #define C_COLS             (C.dim(0).extent())
    #define C_ROWS             (C.dim(1).extent())
    #define A_COLS             (A.dim(0).extent())
    #define A_ROWS             (A.dim(1).extent())
    #define B_COLS             (B.dim(0).extent())
    #define B_ROWS             (B.dim(1).extent())
    #define REDUCTIOIN_LEN     select(TransposeA, A_ROWS, A_COLS)

    // Outer loop bounds, which are determined by the matrices' dimensions.
    #define I                  ((C_ROWS    + (III * II - 1)) / (III * II))
    #define J                  ((C_COLS    + (JJJ * JJ - 1)) / (JJJ * JJ))
    #define K                  ((REDUCTIOIN_LEN + (KKK * KK - 1)) / (KKK * KK))

    // Are we accessing the matrices in their valid ranges? As can be seen above, the outer loop bounds are determined by rounding, thus the
    // addresses of the matrices can be out of valid ranges. In this design, for memory efficiency, the reduction dimension is loaded in KKK-wide vectors,
    // and the column dimension of the output matrix C is stored in JJJ-wide vectors. It is up to the user to ensure that the reduction dimension is
    // a multiple of KKK, and the column dimension of C is a multiple of JJJ.
    #define REDUCTION_IN_RANGE (KKK * (kk + KK * k) < REDUCTIOIN_LEN)
    #define COL_IN_RANGE       (JJJ * (jj + JJ * j) < C_COLS)
    #define ROW_IN_RANGE       (C_row_idx   < C_ROWS)

    // Parameters.
    Param<bool> TransposeA("TransposeA"), ConjugateA("ConjugateA"); // Is matrix A to be transposed? Is it to be conjugated?
    Param<bool> SymmetricA("SymmetricA"), HermitianA("HermitianA"); // Is matrix A symmetric? Is it Hermitian?
    Param<bool> UpA("UpA");                                         // Given matrix A as symmetric or Hermitian, is its upper triangle stored?
    Param<bool> TransposeB("TransposeB"), ConjugateB("ConjugateB");
    Param<bool> SymmetricB("SymmetricB"), HermitianB("HermitianB"), UpB("UpB");
    Param<bool> SymmetricC("SymmetricC"), HermitianC("HermitianC"), UpC("UpC");
    Param<bool> HalfSpaceOut("HalfSpaceOut"); // Compute only half output? This is true when the output is symmetric or Hermitian. In this case, we compute
                                              // only the upper triangle of the output, in terms of tiles. For the tiles crossing the diagonal, we ensure
                                              // the correctness of only their data above or on the diagonal.
    Param<TS>   alpha("alpha"), beta("beta");
    ImageParam  A("A", TA, 2), B("B", TB, 2), C("C", TC, 2);

    // UREs
    Var kkk("kkk"), jjj("jjj"), iii("iii"), jj("jj"), ii("ii"), kk("kk"), k("k"), j("j"), i("i");
    URE X("X", TC, {P}), Y("Y", TC, {P}), Z("Z", TC, {P}), Product("Product");
    URE Add("Add", TC, {P_reorder}), Out("Out", TC, {P_reorder});

    // Logically, the location of the matrix A to read from
    Expr A_col_idx                 = select(TransposeA, C_row_idx,     reduction_idx);
    Expr A_row_idx                 = select(TransposeA, reduction_idx, C_row_idx);
    // Physically, when A is symmetric/Hermitian, if its upper triangle is stored and we want to read the lower triangle or
    // if its lower triangle is stored and we want to read the upper triangle, we have to read from the symmetric location.
    Expr Read_Upper_A              = (A_row_idx <= A_col_idx);
    Expr Read_A_from_symmetric_loc = (SymmetricA || HermitianA) && (UpA != Read_Upper_A);
    // The actual location of A to read.
    Expr A_actual_col_idx          = select(Read_A_from_symmetric_loc, A_row_idx, A_col_idx);
    Expr A_actual_row_idx          = select(Read_A_from_symmetric_loc, A_col_idx, A_row_idx);
    // If we read from the symmetric position, we might need conjugate the value read.
    Expr Do_conjugate_A            = select(HermitianA, Read_A_from_symmetric_loc != ConjugateA, ConjugateA);
    Expr A_value                   = conditional_conjugate(Do_conjugate_A, A(A_actual_col_idx, A_actual_row_idx));
    // If in range, read; otherwise, pad 0.
    Expr Check_Load_A              = select(ROW_IN_RANGE && REDUCTION_IN_RANGE, A_value, ZERO);

    // B is read similarly
    // Logically, the location of the matrix B to read from
    Expr B_col_idx                 = select(TransposeB, reduction_idx, C_col_idx);
    Expr B_row_idx                 = select(TransposeB, C_col_idx,     reduction_idx);
    // Physically, when B is symmetric/Hermitian, if its upper triangle is stored and we want to read the lower triangle or
    // if its lower triangle is stored and we want to read the upper triangle, we have to read from the symmetric location.
    Expr Read_Upper_B              = (B_row_idx <= B_col_idx);
    Expr Read_B_from_symmetric_loc = (SymmetricB || HermitianB) && (UpB != Read_Upper_B);
    // The actual location of B to read.
    Expr B_actual_col_idx          = select(Read_B_from_symmetric_loc, B_row_idx, B_col_idx);
    Expr B_actual_row_idx          = select(Read_B_from_symmetric_loc, B_col_idx, B_row_idx);
    // If we read from the symmetric position, we might need conjugate the value read.
    Expr Do_conjugate_B            = select(HermitianB, Read_B_from_symmetric_loc != ConjugateB, ConjugateB);
    Expr B_value                   = conditional_conjugate(Do_conjugate_B, B(B_actual_col_idx, B_actual_row_idx));
    // If in range, read; otherwise, pad 0.
    Expr Check_Load_B              = select(REDUCTION_IN_RANGE && COL_IN_RANGE, B_value, ZERO);

    X(P) = select(jjj == 0, Check_Load_A, X(P_jjj_minus_1));
    Y(P) = select(iii == 0, Check_Load_B, Y(P_iii_minus_1));
    Z(P) = select(k == 0 && kk == 0 && kkk == 0, ZERO,
                select(kkk == 0, select(kk == 0, Z(P_k_minus_1), Z(P_kk_minus_1)), Z(P_kkk_minus_1)))
                + X(P) * Y(P);
    Product(P_reduced) = select(k == K-1 && kk == KK-1 && kkk == KKK-1, Z(P));

    // C is read similarly
    // Physically, when C is symmetric/Hermitian, if its upper triangle is stored and we want to read the lower triangle or
    // if its lower triangle is stored and we want to read the upper triangle, we have to read from the symmetric location.
    Expr Read_Upper_C              = (C_row_idx <= C_col_idx);
    Expr Read_C_from_symmetric_loc = (SymmetricC || HermitianC) && (UpC != Read_Upper_C);
    // The actual location of C to read.
    Expr C_actual_col_idx          = select(Read_C_from_symmetric_loc, C_row_idx, C_col_idx);
    Expr C_actual_row_idx          = select(Read_C_from_symmetric_loc, C_col_idx, C_row_idx);
    // If we read from the symmetric position, we might need conjugate the value read.
    Expr Do_conjugate_C            = (HermitianC && Read_C_from_symmetric_loc);
    Expr C_value                   = conditional_conjugate(Do_conjugate_C, C(C_actual_col_idx, C_actual_row_idx));
    // If in range, read; otherwise, pad 0.
    Expr Check_Load_C              = select(beta != SCALAR_ZERO && ROW_IN_RANGE && COL_IN_RANGE, beta * C_value, ZERO);

    Add(P_reorder) = alpha * Product(P_reorder) + Check_Load_C;
    Out(P_reorder) = select(true, Add(P_reorder));

    // Put the UREs that compute the product into a loop nest.
    X.merge_ures(Y, Z, Product);
    // Put the UREs that compute the final sum into another loop nest.
    Add.merge_ures(Out);

    // Explicitly set the loop bounds: every loop has a min, and an extent (number of iterations).
    // If HalfSpaceOut is true, we generate the upper triangular output. However, since every invocation of the systolic array produces a rectangular tile,
    // the output is "upper triangular" only in terms of tiles. For the tiles at the diagonal, post-processing is needed to remove values that are within the tiles
    // but are below the diagonal.
    X.set_bounds(jjj, 0, JJJ, iii, 0, III, kkk, 0, KKK)
     .set_bounds(jj,  0, JJ,  ii,  0, II,  kk,  0, KK)
     .set_bounds(j,   select(HalfSpaceOut, i, 0), select(HalfSpaceOut, J-i, J))
     .set_bounds(i,   0, I,   k,   0, K);

    Add.set_bounds(jjj, 0, JJJ, iii, 0, III)
       .set_bounds(jj,  0, JJ,  ii,  0, II)
       .set_bounds(j,   select(HalfSpaceOut, i, 0), select(HalfSpaceOut, J-i, J))
       .set_bounds(i,   0, I);

    // Create a systolic array
    X.space_time_transform(jjj, iii).run_forever();
    Add.vectorize(jjj);

    // I/O network.
    Stensor DA("DA", DRAM), SA("SA", SRAM), DB("DB", DRAM), SB("SB", SRAM), DC("DC", DRAM);
    Stensor RCollector("RCollector", REG), SCollector("SCollector", SRAM), DOut("DOut", DRAM), Output("Output");
    Check_Load_A >> DA.out(kkk) >> FIFO(256) >> SA.scope(k).out(kkk, iii) >> FIFO(256);
    Check_Load_B >> DB.out(kkk) >> FIFO(256) >> SB.scope(k).out(kkk, jjj) >> FIFO(256);
    Check_Load_C >> DC.out(jjj) >> FIFO(256);
    Product >> RCollector.scope(iii).out(jjj) >> FIFO(256) >> SCollector >> FIFO(256);
    Out >> FIFO(256) >> DOut >> Output(C_col_idx, C_row_idx);

    // Compile the above specification and generate an oneAPI/SYCL file, with a C interface for the host to invoke
    Output.compile_to_oneapi(OUTPUT_FILE, {A, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                           B, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                           C,                         SymmetricC, HermitianC, UpC,
                                           HalfSpaceOut, alpha, beta}, KERNEL, IntelFPGA);

    return 0;
}
