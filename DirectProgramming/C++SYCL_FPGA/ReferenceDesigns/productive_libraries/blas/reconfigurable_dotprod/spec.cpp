#include "Halide.h"
#include "parameters.h"

using namespace Halide;

int main()
{
    // Indices. b is an additional loop for batch processing of dot products
    #define P_1             kkk,             kk,      k,     b
    #define P_1_k_minus_1   kkk + KKK - 1,   kk,      k - 1, b
    #define P_1_kkk_minus_1 kkk - 1,         kk,      k,     b
    #define P_2                              kk,             b
    #define P_2_kk_minus_1                   kk - 1,         b
    #define P_out                                            b
    // Linearized addresses
    #define total_k         (kkk + KKK * kk + KKK * KK * k)

    // Outer loop bounds, which are determined by input sizes
    #define K ((X.dim(0).extent() + KK * KKK - 1) / (KK * KKK))
    #define B (X.dim(1).extent())

    #define addr_in_range (KKK * (kk + KK * k) < X.dim(0).extent())

    // Inputs. X and Y are vectors, but we add an outer dimension to give us the flexibility of testing performance in batch mode.
    ImageParam X("X", ITYPE, 2);
    ImageParam Y("Y", ITYPE, 2);
    Param<int> IncX("IncX");
    Param<int> IncY("IncY");
    Param<bool> ConjugateX("ConjugateX");
    Param<bool> SignBitY("SignBitY");
    Param<bool> SqrtRet("SqrtRet");

    X.dim(0).set_stride(IncX);
    Y.dim(0).set_stride(IncY);

    // UREs
    Var kkk("kkk"), kk("kk"), k("k"), b("b");
    URE uY("uY", TTYPE, {P_1}), uX("uX", TTYPE, {P_1}), uZ_1("uZ_1", TTYPE, {P_1}), Z("Z");
    URE uZ_2("uZ_2", TTYPE, {P_2}), Out("Out");

    Expr Check_Load_X = select(addr_in_range, conditional_conjugate(ConjugateX, cast(TTYPE, X(total_k, b))), 0);
    Expr Check_Load_Y = select(addr_in_range, conditional_signbit(SignBitY, cast(TTYPE, Y(total_k, b))), 0);

    // Divide each input into KK parts, each part with KKK*K elements. Calculate the dot products of the KK parts interleavingly:
    // reduce KKK elements of part 1, then reduce KKK elements of part 2, ..., and finally, reduce KKK elements of part KK;
    // then go back to part 1: reduce the next KKK elements of part 1 (This reduction can start immediately because the previous
    // result of part 1 must be available: the interleaving with the reduction of the other parts has introduced enough latency),
    // then reduce the next KKK elements of part 2, ..., and finally, reduce the KKK elements of part KK; then go back to part 1
    // again, until all parts are fully reduced.

    // First, calculate the dot product for every part, indexed by loop kk.
    uX(P_1) = Check_Load_X;
    uY(P_1) = Check_Load_Y;
    uZ_1(P_1) = select(k == 0 && kkk == 0, 0, select(kkk == 0, uZ_1(P_1_k_minus_1), uZ_1(P_1_kkk_minus_1))) + uX(P_1) * uY(P_1);
    Z(P_2) = select(k == K - 1 && kkk == KKK - 1, uZ_1(P_1));

    // Second, sum up the dot product of all the parts.
    uZ_2(P_2) = select(kk == 0, 0, uZ_2(P_2_kk_minus_1)) + Z(P_2);
    Out(P_out) = select(kk == KK - 1, conditional_sqrt(SqrtRet, uZ_2(P_2)));

    // Put the first set of UREs into a loop nest.
    uX.merge_ures(uY, uZ_1, Z);
    // Put the second set of UREs into another loop nest
    uZ_2.merge_ures(Out);
    // Merge the two loop nests at the shared batch loop b
    uX.late_fuse(uZ_2, b);

    // Explicitly set the loop bounds
    uX.set_bounds(kkk,  0, KKK, kk,  0, KK,  k,  0, K)
      .set_bounds(b,    0, B);
    uZ_2.set_bounds(kk,  0, KK)
        .set_bounds(b,   0, B);

    // Create a systolic array with KKK PEs running synchronously.
    uX.space_time_transform(kkk);
    uX.vectorize(kkk);

    // I/O network. On the device side, DX, DY and DOut are responsible for I/O. On the host side, Check_Load_X, Check_Load_Y, and Output are responsible for I/O.
    Stensor DX("xLoader", DRAM), DY("yLoader", DRAM), DOut("unloader", DRAM), Output("deserializer");
    Check_Load_X >> DX.out(kkk) >> FIFO(256);
    Check_Load_Y >> DY.out(kkk) >> FIFO(256);
    Out >> FIFO(256) >> DOut >> Output;

    Output.compile_to_oneapi(OUTPUT_FILE, {ConjugateX, X, IncX, SignBitY, Y, IncY, SqrtRet}, KERNEL, IntelFPGA);
    printf("Success\n");
    return 0;
}
