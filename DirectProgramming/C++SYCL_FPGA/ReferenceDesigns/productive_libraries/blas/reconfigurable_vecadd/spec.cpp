#include "Halide.h"
#include "parameters.h"

using namespace Halide;

int main()
{
    // Indices. b is an additional loop for batch processing of dot products
    #define P       kk, k, b
    // Linearized addresses
    #define total_k (kk + KK * k)

    // Outer loop bounds, which are determined by input sizes
    #define K ((X.dim(0).extent() + KK - 1) / KK)
    #define B (X.dim(1).extent())

    // Inputs. X and Y are vectors, but we add an outer dimension to give us the flexibility of testing performance in batch mode.
    ImageParam X("X", TTYPE, 2);
    ImageParam Y("Y", TTYPE, 2);
    Param<int> IncX("IncX");
    Param<int> IncY("IncY");
    Param<CONST_TYPE> Alpha("Alpha"), Beta("Beta");

    X.dim(0).set_stride(IncX);
    Y.dim(0).set_stride(IncY);

    // UREs
    Var kk("kk"), k("k"), b("b");
    URE uY("uY", TTYPE, {P}), uX("uX", TTYPE, {P}), uZ_1("uZ_1", TTYPE, {P}), Z("Z");

    Expr Load_X = X(total_k, b);
    Expr Load_Y = Y(total_k, b);

    uX(P)   = Load_X;
    uY(P)   = Load_Y;
    uZ_1(P) = Alpha * uX(P) + Beta * uY(P);
    Z(P)    = select(true, uZ_1(P));

    // Put all the UREs inside the same loop nest.
    uX.merge_ures(uY, uZ_1, Z);

    // Explicitly set the loop bounds
    uX.set_bounds(kk, 0, KK, k, 0, K)
      .set_bounds(b,  0, B);
    uX.vectorize(kk);

    // I/O network. On the device side, DX, DY and DZ are responsible for I/O. On the host side, Load_X, Load_Y, and Out are responsible for I/O.
    Stensor DX("xLoader", DRAM), DY("yLoader", DRAM), DZ("unloader", DRAM), Out("deserializer");
    Load_X >> DX >> FIFO(256);
    Load_Y >> DY >> FIFO(256);
    Z >> FIFO(256) >> DZ >> Out;

    // Compile the kernel to an FPGA bitstream, and expose a C interface for the host to invoke
    Out.compile_to_oneapi(OUTPUT_FILE, {Alpha, X, IncX, Beta, Y, IncY}, KERNEL, IntelFPGA);
    printf("Success\n");
    return 0;
}
