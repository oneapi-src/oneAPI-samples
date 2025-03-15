//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

template <typename fp> fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}

//
// helpers for initializing templated scalar data type values.
//
template <typename fp>
fp set_fp_value(fp arg1, fp arg2 = 0.0)
{
    return arg1;
}

template <typename fp>
std::complex<fp> set_fp_value(std::complex<fp> arg1, std::complex<fp> arg2 = 0.0){
    return std::complex<fp>(arg1.real(), arg2.real());
}


// Create the 3arrays CSR representation (ia, ja, values)
// initialized by a stencil-based matrix with size nx=ny=nz
// with 27 point finite difference stencil for the laplacian
template <typename fp, typename intType>
void generate_sparse_matrix(const intType nx, 
                                  intType *ia, 
                                  intType *ja, 
                                  fp *a,
                            const intType index = 0)
{
    intType nz = nx, ny = nx;
    intType nnz = 0;
    intType current_row;

    ia[0] = index;

    for (intType iz = 0; iz < nz; iz++) {
        for (intType iy = 0; iy < ny; iy++) {
            for (intType ix = 0; ix < nx; ix++) {

                current_row = iz * nx * ny + iy * nx + ix;

                for (intType sz = -1; sz <= 1; sz++) {
                    if (iz + sz > -1 && iz + sz < nz) {
                        for (intType sy = -1; sy <= 1; sy++) {
                            if (iy + sy > -1 && iy + sy < ny) {
                                for (intType sx = -1; sx <= 1; sx++) {
                                    if (ix + sx > -1 && ix + sx < nx) {
                                        intType current_column =
                                                current_row + sz * nx * ny + sy * nx + sx;
                                        ja[nnz] = current_column + index;
                                        if (current_column == current_row) {
                                            a[nnz++] = set_fp_value(fp(26.0), fp(0.0));
                                        }
                                        else {
                                            a[nnz++] = set_fp_value(fp(-1.0), fp(0.0));
                                        }
                                    } // end
                                      // x
                                      // bounds
                                      // test
                                }     // end sx loop
                            }         // end y bounds test
                        }             // end sy loop
                    }                 // end z bounds test
                }                     // end sz loop
                ia[current_row + 1] = nnz + index;

            } // end ix loop
        }     // end iy loop
    }         // end iz loop
}




