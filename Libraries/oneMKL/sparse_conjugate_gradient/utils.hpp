//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

template <typename fp> fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}

// Create the 3arrays CSR representation (ia, ja, values)
// iniitialized by a stencil-based matrix with size nx=ny=nz
template <typename fp, typename intType>
void generate_sparse_matrix(const intType nx,
                            std::vector<intType> &ia,
                            std::vector<intType> &ja,
                            std::vector<fp> &a)
{
    intType nz = nx, ny = nx;
    intType nnz = 0;
    intType current_row;

    ia[0] = 0;

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
                                        ja[nnz] = current_column;
                                        if (current_column == current_row) {
                                            a[nnz++] = 26.;
                                        }
                                        else {
                                            a[nnz++] = -1.;
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
                ia[current_row + 1] = nnz;

            } // end ix loop
        }     // end iy loop
    }         // end iz loop
}

