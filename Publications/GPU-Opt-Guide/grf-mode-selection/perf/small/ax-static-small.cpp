#include <omp.h>
#include <stdio.h>

#define ELEM 32000
#define NX1 12
#define ZDIM 6

#define I2(i, j) (j * NX1 + i)
#define I3(i, j, k) (k * NX1 * NX1 + j * NX1 + i)
#define I4(i, j, k, e) (e * NX1 * NX1 * NX1 + k * NX1 * NX1 + j * NX1 + i)

int c_ax_v80_static(int nelt, int nx, double *w, double *u, double *g,
                    double *D) {

  int innerub = ZDIM * nx * nx;

// start OpenMP offload region
#pragma omp target teams distribute thread_limit(ZDIM *NX1 *NX1)
  for (int e = 0; e < nelt; e++) {
    double s_u[NX1 * NX1 * NX1];
    double s_D[NX1 * NX1];
    // SLM used for the three arrays here
    double s_ur[NX1 * NX1 * NX1];
    double s_us[NX1 * NX1 * NX1];
    double s_ut[NX1 * NX1 * NX1];

#pragma omp parallel for
    for (int inner = 0; inner < innerub; inner++) {
      int k = inner / (NX1 * NX1);
      int j = (inner - k * NX1 * NX1) / NX1;
      int i = inner - k * NX1 * NX1 - j * NX1;
      if (k == 0)
        s_D[I2(i, j)] = D[I2(i, j)];
      for (; k < NX1; k += ZDIM) {
        s_u[I3(i, j, k)] = u[I4(i, j, k, e)];
      }
    }

#pragma omp parallel for
    for (int inner = 0; inner < innerub; inner++) {
      int k = inner / (NX1 * NX1);
      int j = (inner - k * NX1 * NX1) / NX1;
      int i = inner - k * NX1 * NX1 - j * NX1;

      double r_G00, r_G01, r_G02, r_G11, r_G12, r_G22;

      for (; k < NX1; k += ZDIM) {
        double r_ur, r_us, r_ut;
        r_ur = r_us = r_ut = 0;
#ifdef FORCE_UNROLL
#pragma unroll NX1
#endif
        for (int m = 0; m < NX1; m++) {
          r_ur += s_D[I2(i, m)] * s_u[I3(m, j, k)];
          r_us += s_D[I2(j, m)] * s_u[I3(i, m, k)];
          r_ut += s_D[I2(k, m)] * s_u[I3(i, j, m)];
        }

        const unsigned gbase = 6 * I4(i, j, k, e);
        r_G00 = g[gbase + 0];
        r_G01 = g[gbase + 1];
        r_G02 = g[gbase + 2];
        s_ur[I3(i, j, k)] = r_G00 * r_ur + r_G01 * r_us + r_G02 * r_ut;
        r_G11 = g[gbase + 3];
        r_G12 = g[gbase + 4];
        s_us[I3(i, j, k)] = r_G01 * r_ur + r_G11 * r_us + r_G12 * r_ut;
        r_G22 = g[gbase + 5];
        s_ut[I3(i, j, k)] = r_G02 * r_ur + r_G12 * r_us + r_G22 * r_ut;
      }
    }

#pragma omp parallel for
    for (int inner = 0; inner < innerub; inner++) {
      int k = inner / (NX1 * NX1);
      int j = (inner - k * NX1 * NX1) / NX1;
      int i = inner - k * NX1 * NX1 - j * NX1;
      for (; k < NX1; k += ZDIM) {
        double wr = 0.0;
        for (int m = 0; m < NX1; m++) {
          double s_D_i = s_D[I2(m, i)];
          double s_D_j = s_D[I2(m, j)];
          double s_D_k = s_D[I2(m, k)];
          wr += s_D_i * s_ur[I3(m, j, k)] + s_D_j * s_us[I3(i, m, k)] +
                s_D_k * s_ut[I3(i, j, m)];
        }
        w[I4(i, j, k, e)] = wr;
      }
    }
  }
  // end OpenMP offload region
  return 0;
}

int main(void) {
#define ASIZE (ELEM * NX1 * NX1 * NX1)

  static double w[ASIZE], u[ASIZE], g[6 * ASIZE], D[NX1 * NX1];

  int nelt = ELEM;
  int nx = NX1;

  omp_set_default_device(0);

  for (int i = 0; i < nx * nx; i++)
    D[i] = 1.0;

  for (size_t i = 0; i < ASIZE; i++) {
    w[i] = 0.0;
    u[i] = 1.0;
    for (int j = 0; j < 6; j++)
      g[i * 6 + j] = 1.0;
  }

#pragma omp target enter data map(alloc                                        \
                                  : w [0:ASIZE], u [0:ASIZE], g [0:6 * ASIZE], \
                                    D [0:NX1 * NX1])
#pragma omp target update to(w [0:ASIZE], u [0:ASIZE], g [0:6 * ASIZE],        \
                             D [0:NX1 * NX1])

  for (int i = 0; i < 100; i++)
    c_ax_v80_static(nelt, nx, w, u, g, D);

#pragma omp target exit data map(delete                                        \
                                 : w [0:ASIZE], u [0:ASIZE], g [0:6 * ASIZE],  \
                                   D [0:NX1 * NX1])
  return 0;
}
