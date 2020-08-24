// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

#ifndef __CRR_COMMON_H__
#define __CRR_COMMON_H__

constexpr int kMaxStringLen = 1024;

// Increments of kMaxNSteps
constexpr size_t kMaxNSteps  = 8189;
constexpr size_t kMaxNSteps1 = 8190;
constexpr size_t kMaxNSteps2 = 8191;
constexpr size_t kMaxNSteps3 = 8192;

// Increment by a small epsilon in order to compute derivative 
// of option price with respect to Vol or Interest. The derivatives
// are then used to compute Vega and Rho. 
constexpr double kEpsilon  = 0.0001;

// Whenever calculations are made for Option Price 0, need to increment
// nsteps by 2 to ensure all the required derivative prices are calculated.
constexpr size_t kOpt0 = 2;


// Solver configuration settings that are dependent on selected
// board. Most notable settings are:

// OUTER_UNROLL controls the number of CRRs that can be processed
// in parallel in a SIMD fashion (number of CRRS must be >= OUTER_UNROLL). 
// This is ideally a power of two, but does not have to be. Since 
// the DRAM bandwidth requirement is low, increasing OUTER_UNROLL 
// should result in fairly linear speedup. (max: 32 on PAC A10)

// INNER_UNROLL controls the degree of parallelization within
// the calculation of a single CRR. This must be a power of two. Increasing
// INNER_UNROLL has a lower area overhead than increasing OUTER_UNROLL;
// however, there are diminishing returns as INNER_UNROLL is increased with
// respect to the number of time steps. (max: 128 on PAC A10)


// Data structure for original input data.
typedef struct {
  int cp;         /* cp = -1 or 1 for Put & Call respectively. */
  double n_steps; /* n_steps = number of time steps in the binomial tree. */
  double strike;  /* strike = exercise price of option. */
  double spot;    /* spot = spot price of the underlying. */
  double fwd;     /* fwd = forward price of the underlying. */
  double vol;     /* vol = per cent volatility, input as a decimal. */
  double df;      /* df = discount factor to option expiry. */
  double t;       /* t = time in years to the maturity of the option. */

} InputData;

// Data structure as the inputs to FPGA.
// Element[i] is used to compute option_price[i]. 
typedef struct {
  double n_steps;   /* n_steps = number of time steps in the binomial tree. */
  double u[3];      /* u = the increase factor of a up movement in the binomial tree,
                       same for each time step. */
  double u2[3];     /* u2 = the square of increase factor. */
  double c1[3];     /* c1 = the probality of a down movement in the binomial tree,
                       same for each time step. */
  double c2[3];     /* c2 = the probality of a up movement in the binomial tree. */
  double umin[3];   /* umin = minimum price of the underlying at the maturity. */
  double param_1[3];/* param_1[i] = cp * umin[i] */ 
  double param_2;   /* param_2 = cp * strike */

} CRRInParams;

// Data structure as the output from ProcessKernelResult().
typedef struct {
  double pgreek[4]; /* Stores the 4 derivative prices in the binomial tree 
                       required to compute the Premium and Greeks. */
  double vals[3];   /* Three option prices calculated */

} InterRes;

// Data structure for option price and five Greeks.
typedef struct {
  double value; /* value = option price. */
  double delta;
  double gamma;
  double vega;
  double theta;
  double rho;
} OutputRes;

// Data structures required by the kernel
typedef struct {
  double u;
  double c1;
  double c2;
  double param_1;
  double param_2;
  short n_steps;
  short pad1;
  int pad2;
  double pad3;
  double pad4;
} CRRMeta;

typedef struct {
  double u2;
  double p1powu;
  double init_optval;
  double pad;
} ArrayEle;

typedef struct {
  ArrayEle array_eles[kMaxNSteps3][3]; /* Second dimension size set to 3 to have a 
                                          separate ArrayEle for each option price */
} CRRArrayEles;

typedef struct {
  ArrayEle array_eles[kMaxNSteps3];
} CRRPerStepMeta;

typedef struct {
  double pgreek[4];
  double optval0;
  double pad[3];
} CRRResParams;

#endif
