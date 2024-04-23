/* Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <dpct/rng_utils.hpp>

#include <dpct/dpl_utils.hpp>

#include <cmath>

#ifdef WITH_FULL_W_MATRIX
#define R_W_MATRICES_SMEM_SLOTS 15
#else
#define R_W_MATRICES_SMEM_SLOTS 12
#endif

#ifndef NUM_PATHS
#define NUM_PATHS 25
#endif

#ifndef NUM_TIMESTEPS
#define NUM_TIMESTEPS 252
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1009:13: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define CHECK_CUDA(call) do {                                                  \
    call;                                                                      \
                                                                               \
  } while (0)

#define CHECK_CURAND(call) do {                                                \
    int status = call;                                                         \
    if (status != 0) {                                                         \
      fprintf(stderr, "CURAND Error at line %d in %s: %d\n", __LINE__,         \
              __FILE__, status);                                               \
      exit((int)status);                                                       \
    }                                                                          \
  } while (0)

// ====================================================================================================================

#define HOST_DEVICE
#define HOST_DEVICE_INLINE __dpct_inline__

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1011:6: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

HOST_DEVICE_INLINE sycl::double3 operator+(const sycl::double3 &u,
                                           const sycl::double3 &v)
{
  return sycl::double3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}
} // namespace dpct_operator_overloading

// ====================================================================================================================

/*
DPCT1011:7: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

HOST_DEVICE_INLINE sycl::double4 operator+(const sycl::double4 &u,
                                           const sycl::double4 &v)
{
  return sycl::double4(u.x() + v.x(), u.y() + v.y(), u.z() + v.z(),
                       u.w() + v.w());
}
} // namespace dpct_operator_overloading

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct PayoffCall
{
  double m_K;
  HOST_DEVICE_INLINE PayoffCall(double K) : m_K(K) {}
  HOST_DEVICE_INLINE double operator()(double S) const {
    return std::max(S - m_K, 0.0);
  }
  HOST_DEVICE_INLINE int is_in_the_money(double S) const { return S > m_K; }
};

struct PayoffPut
{
  double m_K;
  HOST_DEVICE_INLINE PayoffPut(double K) : m_K(K) {}
  HOST_DEVICE_INLINE double operator()(double S) const {
    return std::max(m_K - S, 0.0);
  }
  HOST_DEVICE_INLINE int is_in_the_money(double S) const { return S < m_K; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef WITH_ATOMIC_BETA
static __device__ __forceinline__ void atomic_add(double *address, double val)
{
  unsigned long long *address_as_ull = (unsigned long long *) address; 
  unsigned long long old = __double_as_longlong(address[0]), assumed; 
  do { 
    assumed = old; 
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
  } 
  while(assumed != old); 
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_BLOCK, typename Payoff >

void generate_paths_kernel(int num_timesteps, 
                           int num_paths, 
                           Payoff payoff,
                           double dt, 
                           double S0, 
                           double r, 
                           double sigma, 
                           const double *__restrict samples, 
                           double *__restrict paths,
                           const sycl::nd_item<3> &item_ct1)
{
  // The path generated by this thread.
  int path =
      item_ct1.get_group(2) * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2);

  // Early exit.
  if( path >= num_paths )
    return;
{
  
  // Compute (r - sigma^2 / 2).
  const double r_min_half_sigma_sq_dt = (r - 0.5*sigma*sigma)*dt;
  // Compute sigma*sqrt(dt).
  const double sigma_sqrt_dt = sigma * sycl::sqrt(dt);

  // Keep the previous price.
  double S = S0;

  // The offset.
  int offset = path;

#if USE_DEVICE_API
  oneapi::mkl::rng::device::mrg32k3a engine(0, (path * num_timesteps));
  oneapi::mkl::rng::device::gaussian<double> distr(0.0, 1.0);
  
  // Each thread generates several timesteps. 
  for( int timestep = 0 ; timestep < num_timesteps ; ++timestep, offset += num_paths )
  {
    auto res = oneapi::mkl::rng::device::generate(distr, engine);
    // The asset price.
    S = S * sycl::exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt * res);
    // Store the payoff at expiry.
    paths[offset] = timestep < num_timesteps - 1 ? S : payoff(S);
  }

#else
  // Each thread generates several timesteps. 
  int timestep = 0;
  for( ; timestep < num_timesteps-1 ; timestep++, offset += num_paths )
  {
    S = S * sycl::exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt * samples[timestep + path * num_timesteps]);
    paths[offset] = S;
  }

  // The asset price.
  S = S * sycl::exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt * samples[timestep + path * num_timesteps]);

  // Store the payoff at expiry.
  paths[offset] = payoff(S);
#endif
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1110:1: The total declared local variable size in device function assemble_R
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
static __dpct_inline__ void assemble_R(int m, sycl::double4 &sums,
                                       double *smem_svds)
{
  // Assemble R.

  double x0 = smem_svds[0];
  double x1 = smem_svds[1];
  double x2 = smem_svds[2];

  double x0_sq = x0 * x0;

  double sum1 = sums.x() - x0;
  double sum2 = sums.y() - x0_sq;
  double sum3 = sums.z() - x0_sq * x0;
  double sum4 = sums.w() - x0_sq * x0_sq;

  double m_as_dbl = (double) m;
  double sigma = m_as_dbl - 1.0;
  double mu = sycl::sqrt(m_as_dbl);
  double v0 = -sigma / (1.0 + mu);
  double v0_sq = v0*v0;
  double beta = 2.0 * v0_sq / (sigma + v0_sq);
  
  double inv_v0 = 1.0 / v0;
  double one_min_beta = 1.0 - beta;
  double beta_div_v0  = beta * inv_v0;
  
  smem_svds[0] = mu;
  smem_svds[1] = one_min_beta*x0 - beta_div_v0*sum1;
  smem_svds[2] = one_min_beta*x0_sq - beta_div_v0*sum2;
  
  // Rank update coefficients.
  
  double beta_div_v0_sq = beta_div_v0 * inv_v0;
  
  double c1 = beta_div_v0_sq*sum1 + beta_div_v0*x0;
  double c2 = beta_div_v0_sq*sum2 + beta_div_v0*x0_sq;

  // 2nd step of QR.
  
  double x1_sq = x1*x1;

  sum1 -= x1;
  sum2 -= x1_sq;
  sum3 -= x1_sq*x1;
  sum4 -= x1_sq*x1_sq;
  
  x0 = x1-c1;
  x0_sq = x0*x0;
  sigma = sum2 - 2.0*c1*sum1 + (m_as_dbl-2.0)*c1*c1;
  if (sycl::fabs(sigma) < 1.0e-16)
    beta = 0.0;
  else
  {
    mu = sycl::sqrt(x0_sq + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
    beta = 2.0*v0_sq / (sigma + v0_sq);
  }
  
  inv_v0 = 1.0 / v0;
  beta_div_v0 = beta * inv_v0;
  
  // The coefficient to perform the rank update.
  double c3 = (sum3 - c1*sum2 - c2*sum1 + (m_as_dbl-2.0)*c1*c2)*beta_div_v0;
  double c4 = (x1_sq-c2)*beta_div_v0 + c3*inv_v0;
  double c5 = c1*c4 - c2;
  
  one_min_beta = 1.0 - beta;
  
  // Update R. 
  smem_svds[3] = one_min_beta*x0 - beta_div_v0*sigma;
  smem_svds[4] = one_min_beta*(x1_sq-c2) - c3;
  
  // 3rd step of QR.
  
  double x2_sq = x2*x2;

  sum1 -= x2;
  sum2 -= x2_sq;
  sum3 -= x2_sq*x2;
  sum4 -= x2_sq*x2_sq;
  
  x0 = x2_sq-c4*x2+c5;
  sigma = sum4 - 2.0*c4*sum3 + (c4*c4 + 2.0*c5)*sum2 - 2.0*c4*c5*sum1 + (m_as_dbl-3.0)*c5*c5;
  if (sycl::fabs(sigma) < 1.0e-12)
    beta = 0.0;
  else
  {
    mu = sycl::sqrt(x0 * x0 + sigma);
    if( x0 <= 0.0 )
      v0 = x0 - mu;
    else
      v0 = -sigma / (x0 + mu);
    v0_sq = v0*v0;
    beta = 2.0*v0_sq / (sigma + v0_sq);
  }
  
  // Update R.
  smem_svds[5] = (1.0-beta)*x0 - (beta/v0)*sigma;
}

// ====================================================================================================================

static double off_diag_norm(double A01, double A02, double A12)
{
  return sycl::sqrt(2.0 * (A01 * A01 + A02 * A02 + A12 * A12));
}

// ====================================================================================================================

static __dpct_inline__ void swap(double &x, double &y)
{
  double t = x; x = y; y = t;
}

// ====================================================================================================================

/*
DPCT1110:2: The total declared local variable size in device function svd_3x3
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
static __dpct_inline__ void svd_3x3(int m, sycl::double4 &sums,
                                    double *smem_svds)
{
  // Assemble the R matrix.
  assemble_R(m, sums, smem_svds);

  // The matrix R.
  double R00 = smem_svds[0];
  double R01 = smem_svds[1];
  double R02 = smem_svds[2];
  double R11 = smem_svds[3];
  double R12 = smem_svds[4];
  double R22 = smem_svds[5];

  // We compute the eigenvalues/eigenvectors of A = R^T R.
  
  double A00 = R00*R00;
  double A01 = R00*R01;
  double A02 = R00*R02;
  double A11 = R01*R01 + R11*R11;
  double A12 = R01*R02 + R11*R12;
  double A22 = R02*R02 + R12*R12 + R22*R22;
  
  // We keep track of V since A = Sigma^2 V. Each thread stores a row of V.
  
  double V00 = 1.0, V01 = 0.0, V02 = 0.0;
  double V10 = 0.0, V11 = 1.0, V12 = 0.0;
  double V20 = 0.0, V21 = 0.0, V22 = 1.0;
  
  // The Jacobi algorithm is iterative. We fix the max number of iter and the minimum tolerance.
  
  const int max_iters = 16;
  const double tolerance = 1.0e-12;
  
  // Iterate until we reach the max number of iters or the tolerance.
 
  for( int iter = 0 ; off_diag_norm(A01, A02, A12) >= tolerance && iter < max_iters ; ++iter )
  {
    double c, s, B00, B01, B02, B10, B11, B12, B20, B21, B22;
    
    // Compute the Jacobi matrix for p=0 and q=1.
    
    c = 1.0, s = 0.0;
    if( A01 != 0.0 )
    {
      double tau = (A11 - A00) / (2.0 * A01);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t = sgn / (sgn * tau + sycl::sqrt(1.0 + tau * tau));

      c = 1.0 / sycl::sqrt(1.0 + t * t);
      s = t*c;
    }
    
    // Update A = J^T A J and V = V J.
    
    B00 = c*A00 - s*A01;
    B01 = s*A00 + c*A01;
    B10 = c*A01 - s*A11;
    B11 = s*A01 + c*A11;
    B02 = A02;
    
    A00 = c*B00 - s*B10;
    A01 = c*B01 - s*B11;
    A11 = s*B01 + c*B11;
    A02 = c*B02 - s*A12;
    A12 = s*B02 + c*A12;
    
    B00 = c*V00 - s*V01;
    V01 = s*V00 + c*V01;
    V00 = B00;
    
    B10 = c*V10 - s*V11;
    V11 = s*V10 + c*V11;
    V10 = B10;
    
    B20 = c*V20 - s*V21;
    V21 = s*V20 + c*V21;
    V20 = B20;
    
    // Compute the Jacobi matrix for p=0 and q=2.
    
    c = 1.0, s = 0.0;
    if( A02 != 0.0 )
    {
      double tau = (A22 - A00) / (2.0 * A02);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t = sgn / (sgn * tau + sycl::sqrt(1.0 + tau * tau));

      c = 1.0 / sycl::sqrt(1.0 + t * t);
      s = t*c;
    }
    
    // Update A = J^T A J and V = V J.
    
    B00 = c*A00 - s*A02;
    B01 = c*A01 - s*A12;
    B02 = s*A00 + c*A02;
    B20 = c*A02 - s*A22;
    B22 = s*A02 + c*A22;
    
    A00 = c*B00 - s*B20;
    A12 = s*A01 + c*A12;
    A02 = c*B02 - s*B22;
    A22 = s*B02 + c*B22;
    A01 = B01;
    
    B00 = c*V00 - s*V02;
    V02 = s*V00 + c*V02;
    V00 = B00;
    
    B10 = c*V10 - s*V12;
    V12 = s*V10 + c*V12;
    V10 = B10;
    
    B20 = c*V20 - s*V22;
    V22 = s*V20 + c*V22;
    V20 = B20;
    
    // Compute the Jacobi matrix for p=1 and q=2.
    
    c = 1.0, s = 0.0;
    if( A12 != 0.0 )
    {
      double tau = (A22 - A11) / (2.0 * A12);
      double sgn = tau < 0.0 ? -1.0 : 1.0;
      double t = sgn / (sgn * tau + sycl::sqrt(1.0 + tau * tau));

      c = 1.0 / sycl::sqrt(1.0 + t * t);
      s = t*c;
    }
    
    // Update A = J^T A J and V = V J.
    
    B02 = s*A01 + c*A02;
    B11 = c*A11 - s*A12;
    B12 = s*A11 + c*A12;
    B21 = c*A12 - s*A22;
    B22 = s*A12 + c*A22;
    
    A01 = c*A01 - s*A02;
    A02 = B02;
    A11 = c*B11 - s*B21;
    A12 = c*B12 - s*B22;
    A22 = s*B12 + c*B22;
    
    B01 = c*V01 - s*V02;
    V02 = s*V01 + c*V02;
    V01 = B01;
    
    B11 = c*V11 - s*V12;
    V12 = s*V11 + c*V12;
    V11 = B11;
    
    B21 = c*V21 - s*V22;
    V22 = s*V21 + c*V22;
    V21 = B21;
  }

  // Swap the columns to have S[0] >= S[1] >= S[2].
  if( A00 < A11 )
  {
    swap(A00, A11);
    swap(V00, V01);
    swap(V10, V11);
    swap(V20, V21);
  }
  if( A00 < A22 )
  {
    swap(A00, A22);
    swap(V00, V02);
    swap(V10, V12);
    swap(V20, V22);
  }
  if( A11 < A22 )
  {
    swap(A11, A22);
    swap(V01, V02);
    swap(V11, V12);
    swap(V21, V22);
  }

  //printf("timestep=%3d, svd0=%.8lf svd1=%.8lf svd2=%.8lf\n", blockIdx.x, sqrt(A00), sqrt(A11), sqrt(A22));
  
  // Invert the diagonal terms and compute V*S^-1.

  double inv_S0 = sycl::fabs(A00) < 1.0e-12 ? 0.0 : 1.0 / A00;
  double inv_S1 = sycl::fabs(A11) < 1.0e-12 ? 0.0 : 1.0 / A11;
  double inv_S2 = sycl::fabs(A22) < 1.0e-12 ? 0.0 : 1.0 / A22;

  // printf("SVD: timestep=%3d %12.8lf %12.8lf %12.8lf\n", blockIdx.x, sqrt(A00), sqrt(A11), sqrt(A22));
  
  double U00 = V00 * inv_S0; 
  double U01 = V01 * inv_S1; 
  double U02 = V02 * inv_S2;
  double U10 = V10 * inv_S0; 
  double U11 = V11 * inv_S1; 
  double U12 = V12 * inv_S2;
  double U20 = V20 * inv_S0; 
  double U21 = V21 * inv_S1; 
  double U22 = V22 * inv_S2;
  
  // Compute V*S^-1*V^T*R^T.
  
#ifdef WITH_FULL_W_MATRIX
  double B00 = U00*V00 + U01*V01 + U02*V02;
  double B01 = U00*V10 + U01*V11 + U02*V12;
  double B02 = U00*V20 + U01*V21 + U02*V22;
  double B10 = U10*V00 + U11*V01 + U12*V02;
  double B11 = U10*V10 + U11*V11 + U12*V12;
  double B12 = U10*V20 + U11*V21 + U12*V22;
  double B20 = U20*V00 + U21*V01 + U22*V02;
  double B21 = U20*V10 + U21*V11 + U22*V12;
  double B22 = U20*V20 + U21*V21 + U22*V22;
  
  smem_svds[ 6] = B00*R00 + B01*R01 + B02*R02;
  smem_svds[ 7] =           B01*R11 + B02*R12;
  smem_svds[ 8] =                     B02*R22;
  smem_svds[ 9] = B10*R00 + B11*R01 + B12*R02;
  smem_svds[10] =           B11*R11 + B12*R12;
  smem_svds[11] =                     B12*R22;
  smem_svds[12] = B20*R00 + B21*R01 + B22*R02;
  smem_svds[13] =           B21*R11 + B22*R12;
  smem_svds[14] =                     B22*R22;
#else
  double B00 = U00*V00 + U01*V01 + U02*V02;
  double B01 = U00*V10 + U01*V11 + U02*V12;
  double B02 = U00*V20 + U01*V21 + U02*V22;
  double B11 = U10*V10 + U11*V11 + U12*V12;
  double B12 = U10*V20 + U11*V21 + U12*V22;
  double B22 = U20*V20 + U21*V21 + U22*V22;
  
  smem_svds[ 6] = B00*R00 + B01*R01 + B02*R02;
  smem_svds[ 7] =           B01*R11 + B02*R12;
  smem_svds[ 8] =                     B02*R22;
  smem_svds[ 9] =           B11*R11 + B12*R12;
  smem_svds[10] =                     B12*R22;
  smem_svds[11] =                     B22*R22;
#endif
}

// ====================================================================================================================

template< int NUM_THREADS_PER_BLOCK, typename Payoff >

void prepare_svd_kernel(int num_paths, 
                        int min_in_the_money, 
                        Payoff payoff, 
                        const double */*__restrict*/ paths, 
                        int *__restrict all_out_of_the_money, 
                        double *__restrict svds,
                        const sycl::nd_item<3> &item_ct1,
                        // uint8_t *smem_storage_ct1,
                        double *smem_svds)
{
/*
  // We need to perform a scan to find the first 3 stocks pay off.

  // We need to perform a reduction at the end of the kernel to compute the final sums.

  // The union for the scan/reduce.
  union TempStorage
  {
    typename BlockScan   ::TempStorage for_scan;
    typename BlockReduce1::TempStorage for_reduce1;
    typename BlockReduce4::TempStorage for_reduce4;
  };
  TempStorage &smem_storage = *(TempStorage *)smem_storage_ct1;
*/
  // Shared memory.

  // Shared buffer for the ouput.

  // Each block works on a single timestep.
  const int timestep = item_ct1.get_group(2);
  // The timestep offset.
  const int offset = timestep * num_paths;

  // Sums.
  int m = 0; sycl::double4 sums = {0.0, 0.0, 0.0, 0.0};

  // Initialize the shared memory. DBL_MAX is a marker to specify that the value is invalid.
  if (item_ct1.get_local_id(2) < R_W_MATRICES_SMEM_SLOTS)
    smem_svds[item_ct1.get_local_id(2)] = 0.0;
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Have we already found our 3 first paths which pay off.
  int found_paths = 0;

  // Iterate over the paths.
  for (int path = item_ct1.get_local_id(2); path < num_paths;
       path += NUM_THREADS_PER_BLOCK)
  {
    // Load the asset price to determine if it pays off.
    double S = 0.0;
    if( path < num_paths )
      S = paths[offset + path];

    // Check if it pays off.
    const int in_the_money = payoff.is_in_the_money(S);

    // Try to check if we have found the 3 first stocks.
    if( found_paths < 3 )
    {
      int partial_sum = 0, total_sum = 0;
      partial_sum = dpct::group::exclusive_scan(item_ct1, in_the_money, 0,
                                                sycl::plus<>(), total_sum);
      if( in_the_money && found_paths + partial_sum < 3 )
        smem_svds[found_paths + partial_sum] = S;
      /*
      DPCT1118:3: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      item_ct1.barrier(sycl::access::fence_space::local_space);
      found_paths += total_sum;
    }

    // Early continue if no item pays off.
    if (!sycl::any_of_group(
            item_ct1.get_sub_group(),
            (0xFFFFFFFF &
             (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                in_the_money))
    {
      continue;
    }
    
    // Update the number of payoff items.
    m += in_the_money;

    // The "normalized" value.
    double x = 0.0, x_sq = 0.0;
    if( in_the_money )
    {
      x = S;
      x_sq = S*S;
    }

    // Compute the 4 sums.
    sums.x() += x;
    sums.y() += x_sq;
    sums.z() += x_sq * x;
    sums.w() += x_sq * x_sq;
  }

  // Make sure the scan is finished.
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Compute the final reductions.
  m = sycl::reduce_over_group(item_ct1.get_group(), m, sycl::plus<>());

  // Do we all exit?
  /*
  DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  int not_enough_paths =
      (item_ct1.barrier(),
       sycl::any_of_group(item_ct1.get_group(), item_ct1.get_local_id(2) == 0 &&
                                                    m < min_in_the_money));

  // Early exit if no path is in the money.
  if( not_enough_paths )
  {
    if (item_ct1.get_local_id(2) == 0)
      all_out_of_the_money[item_ct1.get_group(2)] = 1;
    return;
  }

  // Compute the final reductions.
  sums = sycl::reduce_over_group(item_ct1.get_group(), sums, sycl::plus<>());

  // The 1st thread has everything he needs to build R from the QR decomposition.
  if (item_ct1.get_local_id(2) == 0)
    svd_3x3(m, sums, smem_svds);
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Store the final results.
  if (item_ct1.get_local_id(2) < R_W_MATRICES_SMEM_SLOTS)
    svds[16 * item_ct1.get_group(2) + item_ct1.get_local_id(2)] =
        smem_svds[item_ct1.get_local_id(2)];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_THREADS_PER_BLOCK, typename Payoff>
/*
DPCT1110:5: The total declared local variable size in device function
compute_partial_beta_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/

void compute_partial_beta_kernel(int num_paths, Payoff payoff,
                                 const double *__restrict svd,
                                 const double * /*__restrict*/ paths,
                                 const double * /*__restrict*/ cashflows,
                                 const int *__restrict all_out_of_the_money,
                                 double *__restrict partial_sums,
                                 const sycl::nd_item<3> &item_ct1,
                                 double *shared_svd)
{

  // The shared memory storage.

  // The shared memory to store the SVD.

  // Early exit if needed.
  if( *all_out_of_the_money )
  {
    return;
  }

  // The number of threads per grid.
  const int NUM_THREADS_PER_GRID =
      NUM_THREADS_PER_BLOCK * item_ct1.get_group_range(2);

  // The 1st threads loads the matrices SVD and R.
  if (item_ct1.get_local_id(2) < R_W_MATRICES_SMEM_SLOTS)
    shared_svd[item_ct1.get_local_id(2)] = svd[item_ct1.get_local_id(2)];
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Load the terms of R.
  const double R00 = shared_svd[ 0];
  const double R01 = shared_svd[ 1];
  const double R02 = shared_svd[ 2];
  const double R11 = shared_svd[ 3];
  const double R12 = shared_svd[ 4];
  const double R22 = shared_svd[ 5];

  // Load the elements of W.
#ifdef WITH_FULL_W_MATRIX
  const double W00 = shared_svd[ 6];
  const double W01 = shared_svd[ 7];
  const double W02 = shared_svd[ 8];
  const double W10 = shared_svd[ 9];
  const double W11 = shared_svd[10];
  const double W12 = shared_svd[11];
  const double W20 = shared_svd[12];
  const double W21 = shared_svd[13];
  const double W22 = shared_svd[14];
#else
  const double W00 = shared_svd[ 6];
  const double W01 = shared_svd[ 7];
  const double W02 = shared_svd[ 8];
  const double W11 = shared_svd[ 9];
  const double W12 = shared_svd[10];
  const double W22 = shared_svd[11];
#endif

  // Invert the diagonal of R.
  /*
  DPCT1013:8: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  const double inv_R00 = R00 != 0.0 ? (1.0 / R00) : 0.0;
  /*
  DPCT1013:9: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  const double inv_R11 = R11 != 0.0 ? (1.0 / R11) : 0.0;
  /*
  DPCT1013:10: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  const double inv_R22 = R22 != 0.0 ? (1.0 / R22) : 0.0;

  // Precompute the R terms.
  const double inv_R01 = inv_R00*inv_R11*R01;
  const double inv_R02 = inv_R00*inv_R22*R02;
  const double inv_R12 =         inv_R22*R12;
  
  // Precompute W00/R00.
#ifdef WITH_FULL_W_MATRIX
  const double inv_W00 = W00*inv_R00;
  const double inv_W10 = W10*inv_R00;
  const double inv_W20 = W20*inv_R00;
#else
  const double inv_W00 = W00*inv_R00;
#endif

  // Each thread has 3 numbers to sum.
  double beta0 = 0.0, beta1 = 0.0, beta2 = 0.0;

  // Iterate over the paths.
  for (int path = item_ct1.get_group(2) * NUM_THREADS_PER_BLOCK +
                  item_ct1.get_local_id(2);
       path < num_paths; path += NUM_THREADS_PER_GRID)
  {
    // Threads load the asset price to rebuild Q from the QR decomposition.
    double S = paths[path];

    // Is the path in the money?
    const int in_the_money = payoff.is_in_the_money(S);

    // Compute Qis. The elements of the Q matrix in the QR decomposition.
    double Q1i = inv_R11*S - inv_R01;
    double Q2i = inv_R22*S*S - inv_R02 - Q1i*inv_R12;

    // Compute the ith row of the pseudo-inverse of [1 X X^2].
#ifdef WITH_FULL_W_MATRIX
    const double WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
    const double WI1 = inv_W10 + W11 * Q1i + W12 * Q2i;
    const double WI2 = inv_W20 + W21 * Q1i + W22 * Q2i;
#else
    const double WI0 = inv_W00 + W01 * Q1i + W02 * Q2i;
    const double WI1 =           W11 * Q1i + W12 * Q2i;
    const double WI2 =                       W22 * Q2i;
#endif

    // Each thread loads its element from the Y vector.
    double cashflow = in_the_money ? cashflows[path] : 0.0;
  
    // Update beta.
    beta0 += WI0*cashflow;
    beta1 += WI1*cashflow;
    beta2 += WI2*cashflow;
  }

  // Compute the sum of the elements in the block. We could do slightly better by removing the bank conflicts here.
  sycl::double3 sums = sycl::reduce_over_group(
      item_ct1.get_group(), sycl::double3(beta0, beta1, beta2), sycl::plus<>());

  // The 1st thread stores the result to GMEM.
#ifdef WITH_ATOMIC_BETA
  if( threadIdx.x == 0 )
  {
    atomic_add(&partial_sums[0], sums.x);
    atomic_add(&partial_sums[1], sums.y);
    atomic_add(&partial_sums[2], sums.z);
  }
#else
  if (item_ct1.get_local_id(2) == 0)
  {
    partial_sums[0 * NUM_THREADS_PER_BLOCK + item_ct1.get_group(2)] = sums.x();
    partial_sums[1 * NUM_THREADS_PER_BLOCK + item_ct1.get_group(2)] = sums.y();
    partial_sums[2 * NUM_THREADS_PER_BLOCK + item_ct1.get_group(2)] = sums.z();
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_BLOCK >

void compute_final_beta_kernel(const int *__restrict all_out_of_the_money, double *__restrict beta,
                               const sycl::nd_item<3> &item_ct1)
{

  // The shared memory for the reduction.

  // Early exit if needed.
  if( *all_out_of_the_money )
  {
    if (item_ct1.get_local_id(2) < 3)
      beta[item_ct1.get_local_id(2)] = 0.0;
    return;
  }

  // The final sums.
  sycl::double3 sums;

  // We load the elements.
  sums.x() = beta[0 * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2)];
  sums.y() = beta[1 * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2)];
  sums.z() = beta[2 * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2)];

  // Compute the sums.
  sums = sycl::reduce_over_group(item_ct1.get_group(), sums, sycl::plus<>());

  // Store beta.
  if (item_ct1.get_local_id(2) == 0)
  {
    //printf("beta0=%.8lf beta1=%.8lf beta2=%.8lf\n", sums.x, sums.y, sums.z);
    beta[0] = sums.x();
    beta[1] = sums.y();
    beta[2] = sums.z();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// If you enable WITH_FUSED_BETA, that kernel will assemble the beta coefficients from the partial sums computed
// in compute_partial_beta_kernel. Otherwise, it assumes beta has been built either by compute_final_beta_kernel or
// by atomic operations at the end of compute_partial_beta_kernel.

template< int NUM_THREADS_PER_BLOCK, typename Payoff >

void update_cashflow_kernel(int num_paths,
                            Payoff payoff_object,
                            double exp_min_r_dt,
                            const double *__restrict beta,
                            const double */*__restrict*/ paths,
                            const int *__restrict all_out_of_the_money,
                            double */*__restrict*/ cashflows,
                            const sycl::nd_item<3> &item_ct1,
                            double *smem_beta)
{
  const int NUM_THREADS_PER_GRID =
      item_ct1.get_group_range(2) * NUM_THREADS_PER_BLOCK;

  // Are we going to skip the computations.
  const int skip_computations = *all_out_of_the_money;

#ifdef WITH_FUSED_BETA

  // The shared memory for the reduction.

  // The shared memory to exchange beta.

  // The final sums.
  sycl::double3 sums;

  // We load the elements. Each block loads the same elements.
  sums.x() = beta[0 * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2)];
  sums.y() = beta[1 * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2)];
  sums.z() = beta[2 * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2)];

  // Compute the sums.
  sums = sycl::reduce_over_group(item_ct1.get_group(), sums, sycl::plus<>());

  // Store beta.
  if (item_ct1.get_local_id(2) == 0)
  {
    smem_beta[0] = sums.x();
    smem_beta[1] = sums.y();
    smem_beta[2] = sums.z();
  }
  /*
  DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Load the beta coefficients from SMEM.
  const double beta0 = smem_beta[0];
  const double beta1 = smem_beta[1];
  const double beta2 = smem_beta[2];
#else
  // Load the beta coefficients for the linear regression.
  const double beta0 = beta[0];
  const double beta1 = beta[1];
  const double beta2 = beta[2];
#endif

  // Iterate over the paths.
  int path =
      item_ct1.get_group(2) * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2);
  for( ; path < num_paths ; path += NUM_THREADS_PER_GRID )
  {
    // The cashflow.
    const double old_cashflow = exp_min_r_dt*cashflows[path];
    if( skip_computations )
    {
      cashflows[path] = old_cashflow;
      continue;
    }
  
    // Load the asset price.
    double S  = paths[path];
    double S2 = S*S;

    // The payoff.
    double payoff = payoff_object(S);

    // Compute the estimated payoff from continuing.
    double estimated_payoff = beta0 + beta1*S + beta2*S2;

    // Discount the payoff because we did not take it into account for beta.
    estimated_payoff *= exp_min_r_dt;

    // Update the payoff.
    if( payoff <= 1.0e-8 || payoff <= estimated_payoff )
      payoff = old_cashflow;
    
    // Store the updated cashflow.
    cashflows[path] = payoff;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef WITH_CDP
template< int NUM_THREADS_PER_BLOCK, typename Payoff >
__global__ __launch_bounds__(NUM_THREADS_PER_BLOCK, 8)
void cdp_timestep_loop_kernel(int num_timesteps,
                              int num_paths,
                              int update_cashflow_grid,
                              Payoff payoff,
                              double exp_min_r_dt,
                              const double *__restrict svds,
                              const double */*__restrict*/ paths,
                              double */*__restrict*/ cashflows,
                              const int *__restrict all_out_of_the_money,
                              double *__restrict temp_storage)
{
#if __CUDA_ARCH__ >= 350
  for( int timestep = num_timesteps-2 ; timestep >= 0 ; --timestep )
  {
    compute_partial_beta_kernel<NUM_THREADS_PER_BLOCK><<<NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(
      num_paths,
      payoff,
      svds + 16*timestep,
      paths + timestep*num_paths,
      cashflows,
      all_out_of_the_money + timestep,
      temp_storage);

#if defined(WITH_FUSED_BETA) || defined(WITH_ATOMIC_BETA)
#else
    compute_final_beta_kernel<NUM_THREADS_PER_BLOCK><<<1, NUM_THREADS_PER_BLOCK>>>(
      all_out_of_the_money + timestep,
      temp_storage);
#endif

    update_cashflow_kernel<NUM_THREADS_PER_BLOCK><<<update_cashflow_grid, NUM_THREADS_PER_BLOCK>>>(
      num_paths,
      payoff,
      exp_min_r_dt,
      temp_storage,
      paths + timestep*num_paths,
      all_out_of_the_money + timestep,
      cashflows);
  }
#endif
}
#endif // WITH_CDP

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_THREADS_PER_BLOCK >

void compute_partial_sums_kernel(int num_paths, const double *__restrict cashflows, double *__restrict sums,
                                 const sycl::nd_item<3> &item_ct1)
{

  // Shared memory to compute the final sum.

  // Each thread works on a single path.
  const int path =
      item_ct1.get_group(2) * NUM_THREADS_PER_BLOCK + item_ct1.get_local_id(2);

  // Load the final sum.
  double sum = 0.0;
  if( path < num_paths )
    sum = cashflows[path];

  // Compute the sum over the block.
  sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());

  // The block leader writes the sum to GMEM.
  if (item_ct1.get_local_id(2) == 0)
    sums[item_ct1.get_group(2)] = sum;
}

// ====================================================================================================================

template< int NUM_THREADS_PER_BLOCK >

void compute_final_sum_kernel(int num_paths, int num_blocks, double exp_min_r_dt, double *__restrict sums,
                              const sycl::nd_item<3> &item_ct1)
{

  // Shared memory to compute the final sum.

  // The sum.
  double sum = 0.0;
  for (int item = item_ct1.get_local_id(2); item < num_blocks;
       item += NUM_THREADS_PER_BLOCK)
    sum += sums[item];

  // Compute the sum over the block.
  sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());

  // The block leader writes the sum to GMEM.
  if (item_ct1.get_local_id(2) == 0)
  {
    sums[0] = exp_min_r_dt * sum / (double) num_paths;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Payoff>
static inline void
do_run(dpct::queue_ptr stream, dpct::rng::host_rng_ptr rng, int num_timesteps,
       int num_paths, const Payoff &payoff, double dt, double S0, double r,
       double sigma, double *d_samples, double *d_paths, double *d_cashflows,
       double *d_svds, int *d_all_out_of_the_money, double *d_temp_storage,
       double *h_price) try {
  // Generate random samples.
auto start_ct1 = std::chrono::steady_clock::now();
#if !USE_DEVICE_API

    auto engine = oneapi::mkl::rng::mrg32k3a(*stream, 0);
    oneapi::mkl::rng::generate(oneapi::mkl::rng::gaussian<double>(0.0, 1.0), engine,
        num_timesteps * num_paths, d_samples);
#endif

  // Generate asset prices.
  const int NUM_THREADS_PER_BLOCK0 = 256;
  int grid_dim = (num_paths + NUM_THREADS_PER_BLOCK0-1) / NUM_THREADS_PER_BLOCK0;
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_dim) *
                              sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK0),
                          sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK0)),
        [=](sycl::nd_item<3> item_ct1) {
          generate_paths_kernel<NUM_THREADS_PER_BLOCK0>(
              num_timesteps, num_paths, payoff, dt, S0, r, sigma, d_samples,
              d_paths, item_ct1);
        })
        .wait();
  }
  auto stop_ct1 = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("RNG time   : %.3fms\n", elapsed_time);

  // Reset the all_out_of_the_money array.
  CHECK_CUDA(DPCT_CHECK_ERROR(
      stream->memset(d_all_out_of_the_money, 0, num_timesteps * sizeof(int))));

  // Prepare the SVDs.
  const int NUM_THREADS_PER_BLOCK1 = 256;
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
    stream->submit([&](sycl::handler &cgh) {
      /*
      DPCT1054:24: The type of variable smem_storage is declared in device
      function with the name TempStorage. Adjust the code to make the
      TempStorage declaration visible at the accessor declaration point.
      */
      //   sycl::local_accessor<uint8_t[sizeof(TempStorage)], 0>
      //       smem_storage_ct1_acc_ct1(cgh);
      /*
      DPCT1101:25: 'R_W_MATRICES_SMEM_SLOTS' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<double, 1> smem_svds_acc_ct1(
          sycl::range<1>(12 /*R_W_MATRICES_SMEM_SLOTS*/), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_timesteps - 1) *
                                sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK1),
                            sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK1)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            prepare_svd_kernel<NUM_THREADS_PER_BLOCK1>(
                num_paths, 4, payoff, d_paths, d_all_out_of_the_money, d_svds,
                item_ct1, //smem_storage_ct1_acc_ct1.get_pointer(),
                smem_svds_acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get());
          });
    });
  }

  // The constant to discount the payoffs.
  const double exp_min_r_dt = std::exp(-r*dt);

  // Estimate the number of blocks in a wave of update_cashflow.
  dpct::device_info properties;
  int device = 0;
  CHECK_CUDA(device = dpct::dev_mgr::instance().current_device_id());
  CHECK_CUDA(DPCT_CHECK_ERROR(dpct::get_device_info(
      properties, dpct::dev_mgr::instance().get_device(device))));

  // The number of SMs.
  const int num_sms = properties.get_max_compute_units();
  // Number of threads per wave at fully occupancy.
  const int num_threads_per_wave_full_occupancy =
      properties.get_max_work_items_per_compute_unit() * num_sms;

  // Enable 8B mode for SMEM.
  const int NUM_THREADS_PER_BLOCK2 = 128;

  // Update the cashflows.
  grid_dim = (num_paths + NUM_THREADS_PER_BLOCK2-1) / NUM_THREADS_PER_BLOCK2;
  double num_waves = grid_dim*NUM_THREADS_PER_BLOCK2 / (double) num_threads_per_wave_full_occupancy;

  int update_cashflow_grid = grid_dim;
  if( num_waves < 10 && num_waves - (int) num_waves < 0.6 )
    update_cashflow_grid = std::max(1, (int) num_waves) * num_threads_per_wave_full_occupancy / NUM_THREADS_PER_BLOCK2;

  // Run the main loop.
#ifdef WITH_CDP
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 512));
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 1));
  cdp_timestep_loop_kernel<NUM_THREADS_PER_BLOCK2><<<1, 1, 0, stream>>>(num_timesteps,
                                                                        num_paths,
                                                                        update_cashflow_grid,
                                                                        payoff,
                                                                        exp_min_r_dt,
                                                                        d_svds,
                                                                        d_paths,
                                                                        d_cashflows,
                                                                        d_all_out_of_the_money,
                                                                        d_temp_storage);
  CHECK_CUDA(cudaGetLastError());
#else
  for( int timestep = num_timesteps-2 ; timestep >= 0 ; --timestep )
  {
#ifdef WITH_ATOMIC_BETA
    // Reset the buffer to store the results.
    CHECK_CUDA(cudaMemsetAsync(d_temp_storage, 0, 3*sizeof(double)));
#endif

    // Compute beta (two kernels) for that timestep.
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:26: 'R_W_MATRICES_SMEM_SLOTS' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<double, 1> shared_svd_acc_ct1(
            sycl::range<1>(12 /*R_W_MATRICES_SMEM_SLOTS*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK2) *
                                  sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK2),
                              sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK2)),
            [=](sycl::nd_item<3> item_ct1) {
              compute_partial_beta_kernel<NUM_THREADS_PER_BLOCK2>(
                  num_paths, payoff, d_svds + 16 * timestep,
                  d_paths + timestep * num_paths, d_cashflows,
                  d_all_out_of_the_money + timestep, d_temp_storage, item_ct1,
                  shared_svd_acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get());
            });
      });
    }

#if defined(WITH_FUSED_BETA) || defined(WITH_ATOMIC_BETA)
#else
    compute_final_beta_kernel<NUM_THREADS_PER_BLOCK2><<<1, NUM_THREADS_PER_BLOCK2, 0, stream>>>(
      d_all_out_of_the_money + timestep,
      d_temp_storage);
    CHECK_CUDA(cudaGetLastError());
#endif

    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> smem_beta_acc_ct1(sycl::range<1>(3),
                                                          cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, update_cashflow_grid) *
                                  sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK2),
                              sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK2)),
            [=](sycl::nd_item<3> item_ct1) {
              update_cashflow_kernel<NUM_THREADS_PER_BLOCK2>(
                  num_paths, payoff, exp_min_r_dt, d_temp_storage,
                  d_paths + timestep * num_paths,
                  d_all_out_of_the_money + timestep, d_cashflows, item_ct1,
                  smem_beta_acc_ct1.template get_multi_ptr<sycl::access::decorated::yes>().get());
            });
      });
    }
  }
#endif // WITH_CDP

  // Compute the final sum.
  const int NUM_THREADS_PER_BLOCK4 = 128;
  grid_dim = (num_paths + NUM_THREADS_PER_BLOCK4-1) / NUM_THREADS_PER_BLOCK4;

  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_dim) *
                              sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK4),
                          sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK4)),
        [=](sycl::nd_item<3> item_ct1) {
          compute_partial_sums_kernel<NUM_THREADS_PER_BLOCK4>(
              num_paths, d_cashflows, d_temp_storage, item_ct1);
        });
  }
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK4),
                          sycl::range<3>(1, 1, NUM_THREADS_PER_BLOCK4)),
        [=](sycl::nd_item<3> item_ct1) {
          compute_final_sum_kernel<NUM_THREADS_PER_BLOCK4>(
              num_paths, grid_dim, exp_min_r_dt, d_temp_storage, item_ct1);
        });
  }

  // Copy the result to the host.
  CHECK_CUDA(DPCT_CHECK_ERROR(
      stream->memcpy(h_price, d_temp_storage, sizeof(double))));
  stream->wait();
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Payoff >
static double binomial_tree(int num_timesteps, const Payoff &payoff, double dt, double S0, double r, double sigma)
{
  double *tree = new double[num_timesteps+1];

  double u = std::exp( sigma * std::sqrt(dt));
  double d = std::exp(-sigma * std::sqrt(dt));
  double a = std::exp( r     * dt);
  
  double p = (a - d) / (u - d);
  
  double k = std::pow(d, num_timesteps);
  for( int t = 0 ; t <= num_timesteps ; ++t )
  {
    tree[t] = payoff(S0*k);
    k *= u*u;
  }

  for( int t = num_timesteps-1 ; t >= 0 ; --t )
  {
    k = std::pow(d, t);
    for( int i = 0 ; i <= t ; ++i )
    {
      double expected = std::exp(-r*dt) * (p*tree[i+1] + (1.0 - p)*tree[i]);
      double earlyex = payoff(S0*k);
      tree[i] = std::max(earlyex, expected);
      k *= u*u;
    }
  }

  double f = tree[0];
  delete[] tree;
  return f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static double black_scholes_merton_put(double T, double K, double S0, double r, double sigma)
{
  double d1 = (std::log(S0 / K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
  double d2 = d1 - sigma*std::sqrt(T);

  return K * std::exp(-r * T) * sycl::erfc(-d2 / -sycl::sqrt(2.0)) / 2 -
         S0 * sycl::erfc(-d1 / -sycl::sqrt(2.0)) / 2;
}

static double black_scholes_merton_call(double T, double K, double S0, double r, double sigma)
{
  double d1 = (std::log(S0 / K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
  double d2 = d1 - sigma*std::sqrt(T);

  return S0 * sycl::erfc(d1 / -sycl::sqrt(2.0)) / 2 -
         K * std::exp(-r * T) * sycl::erfc(d2 / -sycl::sqrt(2.0)) / 2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef WITH_CPU_REFERENCE

extern "C" void dgesvd_(char*,   // JOBU
                        char*,   // JOBV
                        long*,   // M
                        long*,   // N
                        double*, // A
                        long*,   // LDA
                        double*, // S
                        double*, // U
                        long*,   // LDU
                        double*, // VT
                        long*,   // LDVT
                        double*, // WORK
                        long*,   // LWORK
                        long*);  // INFO

// ====================================================================================================================

static void dump_to_file(const char *name, int timestep, const double *data, int count)
{
  char buffer[256];
  sprintf(buffer, "%s-%d.bin", name, timestep);
  FILE *file = fopen(buffer, "wb");
  if( !file ) 
  {
    fprintf(stderr, "Error cannot open file %s\n", buffer);
    exit(1);
  }
  printf("> Debug info          : Writing %s to binary file %s\n", name, buffer);
  if( count != fwrite(data, sizeof(double), count, file) )
  {
    fprintf(stderr, "Error when dumping the binary values to %s\n", buffer);
    exit(1);
  }
  fclose(file);
}

// ====================================================================================================================

template< typename Payoff >
static double longstaff_schwartz_cpu(int num_timesteps, 
                                     int num_paths, 
                                     const Payoff &payoff, 
                                     double dt,
                                     double S0,
                                     double r,
                                     double sigma,
                                     bool with_debug_info)
{
  // The random samples.
  double *h_samples = new double[num_timesteps*num_paths];
  curandGenerator_t rng;
  CHECK_CURAND(curandCreateGeneratorHost(&rng, CURAND_RNG_PSEUDO_MRG32K3A));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, 0));
  CHECK_CURAND(curandGenerateNormalDouble(rng, h_samples, num_timesteps*num_paths, 0.0, 1.0));
  CHECK_CURAND(curandDestroyGenerator(rng));

  // The paths.
  double *h_paths = new double[num_timesteps*num_paths];

  const double r_min_half_sigma_sq_dt = (r - 0.5*sigma*sigma)*dt;
  const double sigma_sqrt_dt = sigma*sqrt(dt);

  // Generate the paths.
  for( int timestep = 0 ; timestep < num_timesteps ; ++timestep )
  {
    for( int i = 0 ; i < num_paths ; ++i )
    {
      double S = timestep == 0 ? S0 : h_paths[(timestep-1)*num_paths + i];
      S = S * exp(r_min_half_sigma_sq_dt + sigma_sqrt_dt*h_samples[timestep*num_paths + i]);
      h_paths[timestep*num_paths + i] = timestep < num_timesteps-1 ? S : payoff(S);
    }
  }

  // The cashflows (last column of paths).
  double *h_cashflows = &h_paths[(num_timesteps-1)*num_paths];

  // The constant to discount the payoffs.
  const double exp_min_r_dt = std::exp(-r*dt);

  // The matrix [1 x x^2].
  double *h_matrix = new double[3*num_paths];
  // The singular values.
  double *h_S = new double[3];
  // The matrix U of the SVD.
  double *h_U = new double[3*num_paths];
  // The matrix V^T of the SVD.
  double *h_V = new double[3*3];
  // The workspace.
  double *h_work = new double[num_paths + 3*3];

  // Run the main loop.
  for( int timestep = num_timesteps-2 ; timestep >= 0 ; --timestep )
  {
    long m = 0;

    // Prepare the matrix [1 x x^2].
    for( int i = 0 ; i < num_paths ; ++i )
    {
      double S = h_paths[timestep*num_paths + i];
      if( !payoff.is_in_the_money(S) )
        continue;

      h_matrix[0*num_paths + m] = 1.0;
      h_matrix[1*num_paths + m] = S;
      h_matrix[2*num_paths + m] = S*S;

      m++;
    }

    if( with_debug_info )
      dump_to_file("paths", timestep, &h_matrix[num_paths], m);

    // Compute the SVD of the matrix.
    char JOBU = 'S', JOBVT = 'S';
    long ldm = num_paths;
    long N = 3;
    long LWORK = num_paths + 3*3;
    long info = 0;
    dgesvd_(&JOBU, &JOBVT, &m, &N, h_matrix, &ldm, h_S, h_U, &ldm, h_V, &N, h_work, &LWORK, &info);
    if( info )
    {
      fprintf(stderr, "LAPACK error at line %d: %d\n", __LINE__, info);
      exit(1);
    }

    if( with_debug_info )
      printf("> Debug info          : Timestep=%3d, svd0=%.8lf svd1=%.8lf svd2=%.8lf\n", timestep, h_S[0], h_S[1], h_S[2]);

    // Build the pseudo-inverse: V*S^-1*U^T.
    double inv_S0 = abs(h_S[0]) < 1.0e-12 ? 0.0 : 1.0 / h_S[0];
    double inv_S1 = abs(h_S[1]) < 1.0e-12 ? 0.0 : 1.0 / h_S[1];
    double inv_S2 = abs(h_S[2]) < 1.0e-12 ? 0.0 : 1.0 / h_S[2];

    // V = V^T*S^-1.
    h_V[0] *= inv_S0; h_V[1] *= inv_S1; h_V[2] *= inv_S2;
    h_V[3] *= inv_S0; h_V[4] *= inv_S1; h_V[5] *= inv_S2;
    h_V[6] *= inv_S0; h_V[7] *= inv_S1; h_V[8] *= inv_S2;

    // U = V*U^T.
    for( int i = 0 ; i < m ; ++i )
    {
      double a = h_U[0*num_paths + i];
      double b = h_U[1*num_paths + i];
      double c = h_U[2*num_paths + i];

      h_U[0*num_paths + i] = a*h_V[0] + b*h_V[1] + c*h_V[2];
      h_U[1*num_paths + i] = a*h_V[3] + b*h_V[4] + c*h_V[5];
      h_U[2*num_paths + i] = a*h_V[6] + b*h_V[7] + c*h_V[8];
    }

    // Compute beta.
    double beta0 = 0.0, beta1 = 0.0, beta2 = 0.0;
    for( int i = 0, k = 0 ; i < num_paths ; ++i )
    {
      double S = h_paths[timestep*num_paths + i];
      if( !payoff.is_in_the_money(S) )
        continue;

      double cashflow = h_cashflows[i];

      beta0 += h_U[0*num_paths + k]*cashflow;
      beta1 += h_U[1*num_paths + k]*cashflow;
      beta2 += h_U[2*num_paths + k]*cashflow;

      k++;
    }

    if( with_debug_info )
    {
      double *h_tmp_cashflows = new double[m];
      for( int i = 0, k = 0 ; i < num_paths ; ++i )
      {
        double S = h_paths[timestep*num_paths + i];
        if( !payoff.is_in_the_money(S) )
          continue;
        h_tmp_cashflows[k++] = h_cashflows[i];
      }
      dump_to_file("cashflows", timestep, h_tmp_cashflows, m);
      delete[] h_tmp_cashflows;
    }

    if( with_debug_info )
      printf("> Debug info          : Timestep=%3d, beta0=%.8lf beta1=%.8lf beta2=%.8lf\n", timestep, beta0, beta1, beta2);

    // Update the cashflow.
    for( int i = 0 ; i < num_paths ; ++i )
    {
      double S = h_paths[timestep*num_paths + i];
      double p = payoff(S);

      double estimated_payoff = exp_min_r_dt*(beta0 + beta1*S + beta2*S*S);

      if( p <= 1.0e-8 || p <= estimated_payoff )
        p = exp_min_r_dt*h_cashflows[i];
      h_cashflows[i] = p;
    }
  }
    
  // Compute the final sum.
  double sum = 0.0;
  for( int i = 0 ; i < num_paths ; ++i )
    sum += h_cashflows[i];

  delete[] h_V;
  delete[] h_U;
  delete[] h_S;
  delete[] h_matrix;
  delete[] h_paths;
  delete[] h_samples;

  return exp_min_r_dt*sum / (double) num_paths;
}

// ====================================================================================================================

static double longstaff_schwartz_cpu(int num_timesteps, 
                                     int num_paths, 
                                     bool price_put,
                                     double K, 
                                     double dt,
                                     double S0,
                                     double r,
                                     double sigma,
                                     bool with_debug_info)
{
  if( price_put )
    return longstaff_schwartz_cpu(num_timesteps, num_paths, PayoffPut(K), dt, S0, r, sigma, with_debug_info);
  else
    return longstaff_schwartz_cpu(num_timesteps, num_paths, PayoffCall(K), dt, S0, r, sigma, with_debug_info);
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const int MAX_GRID_SIZE = 2048;
  
  // Simulation parameters.
  int num_timesteps = NUM_TIMESTEPS;
  int num_paths     = NUM_PATHS;
  int num_runs      = 10;

  // Option parameters.
  double T     = 1.00;
  double K     = 4.00;
  double S0    = 3.60;
  double r     = 0.06;
  double sigma = 0.20;

  // Bool do we price a put or a call.
  bool price_put = true;
  
  // Do we want debug info.
#ifdef WITH_CPU_REFERENCE
  bool with_debug_info = false;
#endif

  // Read command-line options.
  for( int i = 1 ; i < argc ; ++i )
  {
    if( !strcmp(argv[i], "-timesteps") )
      num_timesteps = strtol(argv[++i], NULL, 10);
    else if( !strcmp(argv[i], "-paths") )
      num_paths = strtol(argv[++i], NULL, 10);
    else if( !strcmp(argv[i], "-runs") )
      num_runs = strtol(argv[++i], NULL, 10);
    else if( !strcmp(argv[i], "-T") )
      T = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-S0") )
      S0 = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-K") )
      K = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-r") )
      r = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-sigma") )
      sigma = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-call") )
      price_put = false;
#ifdef WITH_CPU_REFERENCE
    else if( !strcmp(argv[i], "-debug-info") )
      with_debug_info = true;
#endif
    else
    {
      fprintf(stderr, "Unknown option %s. Aborting!!!\n", argv[i]);
      exit(1);
    }
  }

  // Print the arguments.
  printf("==============\n");
  printf("Num Timesteps         : %d\n",  num_timesteps);
  printf("Num Paths             : %dK\n", num_paths);
  printf("Num Runs              : %d\n",  num_runs);
  printf("T                     : %lf\n", T);
  printf("S0                    : %lf\n", S0);
  printf("K                     : %lf\n", K);
  printf("r                     : %lf\n", r);
  printf("sigma                 : %lf\n", sigma);
  printf("Option Type           : American %s\n",  price_put ? "Put" : "Call");

  // We want x1024 paths.
  num_paths *= 1024;

  // A timestep.
  double dt = T / num_timesteps;

  // Create a stream to issue asynchronous results (and create the CUDA context).
  dpct::queue_ptr stream;
  CHECK_CUDA(DPCT_CHECK_ERROR(stream = dev_ct1.create_queue()));

  // Memory on the GPU to store normally distributed random numbers.
  double *d_samples = NULL;
  CHECK_CUDA(DPCT_CHECK_ERROR(d_samples = sycl::malloc_device<double>(
                                  num_timesteps * num_paths, q_ct1)));

  // Memory on the GPU to store the asset price along the paths. The last column contains the discounted payoffs.
  double *d_paths = NULL;
  CHECK_CUDA(DPCT_CHECK_ERROR(
      d_paths = sycl::malloc_device<double>(num_timesteps * num_paths, q_ct1)));

  // The discounted payoffs are the last column.
  double *d_cashflows = d_paths + (num_timesteps-1)*num_paths;

  // Storage to keep intermediate SVD matrices.
  double *d_svds = NULL;
  CHECK_CUDA(DPCT_CHECK_ERROR(
      d_svds = sycl::malloc_device<double>(16 * num_timesteps, q_ct1)));

  // Memory on the GPU to flag timesteps where no path is in the money.
  int *d_all_out_of_the_money = NULL;
  CHECK_CUDA(DPCT_CHECK_ERROR(
      d_all_out_of_the_money = sycl::malloc_device<int>(num_timesteps, q_ct1)));

  // Memory on the GPU to compute the reductions (beta and the option price).
  int max_temp_storage = 4*MAX_GRID_SIZE;
  double *d_temp_storage = NULL;
  CHECK_CUDA(DPCT_CHECK_ERROR(
      d_temp_storage = sycl::malloc_device<double>(max_temp_storage, q_ct1)));

  // The price on the host.
  double *h_price = NULL;
  /*
  DPCT1048:0: The original value cudaHostAllocDefault is not meaningful in the
  migrated code and was removed or replaced with 0. You may need to check the
  migrated code.
  */
  CHECK_CUDA(DPCT_CHECK_ERROR(h_price = sycl::malloc_host<double>(1, q_ct1)));

  // Create the random-number generator and set the seed.
  dpct::rng::host_rng_ptr rng;
#ifndef USE_DEVICE_API
  CHECK_CURAND(DPCT_CHECK_ERROR(rng = dpct::rng::create_host_rng(
                                    dpct::rng::random_engine_type::mrg32k3a)));
  CHECK_CURAND(DPCT_CHECK_ERROR(rng->set_queue(stream)));
  CHECK_CURAND(DPCT_CHECK_ERROR(rng->set_seed(0)));
#endif

  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  if( price_put )
      do_run(stream,
             rng,
             num_timesteps, 
             num_paths, 
             PayoffPut(K), 
             dt,
             S0,
             r,
             sigma,
             d_samples,
             d_paths,
             d_cashflows,
             d_svds,
             d_all_out_of_the_money,
             d_temp_storage,
             h_price);
  else
      do_run(stream,
             rng,
             num_timesteps, 
             num_paths, 
             PayoffCall(K), 
             dt,
             S0,
             r,
             sigma,
             d_samples,
             d_paths,
             d_cashflows,
             d_svds,
             d_all_out_of_the_money,
             d_temp_storage,
             h_price);
  start_ct1 = std::chrono::steady_clock::now();

  for( int run = 0 ; run < num_runs ; ++run )
  {
#ifndef USE_DEVICE_API
    CHECK_CURAND(DPCT_CHECK_ERROR(rng->skip_ahead(0)));
#endif
    if( price_put )
      do_run(stream,
             rng,
             num_timesteps, 
             num_paths, 
             PayoffPut(K), 
             dt,
             S0,
             r,
             sigma,
             d_samples,
             d_paths,
             d_cashflows,
             d_svds,
             d_all_out_of_the_money,
             d_temp_storage,
             h_price);
    else
      do_run(stream,
             rng,
             num_timesteps, 
             num_paths, 
             PayoffCall(K), 
             dt,
             S0,
             r,
             sigma,
             d_samples,
             d_paths,
             d_cashflows,
             d_svds,
             d_all_out_of_the_money,
             d_temp_storage,
             h_price);
  }
  stop_ct1 = std::chrono::steady_clock::now();

  printf("==============\n");
  printf("GPU Longstaff-Schwartz: %.8lf\n", *h_price);
  
  double price = 0.0;
#ifdef WITH_CPU_REFERENCE
  price = longstaff_schwartz_cpu(num_timesteps, num_paths, price_put, K, dt, S0, r, sigma, with_debug_info);

  printf("CPU Longstaff-Schwartz: %.8lf\n", price);
#endif

  if( price_put )
    price = binomial_tree(num_timesteps, PayoffPut(K), dt, S0, r, sigma);
  else
    price = binomial_tree(num_timesteps, PayoffCall(K), dt, S0, r, sigma);

  printf("Binonmial             : %.8lf\n", price);
  
  if( price_put )
    price = black_scholes_merton_put(T, K, S0, r, sigma);
  else
    price = black_scholes_merton_call(T, K, S0, r, sigma);

  printf("European Price        : %.8lf\n", price);

  printf("==============\n");

  float elapsed_time = 0.0f;
  elapsed_time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  printf("Elapsed time          : %.3fms\n", elapsed_time / num_runs);
  printf("==============\n");
  
  // Release the GPU memory.
  CHECK_CUDA(DPCT_CHECK_ERROR(sycl::free(h_price, q_ct1)));
#ifndef USE_DEVICE_API
  CHECK_CURAND(DPCT_CHECK_ERROR(rng.reset()));
#endif

  CHECK_CUDA(DPCT_CHECK_ERROR(sycl::free(d_temp_storage, q_ct1)));
  CHECK_CUDA(DPCT_CHECK_ERROR(sycl::free(d_all_out_of_the_money, q_ct1)));
  CHECK_CUDA(DPCT_CHECK_ERROR(sycl::free(d_svds, q_ct1)));
  CHECK_CUDA(DPCT_CHECK_ERROR(sycl::free(d_paths, q_ct1)));
#if !USE_DEVICE_API
  CHECK_CUDA(DPCT_CHECK_ERROR(sycl::free(d_samples, q_ct1)));
#endif
  CHECK_CUDA(DPCT_CHECK_ERROR(dev_ct1.destroy_queue(stream)));

  // Reset the GPU (it's a good practice).
  CHECK_CUDA(DPCT_CHECK_ERROR(dev_ct1.reset()));

  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


