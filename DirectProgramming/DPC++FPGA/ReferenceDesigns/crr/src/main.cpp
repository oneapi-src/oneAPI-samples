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

////////////////////////////////////////////////////////////////////////////////
//
// CRRSolver CPU/FPGA Accelerator Demo Program
//
////////////////////////////////////////////////////////////////////////////////
//
// This design implments simple Cox-Ross-Rubinstein(CRR) binomial tree model
// with Greeks for American exercise options.
//
//
// Optimization summary:
//    -- Area-consuming but infrequent calculation is done on CPU.
//    -- Parallelize the calculation of a single CRR.
//    -- Run multiple independent CRRs in parallel.
//    -- Optimized memory configurations to reduce the need for replication
//       and to eliminate the need for double-pumping M20Ks.
//
// The following diagram shows the mechanism of optimizations to CRR.
//
//
//                                               +------+         ^
//                                 +------------>|optval|         |
//                                 |             | [2]  |         |
//                                 |             +------+         |
//                                 |                              |
//                                 |                              |
//                              +--+---+                          |
//                +------------>|optval|                          |
//                |             | [1]  |                          |
//                |             +--+---+                          |
//                |                |                              |
//                |                |                              |
//                |                |                              |   Loop4(L4)
//                |                |                              |   updates
//            +---+--+             +------------>+------+         |   multiple
//            |optval|                           |optval|         |   elements
//            | [0]  |                           | [1]  |         |   in optval[]
//            +---+--+             +------------>+------+         |   simultaneously
//                |                |                              |
//                |                |                              |
//                |                |                              |
//                |                |                              |
//                |             +--+---+                          |
//                |             |optval|                          |
//                +------------>| [0]  |                          |
//                              +--+---+                          |
//                                 |                              |
//                                 |                              |
//                                 |             +------+         |
//                                 |             |optval|         |
//                                 +------------>| [0]  |         |
//                                               +------+         +
//
//
//
//
//                              step 1           step 2
//
//
//                <------------------------------------------+
//                  Loop3(L3) updates each level of the tree
//
//

#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include "CRR_common.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace std;
using namespace sycl;

class CRRSolver;
double CrrSolver(const int n_items, vector<CRRMeta> &in_params,
                  vector<CRRResParams> &res_params,
                  vector<CRRPerStepMeta> &in_params2, queue &q) {
  dpc_common::TimeInterval timer;

  constexpr int steps = kMaxNSteps2;

  const int n_crr =
      (((n_items + (OUTER_UNROLL - 1)) / OUTER_UNROLL) * OUTER_UNROLL) * 3;

  {
    buffer<CRRMeta, 1> i_params(in_params.size());
    buffer<CRRPerStepMeta, 1> a_params(in_params2.size());
    buffer<CRRResParams, 1> r_params(res_params.size());
    r_params.set_final_data(res_params.data());

    event e;
    {
      // copy the input buffers
      q.submit([&](handler& h) {
        auto accessor_v =
          i_params.template get_access<access::mode::discard_write>(h);
        h.copy(in_params.data(), accessor_v);
      });

      q.submit([&](handler& h) {
        auto accessor_v2 =
          a_params.template get_access<access::mode::discard_write>(h);
        h.copy(in_params2.data(), accessor_v2);
      });

      // start the main kernel
      e = q.submit([&](handler &h) {
        auto accessor_v =
            i_params.template get_access<access::mode::read_write>(h);

        auto accessor_v2 =
            a_params.template get_access<access::mode::read_write>(h);

        auto accessor_r =
            r_params.template get_access<access::mode::discard_write>(h);

        h.single_task<CRRSolver>([=]() [[intel::kernel_args_restrict]] {
          // Kernel requires n_crr to be a multiple of OUTER_UNROLL.
          // This is taken care of by the host.
          const int n_crr_div = n_crr / OUTER_UNROLL;

          // Outerloop counter. Use while-loop for better timing-closure
          // characteristics because it tells the compiler the loop body will
          // never be skipped.
          int oc = 0;
          do {
            // Metadata of CRR problems
            [[intelfpga::register]] double u[OUTER_UNROLL];
            [[intelfpga::register]] double c1[OUTER_UNROLL];
            [[intelfpga::register]] double c2[OUTER_UNROLL];
            [[intelfpga::register]] double param_1[OUTER_UNROLL];
            [[intelfpga::register]] double param_2[OUTER_UNROLL];
            [[intelfpga::register]] short n_steps[OUTER_UNROLL];

            // Current values in binomial tree.  We only need to keep track of
            // one level worth of data, not the entire tree.
            [[intelfpga::memory, intelfpga::singlepump,
              intelfpga::bankwidth(sizeof(double)),
              intelfpga::numbanks(INNER_UNROLL * OUTER_UNROLL_POW2),
              intelfpga::private_copies(
                  8)]] double optval[kMaxNSteps3][OUTER_UNROLL_POW2];

            // Initial values in binomial tree, which correspond to the last
            // level of the binomial tree.
            [[intelfpga::memory, intelfpga::singlepump,
              intelfpga::bankwidth(sizeof(double)),
              intelfpga::numbanks(INNER_UNROLL * OUTER_UNROLL_POW2),
              intelfpga::private_copies(
                  8)]] double init_optval[kMaxNSteps3][OUTER_UNROLL_POW2];

            // u2_array precalculates the power function of u2.
            [[intelfpga::memory, intelfpga::singlepump,
              intelfpga::bankwidth(sizeof(double)),
              intelfpga::numbanks(INNER_UNROLL * OUTER_UNROLL_POW2),
              intelfpga::private_copies(
                  8)]] double u2_array[kMaxNSteps3][OUTER_UNROLL_POW2];

            // p1powu_array precalculates p1 multipy the power of u.
            [[intelfpga::memory, intelfpga::singlepump,
              intelfpga::bankwidth(sizeof(double)),
              intelfpga::numbanks(INNER_UNROLL * OUTER_UNROLL_POW2),
              intelfpga::private_copies(
                  8)]] double p1powu_array[kMaxNSteps3][OUTER_UNROLL_POW2];

            // n0_optval stores the binomial tree value corresponding to node 0
            // of a level. This is the same as what's stored in
            // optval/init_optval, but replicating this data allows us to have
            // only one read port for optval and init_optval, thereby removing
            // the need of double-pumping or replication. n0_optval_2 is a copy
            // of n0_optval that stores the node 0 value for a specific layer of
            // the tree. pgreek is the array saving values for post-calculating
            // Greeks.
            [[intelfpga::register]] double n0_optval[OUTER_UNROLL];
            [[intelfpga::register]] double n0_optval_2[OUTER_UNROLL];
            [[intelfpga::register]] double pgreek[4][OUTER_UNROLL];

            // L1 + L2:
            // Populate init_optval -- calculate the last level of the binomial
            // tree.
            for (short ic = 0; ic < OUTER_UNROLL; ++ic) {
              // Transfer data from DRAM to local memory or registers
              const int c = oc * OUTER_UNROLL + ic;
              const CRRMeta param = accessor_v[c];

              u[ic] = param.u;
              c1[ic] = param.c1;
              c2[ic] = param.c2;
              param_1[ic] = param.param_1;
              param_2[ic] = param.param_2;
              n_steps[ic] = param.n_steps;

              for (short t = steps; t >= 0; --t) {
                const ArrayEle param_array = accessor_v2[c].array_eles[t];

                const double init_val = param_array.init_optval;

                init_optval[t][ic] = init_val;

                // n0_optval intends to store the node value at t == 0.
                // Instead of qualifying this statement by an "if (t == 0)",
                // which couples the loop counter to the timing path of the
                // assignment, we reverse the loop direction so the last value
                // stored corresponds to t == 0.
                n0_optval[ic] = init_val;

                // Transfer data from DRAM to local memory or registers
                u2_array[t][ic] = param_array.u2;
                p1powu_array[t][ic] = param_array.p1powu;
              }
            }

            // L3:
            // Update optval[] -- calculate each level of the binomial tree.
            // reg[] helps to achieve updating INNER_UNROLL elements in optval[]
            // simultaneously.
            [[intelfpga::disable_loop_pipelining]] for (short t = 0;
                                                        t <= steps - 1; ++t) {
              [[intelfpga::register]] double reg[INNER_UNROLL + 1][OUTER_UNROLL];

              double val_1, val_2;

              #pragma unroll
              for (short ic = 0; ic < OUTER_UNROLL; ++ic) {
                reg[0][ic] = n0_optval[ic];
              }

              // L4:
              // Calculate all the elements in optval[] -- all the tree nodes
              // for one level of the tree
              [[intelfpga::ivdep]] for (int n = 0; n <= steps - 1 - t;
                                        n += INNER_UNROLL) {

                #pragma unroll
                for (short ic = 0; ic < OUTER_UNROLL; ++ic) {

                  #pragma unroll
                  for (short ri = 1; ri <= INNER_UNROLL; ++ri) {
                    reg[ri][ic] =
                        (t == 0) ? init_optval[n + ri][ic] : optval[n + ri][ic];
                  }

                  #pragma unroll
                  for (short ri = 0; ri < INNER_UNROLL; ++ri) {
                    const double val = sycl::fmax(
                        c1[ic] * reg[ri][ic] + c2[ic] * reg[ri + 1][ic],
                        p1powu_array[t][ic] * u2_array[n + ri][ic] -
                            param_2[ic]);

                    optval[n + ri][ic] = val;
                    if (n + ri == 0) {
                      n0_optval[ic] = val;
                    }
                    if (n + ri == 1) {
                      val_1 = val;
                    }
                    if (n + ri == 2) {
                      val_2 = val;
                    }
                  }

                  reg[0][ic] = reg[INNER_UNROLL][ic];

                  if (t == steps - 5) {
                    pgreek[3][ic] = val_2;
                  }
                  if (t == steps - 3) {
                    pgreek[0][ic] = n0_optval[ic];
                    pgreek[1][ic] = val_1;
                    pgreek[2][ic] = val_2;
                    n0_optval_2[ic] = n0_optval[ic];
                  }
                }
              }
            }

            // L5: transfer crr_res_paramss to DRAM
            #pragma unroll
            for (short ic = 0; ic < OUTER_UNROLL; ++ic) {
              const int c = oc * OUTER_UNROLL + ic;
              if (n_steps[ic] < steps) {
                accessor_r[c].optval0 = n0_optval_2[ic];
              } else {
                accessor_r[c].optval0 = n0_optval[ic];
              }
              accessor_r[c].pgreek[0] = pgreek[0][ic];
              accessor_r[c].pgreek[1] = pgreek[1][ic];
              accessor_r[c].pgreek[2] = pgreek[2][ic];
              accessor_r[c].pgreek[3] = pgreek[3][ic];
            }
            // Increment counters
            oc += 1;
          } while (oc < n_crr_div);
        });
      });
    }
  }

  double diff = timer.Elapsed();
  return diff;
}

void ReadInputFromFile(ifstream &input_file, vector<InputData> &inp) {
  string line_of_args;
  while (getline(input_file, line_of_args)) {
    InputData temp;
    istringstream line_of_args_ss(line_of_args);
    line_of_args_ss >> temp.n_steps;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.cp;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.spot;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.fwd;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.strike;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.vol;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.df;
    line_of_args_ss.ignore(1, ',');
    line_of_args_ss >> temp.t;

    inp.push_back(temp);
  }
}

static string ToStringWithPrecision(const double value, const int p = 6) {
  ostringstream out;
  out.precision(p);
  out << std::fixed << value;
  return out.str();
}

void WriteOutputToFile(ofstream &output_file, const vector<OutputRes> &outp) {
  size_t n = outp.size();
  for (size_t i = 0; i < n; ++i) {
    OutputRes temp;
    temp = outp[i];
    string line = ToStringWithPrecision(temp.value, 12) + " " +
                  ToStringWithPrecision(temp.delta, 12) + " " +
                  ToStringWithPrecision(temp.gamma, 12) + " " +
                  ToStringWithPrecision(temp.vega, 12) + " " +
                  ToStringWithPrecision(temp.theta, 12) + " " +
                  ToStringWithPrecision(temp.rho, 12) + "\n";

    output_file << line;
  }
}

bool FindGetArgString(const string &arg, const char *str, char *str_value,
                      size_t maxchars) {
  size_t found = arg.find(str, 0, strlen(str));
  if (found != string::npos) {
    const char *sptr = &arg.c_str()[strlen(str)];
    for (int i = 0; i < maxchars - 1; i++) {
      char ch = sptr[i];
      switch (ch) {
        case ' ':
        case '\t':
        case '\0':
          str_value[i] = 0;
          return true;
          break;
        default:
          str_value[i] = ch;
          break;
      }
    }
    return true;
  }
  return false;
}

// Perform data pre-processing work
// Three different option prices are required to solve each CRR problem
// The following lists why each option price is required:
// [0] : Used to compute Premium, Delta, Gamma and Theta
// [1] : Used to compute Rho
// [2] : Used to compute Vega
CRRInParams PrepareData(const InputData &inp) {
  CRRInParams in_params;
  in_params.n_steps = inp.n_steps;

  double r[2];
  r[0] = pow(inp.df, 1.0 / inp.n_steps);
  double d_df = exp(-inp.t * kEpsilon);
  r[1] = pow(inp.df * d_df, 1.0 / inp.n_steps);
  in_params.u[0] = exp(inp.vol * sqrt(inp.t / inp.n_steps));
  in_params.u[1] = in_params.u[0];
  in_params.u[2] = exp((inp.vol + kEpsilon) * sqrt(inp.t / inp.n_steps));

  in_params.u2[0] = in_params.u[0] * in_params.u[0];
  in_params.u2[1] = in_params.u[1] * in_params.u[1];
  in_params.u2[2] = in_params.u[2] * in_params.u[2];
  in_params.umin[0] = inp.spot * pow(1 / in_params.u[0], inp.n_steps + kOpt0);
  in_params.umin[1] = inp.spot * pow(1 / in_params.u[1], inp.n_steps);
  in_params.umin[2] = inp.spot * pow(1 / in_params.u[2], inp.n_steps);
  in_params.c1[0] =
      r[0] * (in_params.u[0] - pow(inp.fwd / inp.spot, 1.0 / inp.n_steps)) /
      (in_params.u[0] - 1 / in_params.u[0]);
  in_params.c1[1] =
      r[1] *(in_params.u[1] - pow((inp.fwd / d_df) / inp.spot, 1.0 / inp.n_steps)) /
      (in_params.u[1] - 1 / in_params.u[1]);
  in_params.c1[2] =
      r[0] * (in_params.u[2] - pow(inp.fwd / inp.spot, 1.0 / inp.n_steps)) /
      (in_params.u[2] - 1 / in_params.u[2]);
  in_params.c2[0] = r[0] - in_params.c1[0];
  in_params.c2[1] = r[1] - in_params.c1[1];
  in_params.c2[2] = r[0] - in_params.c1[2];

  in_params.param_1[0] = inp.cp * in_params.umin[0];
  in_params.param_1[1] = inp.cp * in_params.umin[1];
  in_params.param_1[2] = inp.cp * in_params.umin[2];
  in_params.param_2 = inp.cp * inp.strike;

  return in_params;
}

CRRArrayEles PrepareArrData(const CRRInParams &in) {
  CRRArrayEles arr;

  // Write in reverse t-direction to match kernel access pattern
  for (int i = 0; i <= in.n_steps + kOpt0; ++i) {
    for (int inner_func_index = 0; inner_func_index < 3; ++inner_func_index) {
      arr.array_eles[i][inner_func_index].u2 = pow(in.u2[inner_func_index], i);
      arr.array_eles[i][inner_func_index].p1powu =
          in.param_1[inner_func_index] * pow(in.u[inner_func_index], i + 1);
      arr.array_eles[i][inner_func_index].init_optval =
          fmax(in.param_1[inner_func_index] * pow(in.u2[inner_func_index], i) -
                   in.param_2, 0.0);
    }
  }

  return arr;
}

// Metadata, used in the Kernel, is generated from the input data
// Each CRR problem is split into 3 subproblems to calculate
// each required option price separately
void PrepareKernelData(vector<CRRInParams> &in_params,
                       vector<CRRArrayEles> &array_params,
                       vector<CRRMeta> &in_buff_params,
                       vector<CRRPerStepMeta> &in_buff2_params,
                       const int n_crrs) {

  constexpr short offset = 0;

  for (int wi_idx = offset, dst = offset * 3; wi_idx < n_crrs; ++wi_idx) {
    CRRInParams &src_crr_params = in_params[wi_idx];

    CRRArrayEles &src_crr_eles = array_params[wi_idx];

    for (int inner_func_index = 0; inner_func_index < 3;
         ++inner_func_index, ++dst) {
      CRRMeta &dst_crr_meta = in_buff_params[dst];
      CRRPerStepMeta &dst_crr_per_step_meta = in_buff2_params[dst];

      dst_crr_meta.u = src_crr_params.u[inner_func_index];
      dst_crr_meta.c1 = src_crr_params.c1[inner_func_index];
      dst_crr_meta.c2 = src_crr_params.c2[inner_func_index];

      dst_crr_meta.param_1 = src_crr_params.param_1[inner_func_index];
      dst_crr_meta.param_2 = src_crr_params.param_2;

      if (inner_func_index == 0) {
        dst_crr_meta.n_steps = src_crr_params.n_steps + kOpt0;
      } else {
        dst_crr_meta.n_steps = src_crr_params.n_steps;
      }
      for (int i = 0; i <= kMaxNSteps2; ++i) {
        dst_crr_per_step_meta.array_eles[i].u2 =
            src_crr_eles.array_eles[i][inner_func_index].u2;
        dst_crr_per_step_meta.array_eles[i].p1powu =
            src_crr_eles.array_eles[i][inner_func_index].p1powu;
        dst_crr_per_step_meta.array_eles[i].init_optval =
            src_crr_eles.array_eles[i][inner_func_index].init_optval;
      }
    }
  }
}

// Takes in the result from the kernel and stores the 3 option prices
// belonging to the same CRR problem in one InterRes element
void ProcessKernelResult(const vector<CRRResParams> &res_params,
                         vector<InterRes> &postp_buff, const int n_crrs) {
  constexpr int offset = 0;

  for (int wi_idx = offset, src = offset * 3; wi_idx < n_crrs; ++wi_idx) {
    InterRes &dst_res = postp_buff[wi_idx];

    for (int inner_func_index = 0; inner_func_index < 3;
         ++inner_func_index, ++src) {
      const CRRResParams &src_res = res_params[src];

      for (int i = 0; i < 4; ++i) {
        if (inner_func_index == 0) {
          dst_res.pgreek[i] = src_res.pgreek[i];
        }
      }

      dst_res.vals[inner_func_index] = src_res.optval0;
    }
  }
}

// Computes the Premium and Greeks
OutputRes ComputeOutput(const InputData &inp, const CRRInParams &in_params,
                        const InterRes &res_params) {
  double h;
  OutputRes res;
  h = inp.spot * (in_params.u2[0] - 1 / in_params.u2[0]);
  res.value = res_params.pgreek[1];
  res.delta = (res_params.pgreek[2] - res_params.pgreek[0]) / h;
  res.gamma = 2 / h *
              ((res_params.pgreek[2] - res_params.pgreek[1]) / inp.spot /
                   (in_params.u2[0] - 1) -
               (res_params.pgreek[1] - res_params.pgreek[0]) / inp.spot /
                   (1 - (1 / in_params.u2[0])));
  res.theta =
      (res_params.vals[0] - res_params.pgreek[3]) / 4 / inp.t * inp.n_steps;
  res.rho = (res_params.vals[1] - res.value) / kEpsilon;
  res.vega = (res_params.vals[2] - res.value) / kEpsilon;
  return res;
}

// Perform CRR solving using the CPU and compare FPGA resutls with CPU results
// to test correctness.
void TestCorrectness(int k, int n_crrs, bool &pass, const InputData &inp,
                     CRRInParams &vals, const OutputRes &fpga_res) {
  if (k == 0) {
    std::cout << "\n============= Correctness Test ============= \n";
    std::cout << "Running analytical correctness checks... \n";
  }

  // This CRR benchmark ensures a minimum 4 decimal points match between FPGA and CPU
  // "threshold" is chosen to enforce this guarantee
  float threshold = 0.00001;
  int i, j, q;
  double x;
  int n_steps = vals.n_steps;
  int m = n_steps + kOpt0;
  vector<double> pvalue(kMaxNSteps3);
  vector<double> pvalue_1(kMaxNSteps1);
  vector<double> pvalue_2(kMaxNSteps1);
  vector<double> pgreek(5);
  InterRes cpu_res_params;
  OutputRes cpu_res;

  // option value computed at each final node
  x = vals.umin[0];
  for (i = 0; i <= m; i++, x *= vals.u2[0]) {
    pvalue[i] = fmax(inp.cp * (x - inp.strike), 0.0);
  }

  // backward recursion to evaluate option price
  for (i = m - 1; i >= 0; i--) {
    vals.umin[0] *= vals.u[0];
    x = vals.umin[0];
    for (j = 0; j <= i; j++, x *= vals.u2[0]) {
      pvalue[j] = fmax(vals.c1[0] * pvalue[j] + vals.c2[0] * pvalue[j + 1],
                       inp.cp * (x - inp.strike));
    }
    if (i == 4) {
      pgreek[4] = pvalue[2];
    }
    if (i == 2) {
      for (q = 0; q <= 2; q++) {
        pgreek[q + 1] = pvalue[q];
      }
    }
  }
  cpu_res_params.vals[0] = pvalue[0];

  // the above computation is repeated for each option price
  x = vals.umin[1];
  for (i = 0; i <= n_steps; i++, x *= vals.u2[1]) {
    pvalue_1[i] = fmax(inp.cp * (x - inp.strike), 0.0);
  }

  for (i = n_steps - 1; i >= 0; i--) {
    vals.umin[1] *= vals.u[1];
    x = vals.umin[1];

    for (j = 0; j <= i; j++, x *= vals.u2[1]) {
      pvalue_1[j] =
          fmax(vals.c1[1] * pvalue_1[j] + vals.c2[1] * pvalue_1[j + 1],
               inp.cp * (x - inp.strike));
    }
  }
  cpu_res_params.vals[1] = pvalue_1[0];

  x = vals.umin[2];
  for (i = 0; i <= n_steps; i++, x *= vals.u2[2]) {
    pvalue_2[i] = fmax(inp.cp * (x - inp.strike), 0.0);
  }

  for (i = n_steps - 1; i >= 0; i--) {
    vals.umin[2] *= vals.u[2];
    x = vals.umin[2];
    for (j = 0; j <= i; j++, x *= vals.u2[2]) {
      pvalue_2[j] =
          fmax(vals.c1[2] * pvalue_2[j] + vals.c2[2] * pvalue_2[j + 1],
               inp.cp * (x - inp.strike));
    }
  }
  cpu_res_params.vals[2] = pvalue_2[0];
  pgreek[0] = 0;

  for (i = 1; i < 5; ++i) {
    cpu_res_params.pgreek[i - 1] = pgreek[i];
  }

  cpu_res = ComputeOutput(inp, vals, cpu_res_params);

  if (abs(cpu_res.value - fpga_res.value) > threshold) {
    pass = false;
    std::cout << "fpga_res.value " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.value << "\n";
    std::cout << "cpu_res.value " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.value << "\n";
    std::cout << "Mismatch detected for value of crr " << k << "\n";
  }
  if (abs(cpu_res.delta - fpga_res.delta) > threshold) {
    pass = false;
    std::cout << "fpga_res.delta " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.delta << "\n";
    std::cout << "cpu_res.delta " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.delta << "\n";
    std::cout << "Mismatch detected for value of crr " << k << "\n";
  }
  if (abs(cpu_res.gamma - fpga_res.gamma) > threshold) {
    pass = false;
    std::cout << "fpga_res.gamma " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.gamma << "\n";
    std::cout << "cpu_res.gamma " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.gamma << "\n";
    std::cout << "Mismatch detected for value of crr " << k << "\n";
  }
  if (abs(cpu_res.vega - fpga_res.vega) > threshold) {
    pass = false;
    std::cout << "fpga_res.vega " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.vega << "\n";
    std::cout << "cpu_res.vega " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.vega << "\n";
    std::cout << "Mismatch detected for value of crr " << k << "\n";
  }
  if (abs(cpu_res.theta - fpga_res.theta) > threshold) {
    pass = false;
    std::cout << "fpga_res.theta " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.theta << "\n";
    std::cout << "cpu_res.theta " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.theta << "\n";
    std::cout << "Mismatch detected for value of crr " << k << "\n";
  }
  if (abs(cpu_res.rho - fpga_res.rho) > threshold) {
    pass = false;
    std::cout << "fpga_res.rho " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.rho << "\n";
    std::cout << "cpu_res.rho " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.rho << "\n";
    std::cout << "Mismatch detected for value of crr " << k << "\n";
  }

  if (k == n_crrs - 1) {
    std::cout << "CPU-FPGA Equivalence: " << (pass ? "PASS" : "FAIL") << "\n";
  }
}

// Print out the achieved CRR throughput
void TestThroughput(const double &time, const int &n_crrs) {
  std::cout << "\n============= Throughput Test =============\n";

  std::cout << "   Avg throughput:   " << std::fixed << std::setprecision(1)
            << (n_crrs / time) << " assets/s\n";
}

int main(int argc, char *argv[]) {
  string infilename = "";
  string outfilename = "";

  const string default_ifile = "src/data/ordered_inputs.csv";
  const string default_ofile = "src/data/ordered_outputs.csv";

  char str_buffer[kMaxStringLen] = {0};
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      string sarg(argv[i]);

      FindGetArgString(sarg, "-o=", str_buffer, kMaxStringLen);
      FindGetArgString(sarg, "--output-file=", str_buffer, kMaxStringLen);
    } else {
      infilename = string(argv[i]);
    }
  }

  try {
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector device_selector;
#else
    INTEL::fpga_selector device_selector;
#endif

    queue q(device_selector, dpc_common::exception_handler);

    std::cout << "Running on device:  "
              << q.get_device().get_info<info::device::name>().c_str() << "\n";

    device device = q.get_device();
    std::cout << "Device name: "
              << device.get_info<info::device::name>().c_str() << "\n \n \n";

    vector<InputData> inp;

    // Get input file name, if users don't have their test input file, this
    // design will use the default input file
    if (infilename == "") {
      infilename = default_ifile;
    }
    ifstream inputFile(infilename);

    if (!inputFile.is_open()) {
      std::cerr << "Input file doesn't exist \n";
      return 1;
    }

    // Check input file format
    string filename = infilename;
    std::size_t found = filename.find_last_of(".");
    if (!(filename.substr(found + 1).compare("csv") == 0)) {
      std::cerr << "Input file format only support .csv\n";
      return 1;
    }

    // Get output file name, if users don't define output file name, the design
    // will use the default output file
    outfilename = default_ofile;
    if (strlen(str_buffer)) {
      outfilename = string(str_buffer);
    }

    // Check output file format
    filename = outfilename;
    found = filename.find_last_of(".");
    if (!(filename.substr(found + 1).compare("csv") == 0)) {
      std::cerr << "Output file format only support .csv\n";
      return 1;
    }

    // Read inputs data from input file
    ReadInputFromFile(inputFile, inp);

// Get the number of data from the input file
// Emulator mode only goes through one input (or through OUTER_UNROLL inputs) to
// ensure fast runtime
#if defined(FPGA_EMULATOR)
    int temp_crrs = 1;
#else
    int temp_crrs = inp.size();
#endif

    // Check if n_crrs >= OUTER_UNROLL
    if (OUTER_UNROLL >= temp_crrs) {
      if (inp.size() < OUTER_UNROLL) {
        std::cerr << "Input size must be greater than or equal to OUTER_UNROLL\n";
        return 1;
      } else {
        temp_crrs = OUTER_UNROLL;
      }
    }

    const int n_crrs = temp_crrs;

    vector<CRRInParams> in_params(n_crrs);
    vector<CRRArrayEles> array_params(n_crrs);

    for (int j = 0; j < n_crrs; ++j) {
      in_params[j] = PrepareData(inp[j]);
      array_params[j] = PrepareArrData(in_params[j]);
    }

    // following vectors are arguments for CrrSolver
    vector<CRRMeta> in_buff_params(n_crrs * 3);
    vector<CRRPerStepMeta> in_buff2_params(n_crrs * 3);

    vector<CRRResParams> res_params(n_crrs * 3);
    vector<CRRResParams> res_params_dummy(n_crrs * 3);

    // Prepare metadata as input to kernel
    PrepareKernelData(in_params, array_params, in_buff_params, in_buff2_params,
                      n_crrs);

    // warmup run - use this run to warmup accelerator
    CrrSolver(n_crrs, in_buff_params, res_params_dummy, in_buff2_params,
               q);
    // Timed run - profile performance
    double time = CrrSolver(n_crrs, in_buff_params, res_params,
                             in_buff2_params, q);
    bool pass = true;

    // Postprocessing step
    // process_res used to compute final results
    vector<InterRes> process_res(n_crrs);
    ProcessKernelResult(res_params, process_res, n_crrs);

    vector<OutputRes> result(n_crrs);
    for (int i = 0; i < n_crrs; ++i) {
      result[i] = ComputeOutput(inp[i], in_params[i], process_res[i]);
      TestCorrectness(i, n_crrs, pass, inp[i], in_params[i], result[i]);
    }

    // Write outputs data to output file
    ofstream outputFile(outfilename);

    WriteOutputToFile(outputFile, result);

    TestThroughput(time, n_crrs);

  } catch (sycl::exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly\n";
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR\n";
    return 1;
  }
  return 0;
}
