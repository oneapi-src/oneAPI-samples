#ifndef __QUERY9_KERNEL_HPP__
#define __QUERY9_KERNEL_HPP__
#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "../dbdata.hpp"

using namespace sycl;

bool SubmitQuery9(queue& q, Database& dbinfo,
                  std::string colour,
                  std::array<DBDecimal, 25 * 2020>& sum_profit,
                  double& kernel_latency, double& total_latency);

#endif  //__QUERY9_KERNEL_HPP__
