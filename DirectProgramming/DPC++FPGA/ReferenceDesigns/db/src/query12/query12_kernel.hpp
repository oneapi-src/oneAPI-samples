#ifndef __QUERY12_KERNEL_HPP__
#define __QUERY12_KERNEL_HPP__
#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "../dbdata.hpp"

using namespace sycl;

bool SubmitQuery12(queue& q, Database& dbinfo,
                   DBDate low_date, DBDate high_date,
                   int shipmode1, int shipmode2,
                   std::array<DBDecimal, 2>& high_line_count,
                   std::array<DBDecimal, 2>& low_line_count,
                   double& kernel_latency, double& total_latency);

#endif  //__QUERY12_KERNEL_HPP__
