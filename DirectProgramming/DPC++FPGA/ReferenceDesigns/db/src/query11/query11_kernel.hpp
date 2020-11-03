#ifndef __QUERY11_KERNEL_HPP__
#define __QUERY11_KERNEL_HPP__
#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "../dbdata.hpp"

using namespace sycl;

bool SubmitQuery11(queue& q, Database& dbinfo,
                   std::string& nation,
                   std::vector<DBIdentifier>& partkeys,
                   std::vector<DBDecimal>& values,
                   double& kernel_latency, double& total_latency);

#endif  //__QUERY11_KERNEL_HPP__
