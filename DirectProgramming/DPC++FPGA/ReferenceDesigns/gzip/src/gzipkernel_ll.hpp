#ifndef __GZIPKERNEL_H__
#define __GZIPKERNEL_H__
#pragma once

#include <CL/sycl.hpp>
#include "kernels.hpp"

using namespace cl::sycl;

//extern "C" 
std::vector<event> SubmitGzipTasks(queue &q, size_t block_size,
                                   struct GzipOutInfo *gzip_out_buf,
                                   uint32_t *result_crc, bool last_block,
                                   std::vector<event> depend_on,
                                   std::array<char *, BATCH_SIZE> in_ptrs,
                                   std::array<char *, BATCH_SIZE> out_ptrs,
                                   size_t engineID);

#endif  //__GZIPKERNEL_H__
