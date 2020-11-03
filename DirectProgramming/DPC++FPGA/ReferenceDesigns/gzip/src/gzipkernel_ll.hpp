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
