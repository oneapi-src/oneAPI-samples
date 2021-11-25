#ifndef __GZIPKERNEL_H__
#define __GZIPKERNEL_H__
#pragma once

#include <CL/sycl.hpp>

using namespace cl::sycl;

extern "C" void SubmitGzipTasks(
    queue &sycl_device,
    size_t block_size,  // size of block to compress.
    buffer<char, 1> *pibuf, buffer<char, 1> *pobuf,
    buffer<struct GzipOutInfo, 1> *gzip_out_buf,
    buffer<unsigned, 1> *current_crc, bool last_block, event &e_crc,
    event &e_lz, event &e_huff, size_t engineID);

#endif  //__GZIPKERNEL_H__
