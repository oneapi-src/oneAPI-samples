#ifndef __GZIPKERNEL_H__
#define __GZIPKERNEL_H__
#pragma once

#include <sycl/sycl.hpp>

using namespace sycl;

extern "C" void SubmitGzipTasks(
    queue &sycl_device,
    size_t block_size,  // size of block to compress.
    buffer<char, 1> *pibuf, buffer<char, 1> *pobuf,
    buffer<struct GzipOutInfo, 1> *gzip_out_buf,
    buffer<unsigned, 1> *current_crc, bool last_block, std::vector<event> &e_crc,
    std::vector<event> &e_lz, std::vector<event> &e_huff, size_t engineID, int buffer_index);

#endif  //__GZIPKERNEL_H__
