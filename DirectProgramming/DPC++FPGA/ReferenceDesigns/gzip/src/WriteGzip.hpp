#ifndef __WRITEGZIP_H__
#define __WRITEGZIP_H__
#pragma once

#include <iostream>
#include <string>

// returns 0 on success, otherwise failure
int WriteBlockGzip(
    std::string &original_filename,  // Original file name being compressed
    std::string &out_filename,       // gzip filename
    char *obuf,                      // pointer to compressed data block
    size_t blen,                     // length of compressed data block
    size_t ilen,                     // original block length
    uint32_t buffer_crc);            // the block's crc

#endif  //__WRITEGZIP_H__
