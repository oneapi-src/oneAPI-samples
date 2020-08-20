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

#define _CRT_SECURE_NO_WARNINGS
#include "WriteGzip.hpp"

#include <fcntl.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <CL/sycl.hpp>
#include <chrono>
#include <string>

constexpr int kDeflated = 8;
#define GZIP_MAGIC "\037\213"  // Magic header for gzip files, 1F 8B

#define ORIG_NAME 0x08
#define OS_CODE 0x03  // Unix OS_CODE

typedef struct GzipHeader {
  unsigned char magic[2];         // 0x1f, 0x8b
  unsigned char compress_method;  // 0-7 reserved, 8=deflate -- kDeflated
  unsigned char flags;            // b0: file probably ascii
                                  // b1: header crc-16 present
                                  // b2: extra field present
                                  // b3: original file name present
                                  // b4: file comment present
                                  // b5,6,7: reserved
  unsigned long time;             // file modification time in Unix format.
                                  // Set this to 0 for now.

  unsigned char extra;  // depends on compression method
  unsigned char os;     // operating system on which compression took place

  // ...
  //  ? bytes ... compressd data ...

  unsigned long crc;
  unsigned long uncompressed_sz;

} gzip_header, *pgzip_header;

inline static void PutUlong(uint8_t *pc, unsigned long l) {
  pc[0] = l & 0xff;
  pc[1] = (l >> 8) & 0xff;
  pc[2] = (l >> 16) & 0xff;
  pc[3] = (l >> 24) & 0xff;
}

// returns 0 on success, otherwise failure
int WriteBlockGzip(
    std::string &original_filename,  // Original file name being compressed
    std::string &out_filename,       // gzip filename
    char *obuf,                      // pointer to compressed data block
    size_t blen,                     // length of compressed data block
    size_t ilen,                     // original block length
    uint32_t buffer_crc)             // the block's crc
{
  //------------------------------------------------------------------
  // Setup the gzip output file header.
  //  max filename size is arbitrarily set to 256 bytes long
  //  Method is always DEFLATE
  //  Original filename is always set in header
  //  timestamp is set to 0 - ignored by gunzip
  //  deflate flags set to 0
  //  OS code is 0

  int max_filename_sz = 256;

  unsigned char *pgziphdr =
      (unsigned char *)malloc(sizeof(gzip_header) + max_filename_sz);

  if (!pgziphdr) {
    std::cout << "pgzip header cannot be allocated\n";
    return 1;
  }

  pgziphdr[0] = GZIP_MAGIC[0];
  pgziphdr[1] = GZIP_MAGIC[1];
  pgziphdr[2] = kDeflated;
  pgziphdr[3] = ORIG_NAME;

  // Set time in header to 0, this is ignored by gunzip.
  pgziphdr[4] = 0;
  pgziphdr[5] = 0;
  pgziphdr[6] = 0;
  pgziphdr[7] = 0;

  // Deflate flags
  pgziphdr[8] = 0;

  // OS code is Linux in this case.
  pgziphdr[9] = OS_CODE;

  int ondx = 10;

  const char *p = original_filename.c_str();
  do {
    pgziphdr[ondx++] = (*p);
  } while (*p++);

  int header_bytes = ondx;

  unsigned char prolog[8];

  PutUlong(((unsigned char *)prolog), buffer_crc);
  PutUlong(((unsigned char *)&prolog[4]), ilen);

  FILE *fo = fopen(out_filename.c_str(), "w+");
  if (ferror(fo)) {
    std::cout << "Cannot open file for output: " << out_filename << "\n";
    free(pgziphdr);
    return 1;
  }

  fwrite(pgziphdr, 1, header_bytes, fo);
  fwrite(obuf, 1, blen, fo);
  fwrite(prolog, 1, 8, fo);

  if (ferror(fo)) {
    std::cout << "gzip output file write failure.\n";
    free(pgziphdr);
    return 1;
  }

  if (fclose(fo)) {
    perror("close");
    free(pgziphdr);
    return 1;
  }
  free(pgziphdr);
  return 0;
}
