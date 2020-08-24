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

#ifndef __KERNELS_H__
#define __KERNELS_H__
#pragma once

#ifndef NUM_ENGINES
  #define NUM_ENGINES 1
#endif

constexpr int kNumEngines = NUM_ENGINES;

// kVecPow == 2 means kVec == 4.
// kVecPow == 3 means kVec == 8.
// kVecPow == 4 means kVec == 16.
constexpr int kVecPow = 4;

constexpr int kVec = 1 << kVecPow;
constexpr int kVecX2 = 2 * kVec;

constexpr int kHufTableSize = 256;

// Maximum length of huffman codes
constexpr int kMaxHuffcodeBits = 16;

struct Uint2Gzip {
  unsigned int y;
  unsigned int x;
};

struct LzInput {
  unsigned char data[kVec];
};

typedef struct DistLen {
  unsigned char data[kVec];
  char len[kVec];
  short dist[kVec];
} DistLen, *pdist_len_t;

struct HuffmanOutput {
  unsigned int data[kVec];
  bool write;
};

struct TrailingOutput {
  int bytecount_left;
  int bytecount;
  unsigned char bytes[kVec * sizeof(unsigned int)];
};

struct GzipOutInfo {
  // final compressed block size
  size_t compression_sz;
  unsigned long crc;
};

// kLen must be == kVec
constexpr int kLen = kVec;

// depth of the dictionary buffers
constexpr int kDepth = 512;

// Assumes kDepth is a power of 2 number.
constexpr int kHashMask = kDepth - 1;

#define CONSTANT __constant

constexpr int kDebug = 1;
#define TRACE(x)          \
  do {                    \
    if (kDebug) printf x; \
  } while (0)

constexpr int kStaticTrees = 1;

typedef struct CtData {
  unsigned short code;
  unsigned short len;
} CtData;

constexpr int kMaxMatch = 258;
constexpr int kMinMatch = 3;

constexpr int kTooFar = 4096;

// All codes must not exceed kMaxBits
constexpr int kMaxBits = 15;

// number of length codes, not counting the special kEndBlock code
constexpr int kLengthCodes = 29;

// number of literal bytes, 0..255
constexpr int kLiterals = 256;

// end of literal code block
constexpr int kEndBlock = 256;

// number of literal or length codes, including kEndBlock
constexpr int kLCodes = (kLiterals + 1 + kLengthCodes);

// number of distance codes
constexpr int kDCodes = 30;

// number of codes used to transfer the bit lengths
constexpr int kBLCodes = 19;

constexpr int kMaxDistance = ((32 * 1024));

constexpr int kMinBufferSize = 16384;

struct DictString {
  unsigned char s[kLen];
};

// Mapping from a distance to a distance code. dist is the distance - 1 and
// must not have side effects. dist_code[256] and dist_code[257] are never
// used.
#define d_code(dist) \
  ((dist) < 256 ? dist_code[dist] : dist_code[256 + ((dist) >> 7)])

#endif  //__KERNELS_H__
