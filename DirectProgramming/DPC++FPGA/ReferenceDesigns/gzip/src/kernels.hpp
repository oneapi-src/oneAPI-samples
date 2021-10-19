#ifndef __KERNELS_H__
#define __KERNELS_H__
#pragma once

#ifndef NUM_ENGINES
  #define NUM_ENGINES 1
#endif

// BATCH_SIZE is the number of input files the kernel should be capable of
// compressing per invocation of the GZIP engine. This is a compile time
// constant so that hardware is built to support this number. To ensure maximum
// throughput, the minimum batch size should be chosen to cover the latency of
// re-launching the gzip engine, which requires some experimentation on the
// given system. The maximum batch size should be chosen to ensure the gzip
// engine completes execution within the desired execution time (also considered
// the latency to the receive the compile result).
constexpr int BATCH_SIZE = 12;

constexpr int kNumEngines = NUM_ENGINES;

constexpr int kCRCIndex = 0;
constexpr int kLZReductionIndex = 1;
constexpr int kStaticHuffmanIndex = 2;

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

typedef struct char_arr_32 {
  unsigned char arr[32];
} char_arr_32;

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
