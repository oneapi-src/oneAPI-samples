#ifndef __GZIP_DECOMPRESSOR_HPP__
#define __GZIP_DECOMPRESSOR_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "common.hpp"
#include "header_reader.hpp"
#include "huffman_decoder.hpp"
#include "literal_stacker.hpp"
#include "lz77_decoder.hpp"

using namespace sycl;

// declare the kernel and pipe names globally to reduce name mangling
class GzipHeaderReaderKernelID;
class HuffmanDecoderKernelID;
class LZ77DecoderKernelID;
class LiteralStackerKernelID;

class GzipHeaderToHuffmanPipeID;
class HuffmanToLZ77PipeID;
class LZ77ToLiteralStackerPipeID;

constexpr int kHuffmanToLZ77PipeDepth = 64;

//
// Submits the kernels for the GZIP decompression engine and returns the
// SYCL events for each kernel
//
template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
std::vector<event> SubmitGzipDecompressKernels(queue& q, int in_count,
                                               GzipHeaderData *hdr_data_out,
                                               int *crc_out, int *size_out) {
  // the inter-kernel pipes for the GZIP decompression engine
  using GzipHeaderToHuffmanPipe =
    ext::intel::pipe<GzipHeaderToHuffmanPipeID, FlagBundle<unsigned char>>;
  using HuffmanToLZ77Pipe =
    ext::intel::pipe<HuffmanToLZ77PipeID, FlagBundle<HuffmanData>, kHuffmanToLZ77PipeDepth>;
  using LZ77ToLiteralStackerPipe =
    ext::intel::pipe<LZ77ToLiteralStackerPipeID, FlagBundle<LiteralPack<literals_per_cycle>>>;

  // submit the GZIP decompression kernels
  auto header_event = SubmitGzipHeaderReader<GzipHeaderReaderKernelID, InPipe, GzipHeaderToHuffmanPipe>(q, in_count, hdr_data_out, crc_out, size_out);
  auto huffman_event = SubmitHuffmanDecoder<HuffmanDecoderKernelID, GzipHeaderToHuffmanPipe, HuffmanToLZ77Pipe>(q);
  auto lz77_event = SubmitLZ77Decoder<LZ77DecoderKernelID, HuffmanToLZ77Pipe, LZ77ToLiteralStackerPipe, literals_per_cycle>(q);
  auto lit_stacker_event = SubmitLiteralStacker<LiteralStackerKernelID, LZ77ToLiteralStackerPipe, OutPipe, literals_per_cycle>(q);

  return {header_event, huffman_event, lz77_event, lit_stacker_event};
}

#endif /* __GZIP_DECOMPRESSOR_HPP__ */