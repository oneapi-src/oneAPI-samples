#ifndef __GZIP_DECOMPRESSOR_HPP__
#define __GZIP_DECOMPRESSOR_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "common.hpp"
#include "constexpr_math.hpp"
#include "gzip_metadata_reader.hpp"
#include "huffman_decoder.hpp"
#include "byte_stacker.hpp"
#include "lz77_decoder.hpp"

using namespace sycl;

// declare the kernel and pipe names globally to reduce name mangling
class GzipMetadataReaderKernelID;
class HuffmanDecoderKernelID;
class LZ77DecoderKernelID;
class ByteStackerKernelID;

class GzipMetadataToHuffmanPipeID;
class HuffmanToLZ77PipeID;
class LZ77ToByteStackerPipeID;

// the depth of the pipe between the Huffman decoder and the LZ77 decoder.
// adding some extra depth here helps add some elasticity so that the Huffman
// decoder can computing while the LZ77 kernel reads from the history buffer
constexpr int kHuffmanToLZ77PipeDepth = 64;

//
// Submits the kernels for the GZIP decompression engine and returns the
// SYCL events for each kernel
//
template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
std::vector<event> SubmitGzipDecompressKernels(queue& q, int in_count,
                                               GzipHeaderData *hdr_data_out,
                                               int *crc_out, int *count_out) {
  static_assert(literals_per_cycle > 0);
  static_assert(fpga_tools::IsPow2(literals_per_cycle));
  
  // the inter-kernel pipes for the GZIP decompression engine
  using GzipMetadataToHuffmanPipe =
    ext::intel::pipe<GzipMetadataToHuffmanPipeID, FlagBundle<unsigned char>>;
  using HuffmanToLZ77Pipe =
    ext::intel::pipe<HuffmanToLZ77PipeID, FlagBundle<GzipLZ77InputData<2>>, kHuffmanToLZ77PipeDepth>;
  using LZ77ToByteStackerPipe =
    ext::intel::pipe<LZ77ToByteStackerPipeID, FlagBundle<BytePack<literals_per_cycle>>>;

  // submit the GZIP decompression kernels
  auto header_event = SubmitGzipMetadataReader<GzipMetadataReaderKernelID, InPipe, GzipMetadataToHuffmanPipe>(q, in_count, hdr_data_out, crc_out, count_out);
  auto huffman_event = SubmitHuffmanDecoder<HuffmanDecoderKernelID, GzipMetadataToHuffmanPipe, HuffmanToLZ77Pipe>(q);
  auto lz77_event = SubmitLZ77Decoder<LZ77DecoderKernelID, HuffmanToLZ77Pipe, LZ77ToByteStackerPipe, literals_per_cycle>(q);
  auto byte_stacker_event = SubmitByteStacker<ByteStackerKernelID, LZ77ToByteStackerPipe, OutPipe, literals_per_cycle>(q);

  return {header_event, huffman_event, lz77_event, byte_stacker_event};
}

#endif /* __GZIP_DECOMPRESSOR_HPP__ */