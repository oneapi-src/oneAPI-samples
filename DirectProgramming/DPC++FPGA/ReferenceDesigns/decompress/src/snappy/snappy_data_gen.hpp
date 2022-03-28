#ifndef __SNAPPY_DATA_GEN_HPP__
#define __SNAPPY_DATA_GEN_HPP__

#include <vector>

//
// Function to generate compressed Snappy data.
// Generates a file as follows:
//    'num_lit_strs' literal strings of length 'lit_str_len'
//    'num_copies' of length 'copy_len' and offset max(16k, lit_str_len)
//    'repeat' copies of the above.
//
std::vector<unsigned char> GenerateSnappyCompressedData(unsigned lit_str_len,
                                                        unsigned num_lit_strs,
                                                        unsigned copy_len,
                                                        unsigned num_copies,
                                                        unsigned repeats) {
  // error checking the input arguments
  if (lit_str_len <= 0) {
    std::cerr << "ERROR: 'lit_str_len' must be greater than 0" << std::endl;
    std::terminate();
  }
  if (num_lit_strs <= 0) {
    std::cerr << "ERROR: 'num_lit_strs' must be greater than 0" << std::endl;
    std::terminate();
  }
  if (copy_len > 64) {
    std::cerr << "ERROR: 'copy_len' must be less than or equal to 64"
              << std::endl;
    std::terminate();
  }
  if (repeats <= 0) {
    std::cerr << "ERROR: 'repeats' must be greater than 0" << std::endl;
    std::terminate();
  }
  if (num_copies > 0 && copy_len <= 0) {
    std::cerr << "ERROR: if 'num_copies' is non-zero, then 'copy_len' must be "
              << "greater than 0" << std::endl;
    std::terminate();
  }

  // the expected uncompressed length
  unsigned uncompressed_length =
      (lit_str_len * num_lit_strs + copy_len * num_copies) * repeats;

  std::vector<unsigned char> ret;

  // the "smart" data we will fill our dummy buffer with ... ;)
  constexpr unsigned char dummy_alphabet[] = {'I', 'N', 'T', 'E', 'L'};
  constexpr unsigned dummy_alphabet_count =
      sizeof(dummy_alphabet) / sizeof(dummy_alphabet[0]);

  // lambda to convert an unsigned int to a byte array
  auto unsigned_to_byte_array = [](unsigned val) {
    std::vector<unsigned char> arr(4);
    for (int i = 0; i < 4; i++) {
      arr[i] = (val >> (i * 8)) & 0xFF;
    }
    return arr;
  };

  // generate the preamble: the uncompressed length varint
  // see the README for more information on what a varint is
  unsigned uncompressed_length_bytes = 0;
  unsigned uncompressed_length_varint = 0;
  while (uncompressed_length != 0) {
    auto data = uncompressed_length & 0x7F;
    auto uncompressed_length_next = uncompressed_length >> 7;
    unsigned more_bytes = (uncompressed_length_next != 0) ? 1 : 0;
    auto mask = (more_bytes << 7) | data;

    uncompressed_length_varint |= mask << (uncompressed_length_bytes * 8);
    uncompressed_length_bytes++;
    uncompressed_length = uncompressed_length_next;
  }

  // error check the result converting the uncompressed length to a varint
  if (uncompressed_length_bytes > 5) {
    std::cerr << "ERROR: generating the preamble, uncompressed_length_bytes = "
              << uncompressed_length_bytes << "\n";
    std::terminate();
  }
  if (uncompressed_length_varint <= 0) {
    std::cerr << "ERROR: generating the preamble, uncompressed_length_varint = "
              << uncompressed_length_varint << "\n";
    std::terminate();
  }

  // convert the varint to an array of bytes and add them to the output
  auto uncompressed_length_varint_bytes =
      unsigned_to_byte_array(uncompressed_length_varint);
  for (int i = 0; i < uncompressed_length_bytes; i++) {
    ret.push_back(uncompressed_length_varint_bytes[i]);
  }

  // determine the literal string and copy tag byte once, since it won't change
  // across the 'repeats' iterations of the loop to generate the data
  constexpr unsigned char lit_str_tag = 0;
  unsigned lit_str_byte_count = 0;
  unsigned char lit_str_bytes[5];

  if (lit_str_len <= 60) {
    // write the literal string tag byte
    lit_str_bytes[0] = ((lit_str_len - 1) << 2) | lit_str_tag;
    lit_str_byte_count = 1;
  } else {
    // how many bytes are needed to store the literal length
    unsigned lit_str_extra_byte_count = 1;
    while ((1 << (lit_str_extra_byte_count * 8)) < lit_str_len) {
      lit_str_extra_byte_count += 1;
    }

    // store the tag byte
    auto length_bytes_mask = 60 + lit_str_extra_byte_count - 1;
    lit_str_bytes[0] = (length_bytes_mask << 2) | lit_str_tag;

    // store the extra bytes
    auto lit_len_byte_array = unsigned_to_byte_array(lit_str_len - 1);
    for (int j = 0; j < lit_str_extra_byte_count; j++) {
      lit_str_bytes[j + 1] = lit_len_byte_array[j];
    }

    lit_str_byte_count = lit_str_extra_byte_count + 1;
  }

  // generate the compressed data
  for (int i = 0; i < repeats; i++) {
    // literal strings
    for (int j = 0; j < num_lit_strs; j++) {
      // write the literal tag byte and optional extra bytes for the length
      for (int k = 0; k < lit_str_byte_count; k++) {
        ret.push_back(lit_str_bytes[k]);
      }

      // write the literals following the literal tag byte
      for (int k = 0; k < lit_str_len; k++) {
        ret.push_back(dummy_alphabet[k % dummy_alphabet_count]);
      }
    }

    // copies
    for (int j = 0; j < num_copies; j++) {
      // the copy tag byte (always 2 byte copies)
      constexpr unsigned char copy_tag_type = 2;
      unsigned char tag_byte = ((copy_len - 1) << 2) | copy_tag_type;
      ret.push_back(tag_byte);

      // the extra 2 bytes for the offset
      unsigned offset = std::min(16383U, lit_str_len - 1);
      auto offset_bytes = unsigned_to_byte_array(offset);
      ret.push_back(offset_bytes[1]);
      ret.push_back(offset_bytes[0]);
    }
  }

  return ret;
}

#endif /* __SNAPPY_DATA_GEN_HPP__ */