#ifndef __SIMPLE_CRC32_HPP__
#define __SIMPLE_CRC32_HPP__

//
// A simple CRC-32 implementation (not optimized for high performance).
// Compute CRC-32 on 'len' elements in 'buf', starting with a CRC of 'init'.
//
// Arguments:
//    init: the initial CRC value. This is used to string together multiple
//      calls to SimpleCRC32. For the first iteration, use 0.
//    buf: a pointer to the data
//    len: the number of bytes pointer to by 'buf'
//
unsigned int SimpleCRC32(unsigned init, const void* buf, size_t len) {
  // generate the 256-element table
  constexpr uint32_t polynomial = 0xEDB88320;
  constexpr auto table = [] {
    std::array<uint32_t, 256> a{};
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (uint32_t j = 0; j < 8; j++) {
        if (c & 1) {
          c = polynomial ^ (c >> 1);
        } else {
          c >>= 1;
        }
      }
      a[i] = c;
    }
    return a;
  }();

  // compute the CRC-32 for the input data
  unsigned c = init ^ 0xFFFFFFFF;
  const uint8_t* u = static_cast<const uint8_t*>(buf);
  for (size_t i = 0; i < len; i++) {
    c = table[(c ^ u[i]) & 0xFF] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFF;
}

#endif /* __SIMPLE_CRC32_HPP__ */