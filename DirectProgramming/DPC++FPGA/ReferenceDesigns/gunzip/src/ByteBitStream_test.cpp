#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include <algorithm>
#include <iostream>

#include "ByteBitStream.hpp"

class ProducerID;
class ConsumerID;
class KernelID;
class InPipeID;
class OutPipeID;

using InPipe = ext::intel::pipe<InPipeID, char>;
using OutPipe = ext::intel::pipe<OutPipeID, char>;

constexpr int bits_to_write_array[2] = {3, 5};

event SubmitProducer(queue& q, unsigned char* in_ptr, int count) {
  return q.single_task<ProducerID>([=] {
    device_ptr<unsigned char> in(in_ptr);

    for (int i = 0; i < count; i++) {
      auto tmp = in[i];
      InPipe::write(tmp);
    }
  });
}

event SubmitConsumer(queue& q, unsigned char* out_ptr, int count) {
  return q.single_task<ConsumerID>([=] {
    device_ptr<unsigned char> out(out_ptr);

    for (int i = 0; i < count*2; i++) {
      auto tmp = OutPipe::read();
      out[i] = tmp;
    }
  });
}

event SubmitKernel(queue& q, int count) {
  return q.single_task<KernelID>([=] {
    unsigned char byte = 0;
    ByteBitStream bbs;
    int bytes_read = 0;
    int bytes_written = 0;
    int bits_to_write_array[2] = {3, 5};

    while (bytes_read < count || bytes_written < 2*count || !bbs.Empty()) {
      if (bytes_read < count && bbs.HasSpaceForByte()) {
        unsigned char new_byte = InPipe::read();
        bbs.NewByte(new_byte);
        bytes_read++;
      }

      int bits_to_write = bits_to_write_array[bytes_written & 0x1];
      if (bytes_written < 2*count && bbs.HasEnoughBits(bits_to_write)) {
        unsigned char bits = bbs.ReadUInt(bits_to_write);
        OutPipe::write(bits);
        bytes_written++;
      }
    }
  });
}

int main(int argc, char* argv[]) {
  int count = 32;
  if(argc > 1) {
    count = atoi(argv[1]);
  }

  std::vector<unsigned char> in_bytes(count);
  std::vector<unsigned char> out_bytes(count*2);

  std::generate(in_bytes.begin(), in_bytes.end(), [] { return rand() % 100; } );
  std::fill(out_bytes.begin(), out_bytes.end(), 0);

  // the device selector
#ifdef FPGA_EMULATOR
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  queue q(selector);

  unsigned char* in = malloc_device<unsigned char>(count, q);
  unsigned char* out = malloc_device<unsigned char>(count * 2, q);
  assert(in != nullptr);
  assert(out != nullptr);

  q.memcpy(in, in_bytes.data(), count * sizeof(unsigned char)).wait();

  auto pe = SubmitProducer(q, in, count);
  auto ce = SubmitConsumer(q, out, count);
  auto ke = SubmitKernel(q, count);

  pe.wait();
  ce.wait();
  ke.wait();

  q.memcpy(out_bytes.data(), out, 2* count * sizeof(unsigned char)).wait();

  bool passed = true;
  std::vector<unsigned char> rebuilt_out(count);
  for (int i = 0; i < count; i++) {
    unsigned char ref = in[i];
    //unsigned char rebuilt = (out[2 * i] << 5) | out[2 * i + 1];
    unsigned char rebuilt = (out[2 * i + 1] << 3) | out[2 * i];
    if (ref != rebuilt) {
      fprintf(stderr, "ERROR at %d: 0x%02X != 0x%02X\n", i, ref, rebuilt);
      passed = false;
    }
  }

  if (passed) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
  }
}