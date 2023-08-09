#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Generate a arithmetic sequence to pipe
template<typename sequence_out>
struct ArithSequence {
  int first_term;
  int length;

  void operator()() const {
    for (int factor = 1; factor < 6; factor ++) {
      // [[intel::max_reinvocation_delay(1)]]
      for (int i = 0; i < length; i ++) {
        int cur_term = first_term + factor * i;
        sequence_out::write(cur_term);
      }
    }
  }
};

// sum up the sequence from pipe
template<typename sequence_in>
struct Summing {
  int* sum;
  int length;

  void operator()() const {
    for (int f = 0; f < 5; f ++) {
      int cur_sum = 0;
      long int i = 1;
      // [[intel::max_reinvocation_delay(1)]]
      while (sycl::log10((double)(i)) < length) {
        cur_sum += sequence_in::read();
        i *= 10;
      }
      sum[f] = cur_sum;
    }
  }
};

int get_arithmetic_sum(int first_term, int factor, int length) {
  return (int)((length / 2) * (2 * first_term + (length - 1) * factor));
}

class SequencePipe;
class SequenceGen;
class SequenceSum;

int main() {
  int start_term = 0;
  int seq_length = 10;
  int final_sum[5];
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
  // create the device queue
  sycl::queue q(selector);

  auto device = q.get_device();

  std::cout << "Running on device: "
      << device.get_info<sycl::info::device::name>().c_str()
      << std::endl;

  using SeqPipe = sycl::ext::intel::pipe<SequencePipe, int, 10>;

  int* sum_shared = sycl::malloc_shared<int>(10, q);
  
  auto e = q.single_task<SequenceGen>(ArithSequence<SeqPipe>{start_term, seq_length});

  // q.single_task<class Consumer>([=]() {
  //   while (1) {
  //     SeqPipe::read();
  //   }
  // });

  // e.wait();

  q.single_task<SequenceSum>(Summing<SeqPipe>{sum_shared, seq_length}).wait();

  q.memcpy(final_sum, sum_shared, 10 * sizeof(int)).wait();
  // check for correctness
  for (int f = 0; f < 5; f ++) {
    int exp_sum = get_arithmetic_sum(start_term, f+1, seq_length);
    if (final_sum[f] != exp_sum) {
      std::cout << "Factor: " << (f+1) << " ,length: " << seq_length << std::endl;
      std::cout << "FAILED, expect " << exp_sum << " get " << final_sum[f] << std::endl;
    } else {
      std::cout << "SUCCESS, sum = " << final_sum[f] << std::endl;
    }
  }
  std::cout << "\nDone\n";
  free(sum_shared, q);
}