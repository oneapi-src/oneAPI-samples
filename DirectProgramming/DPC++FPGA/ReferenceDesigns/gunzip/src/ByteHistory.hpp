#ifndef __BYTEHISTORY_HPP__
#define __BYTEHISTORY_HPP__

#include "mp_math.hpp"

/*
TODO
*/
template<unsigned size>
class ByteHistory {
public:
  //
  // Constructor
  //
  ByteHistory() : idx_(0) {}

  //
  // Append to the history
  //
  void Append(char d) {
    // add to the history and move the index
    data_[idx_] = d;
    idx_ = NextIdx(idx_);
  }

  //
  // TODO:
  // multi-element?
  // more generic out stream?
  //
  template<typename Stream>
  void Copy(int dist, int len) {
    // TODO: fpga_tools::IsPow2(size) optimization
    int read_idx = (idx_ - dist) & size;

    //[[intel::ivdep(data_)]]
    for (int i = 0; i < len; i++) {
      // read the history
      auto c = data_[read_idx];

      // stream out the history
      Stream::write(c);

      // append this byte back into the history
      Append(c);

      // move the read index
      read_idx = NextIdx(read_idx);
    }
  }

private:
  char data_[size];
  int idx_;
  static constexpr int size_mask = size - 1;

  //
  // Helper function to move to the next index
  //
  int NextIdx(int idx) {
    if constexpr (fpga_tools::IsPow2(size)) {
      // optimized for power-of-2 size
      return (idx + 1) & size_mask;
    } else {
      // generic version
      return (idx + 1) % size;
    }
  }
};

#endif // __BYTEHISTORY_HPP__