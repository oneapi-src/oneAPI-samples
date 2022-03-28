#ifndef __MEMORY_UTILS_HPP__
#define __MEMORY_UTILS_HPP__

#include <type_traits>

#include "metaprogramming_utils.hpp"

//
// The utilities in this file are used for converting streaming data to/from
// memory from/to a pipe.
//

namespace fpga_tools {

namespace detail {
//
// Streams data from 'in_ptr' into 'Pipe', 'elements_per_cycle' elements at a
// time
//
template<typename Pipe, int elements_per_cycle, typename PtrT>
void MemoryToPipeRemainder(PtrT in_ptr, size_t full_count, size_t remainder_count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(std::is_same_v<std::remove_reference_t<decltype(std::declval<PipeT>()[0])>, std::remove_const_t<std::remove_reference_t<decltype(std::declval<PtrT>()[0])>>>);

  for (size_t i = 0; i < full_count; i++) {
    PipeT pipe_data;
    #pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      pipe_data[j] = in_ptr[i * elements_per_cycle + j];
    }
    Pipe::write(pipe_data);
  }

  PipeT pipe_data;
  for (size_t i = 0; i < remainder_count; i++) {
    pipe_data[i] = in_ptr[full_count * elements_per_cycle + i];
  }
  Pipe::write(pipe_data);
}

//
// Streams data from 'in_ptr' into 'Pipe', 'elements_per_cycle' elements at a
// time with the guarantee that 'elements_per_cycle' is a multiple of 'count'
//
template<typename Pipe, int elements_per_cycle, typename PtrT>
void MemoryToPipeNoRemainder(PtrT in_ptr, size_t count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(std::is_same_v<std::remove_reference_t<decltype(std::declval<PipeT>()[0])>, std::remove_const_t<std::remove_reference_t<decltype(std::declval<PtrT>()[0])>>>);

  for (size_t i = 0; i < count; i++) {
    PipeT pipe_data;
    #pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      pipe_data[j] = in_ptr[i * elements_per_cycle + j];
    }
    Pipe::write(pipe_data);
  }
}

//
// Streams data from 'Pipe' to 'out_ptr', 'elements_per_cycle' elements at a
// time
//
template<typename Pipe, int elements_per_cycle, typename PtrT>
void PipeToMemoryRemainder(PtrT out_ptr, size_t full_count, size_t remainder_count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(std::is_same_v<std::remove_reference_t<decltype(std::declval<PipeT>()[0])>, std::remove_const_t<std::remove_reference_t<decltype(std::declval<PtrT>()[0])>>>);

  for (size_t i = 0; i < full_count; i++) {
    auto pipe_data = Pipe::read();
    #pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      out_ptr[i * elements_per_cycle + j] = pipe_data[j];
    }
  }

  auto pipe_data = Pipe::read();
  for (size_t i = 0; i < remainder_count; i++) {
    out_ptr[full_count * elements_per_cycle + i] = pipe_data[i];
  }
}

//
// Streams data from 'Pipe' to 'out_ptr', 'elements_per_cycle' elements at a
// time with the guarantee that 'elements_per_cycle' is a multiple of 'count'
//
template<typename Pipe, int elements_per_cycle, typename PtrT>
void PipeToMemoryNoRemainder(PtrT out_ptr, size_t count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(std::is_same_v<std::remove_reference_t<decltype(std::declval<PipeT>()[0])>, std::remove_const_t<std::remove_reference_t<decltype(std::declval<PtrT>()[0])>>>);

  for (size_t i = 0; i < count; i++) {
    auto pipe_data = Pipe::read();
    #pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      out_ptr[i * elements_per_cycle + j] = pipe_data[j];
    }
  }
}

}  // namespace detail

//
// Streams data from memory to a SYCL pipe 1 element a time
//
template<typename Pipe, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(std::is_same_v<PipeT, std::remove_const_t<std::remove_reference_t<decltype(std::declval<PtrT>()[0])>>>);

  for (size_t i = 0; i < count; i++) {
    Pipe::write(in_ptr[i]);
  }
}

//
// Streams data from memory to a SYCL pipe 'elements_per_cycle' elements a time
//
template<typename Pipe, int elements_per_cycle, bool no_remainder, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t count) {
  if constexpr (no_remainder) {
    // user promises there is not remainder
    detail::MemoryToPipeNoRemainder<Pipe, elements_per_cycle>(in_ptr, count);
  } else {
    // might have a remainder and it was not specified, so calculate it
    auto full_count = (count / elements_per_cycle) * elements_per_cycle;
    auto remainder_count = count % elements_per_cycle;
    detail::MemoryToPipeRemainder<Pipe, elements_per_cycle>(in_ptr, full_count, remainder_count);
  }
}

//
// Streams data from memory to a SYCL pipe 'elements_per_cycle' elements a time
// In this version, the user has specified a the amount of remainder
//
template<typename Pipe, int elements_per_cycle, bool no_remainder, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t full_count, size_t remainder_count) {
  if constexpr (no_remainder) {
    // user promises there is not remainder
    detail::MemoryToPipeNoRemainder<Pipe, elements_per_cycle>(in_ptr, full_count);
  } else {
    // might have a remainder that was specified by the user
    detail::MemoryToPipeRemainder<Pipe, elements_per_cycle>(in_ptr, full_count, remainder_count);
  }
}

//
// Streams data from a SYCL pipe to memory 1 element a time
//
template<typename Pipe, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(std::is_same_v<PipeT, std::remove_const_t<std::remove_reference_t<decltype(std::declval<PtrT>()[0])>>>);

  for (size_t i = 0; i < count; i++) {
    out_ptr[i] = Pipe::read();
  }
}

//
// Streams data from a SYCL pipe to memory 'elements_per_cycle' elements a time
//
template<typename Pipe, int elements_per_cycle, bool no_remainder, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t count) {
  if constexpr (no_remainder) {
    detail::PipeToMemoryNoRemainder<Pipe, elements_per_cycle>(out_ptr, count);
  } else {
    auto full_count = (count / elements_per_cycle) * elements_per_cycle;
    auto remainder_count = count % elements_per_cycle;
    detail::PipeToMemoryRemainder<Pipe, elements_per_cycle>(out_ptr, full_count, remainder_count);
  }
}

//
// Streams data from a SYCL pipe to memory 'elements_per_cycle' elements a time
// In this version, the user has specified a the amount of remainder
//
template<typename Pipe, int elements_per_cycle, bool no_remainder, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t full_count, size_t remainder_count) {
  if constexpr (no_remainder) {
    detail::PipeToMemoryNoRemainder<Pipe, elements_per_cycle>(out_ptr, full_count);
  } else {
    detail::PipeToMemoryRemainder<Pipe, elements_per_cycle>(out_ptr, full_count, remainder_count);
  }
}

} // namespace fpga_tools

#endif /* __MEMORY_UTILS_HPP__ */