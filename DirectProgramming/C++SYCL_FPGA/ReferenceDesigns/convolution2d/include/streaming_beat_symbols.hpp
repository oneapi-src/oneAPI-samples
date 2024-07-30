#pragma once

#include <array>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

namespace intel_vvp {

/// @brief This type extends the sycl::ext::intel::experimental::StreamingBeat
/// type and adds support for multiple symbols.
/// @tparam T The type of each symbol
/// @tparam N The number of symbols transferred in each beat
template <typename T, int N>
struct StreamingBeatSymbols
    : sycl::ext::intel::experimental::StreamingBeat<std::array<T, N>, true,
                                                    true> {
  /// @brief The type used to contain the symbols
  using SymbolCollectionType = std::array<T, N>;

  /// @brief The number of symbols in this beat
  constexpr int kSymbols = N;

  /// @brief The symbol  type
  using SymbolType = T;

  /// @brief Access a specific symbol in this beat
  /// @param idx
  /// @return
  T& operator[](int idx) { return data[idx]; }
  /// @brief Access a specific symbol in this beat
  /// @param idx
  /// @return
  T operator[](int idx) const { return data[idx]; }
};
}  // namespace intel_vvp