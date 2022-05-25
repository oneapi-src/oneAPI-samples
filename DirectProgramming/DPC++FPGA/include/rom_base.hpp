#ifndef __ROM_BASE_HPP__
#define __ROM_BASE_HPP__

#include <type_traits>

//
// A base class for creating a constexpr ROM.
//
// TEMPLATE PARAMETERS
//    T:      the datatype stored in the ROM
//    _depth: the depth of the ROM
//
// EXAMPLE USAGE
//  To use the ROM, you must create a class that inherits from this class and
//  provides a constexpr functor to the constructor which determines how the
//  ROM is initialized. The following examples show two methods for creating
//  a ROM that stores x^2, where 'x' is the index into the ROM.
//
//  USING A FUNCTOR
//    struct SquareFunctor {
//      constexpr float operator () (int x) const { return x * x }
//      constexpr SquareFunctor() = default;
//    };
//
//    constexpr int lut_depth = 1024;
//    struct SquareLUT : ROMBase<int, lut_depth> {
//      constexpr SquareLUT() : ROMBase<int, lut_depth>(SquareFunctor()) {}
//    };
//
//  USING A LAMBDA
//    constexpr int lut_depth = 1024;
//    struct SquareLUT : ROMBase<int, lut_depth> {
//      constexpr SquareLUT() : ROMBase<int, lut_depth>(
//        [](int x) { return x * x; }) {}
//    };
//
namespace fpga_tools {

template<typename T, int rom_depth>
struct ROMBase {
  // ensure a positive depth
  static_assert(rom_depth > 0);
  
  // allows the depth of the ROM to be queried
  static constexpr int depth = rom_depth;

  // allows the type stored in the ROM to be queried
  using ValType = T;

  // constexpr constructor that initializes the contents of the ROM
  // using a user specified Functor. NOTE: the functor must be constexpr,
  // which can be achieved with a lambda or by marking the operator() function
  // as constexpr.
  template<typename InitFunctor>
  constexpr ROMBase(const InitFunctor& func) : data_() {
    static_assert(std::is_invocable_r_v<T, InitFunctor, int>);

    for (int i = 0; i < rom_depth; i++) {
      data_[i] = func(i);
    }
  }

  // only define a const operator[], since this is a ROM
  const T& operator[](int i) const { return data_[i]; }

 protected:
  T data_[rom_depth];
};

}  // namespace fpga_tools

#endif /* __ROM_BASE_HPP__ */
