#ifndef __ROM_BASE_HPP__
#define __ROM_BASE_HPP__

#include <type_traits>

template<typename T, int _depth>
struct ROMBase {
  // ensure a positive depth
  static_assert(_depth > 0);
  
  // allows the depth of the ROM to be queried
  static constexpr int depth = _depth;

  // allows the type stored in the ROM to be queried
  using val_type = T;

  // constexpr constructor that initializes the contents of the ROM
  // using a user specified Functor. NOTE: the functor must be constexpr,
  // which can be achieved with a lamda or by marking the operator() function
  // as constexpr.
  template<typename InitFunctor>
  constexpr ROMBase(const InitFunctor& func) : data_() {
    static_assert(std::is_invocable_r_v<T, InitFunctor, int>);

    for (int i = 0; i < depth; i++) {
      data_[i] = func(i);
    }
  }

  // only define a const operator[], since this is a ROM
  const T& operator[](int i) const { return data_[i]; }

 protected:
  T data_[depth];
};

#endif /* __ROM_BASE_HPP__ */