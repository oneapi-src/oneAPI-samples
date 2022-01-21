#ifndef __UTILS_HPP__
#define __UTILS_HPP__

// A structure that holds a table a of count elements of type T.
template <unsigned count, typename T>
struct PipeTable {
  T elem[count];

  template <int idx>
  T get() {
    return elem[idx];
  }

  template <int idx>
  void set(T &in) {
    elem[idx] = in;
  }
};

#endif /* __UTILS_HPP__ */