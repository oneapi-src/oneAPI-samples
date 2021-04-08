#ifndef __NULL_PIPE_HPP__
#define __NULL_PIPE_HPP__

// NullPipe
// Intended to connect to kernels that have unneeded pipes.
// Blocking writes accept and discard data and return immediately.
// Non-blocking writes accept and discard data and return immediately with
// success = true.
// Blocking reads never return.
// Non-blocking reads return immediately with success = false.
template <class Id,   // name of the pipe
          typename T  // data type accepted by the pipe
          >
struct NullPipe {
  NullPipe() = delete;  // ensure we cannot create an instance

  // Non-blocking write
  static void write(const T & /*data*/, bool &success) { success = true; }

  // Blocking write
  static void write(const T & /*data*/) {
    // do nothing
  }

  // Non-blocking read
  static T read(bool &success) {
    T return_value = T();  // call the default constructor to avoid warnings
                           // about uninitialized variables
    success = false;
    return (return_value);
  }

  // Blocking read
  static T read() {
    T return_value = T();  // call the default constructor to avoid warnings
                           // about uninitialized variables
    while (1) {
      // do nothing, never return
    }
    return (return_value);
  }

};  // end of struct NullPipe

#endif  // ifndef __NULL_PIPE_HPP__
