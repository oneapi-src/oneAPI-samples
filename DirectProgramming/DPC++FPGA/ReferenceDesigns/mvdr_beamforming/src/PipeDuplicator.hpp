//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#ifndef __PIPE_DUPLICATOR_HPP__
#define __PIPE_DUPLICATOR_HPP__

// PipeDuplicator
// Connect a kernel that writes to a single pipe to multiple pipe instances,
// each of which will receive the same data.
// A blocking write will perform a blocking write to each pipe.  A non-blocking
// write will perform a non-blocking write to each pipe, and set success to
// true only if ALL writes were successful.

// primary template, dummy
template <class Id,          // name of this PipeDuplicator
          typename T,        // data type to transfer
          typename... Pipes  // all pipes to send duplicated writes to
          >
struct PipeDuplicator {};

// recursive case, write to each pipe
template <class Id,                   // name of this PipeDuplicator
          typename T,                 // data type to transfer
          typename FirstPipe,         // at least one output pipe
          typename... RemainingPipes  // additional copies of the output pipe
          >
struct PipeDuplicator<Id, T, FirstPipe, RemainingPipes...> {
  PipeDuplicator() = delete;  // ensure we cannot create an instance

  // Non-blocking write
  static void write(const T &data, bool &success) {
    bool local_success;
    FirstPipe::write(data, local_success);
    success = local_success;
    PipeDuplicator<Id, T, RemainingPipes...>::write(data, local_success);
    success &= local_success;
  }

  // Blocking write
  static void write(const T &data) {
    FirstPipe::write(data);
    PipeDuplicator<Id, T, RemainingPipes...>::write(data);
  }
};

// base case for recursion, no pipes to write to
// also useful as a 'null' pipe, writes don't do anything
template <class Id,   // name of this PipeDuplicator
          typename T  // data type to transfer
          >
struct PipeDuplicator<Id, T> {
  PipeDuplicator() = delete;  // ensure we cannot create an instance

  // Non-blocking write
  static void write(const T & /*data*/, bool &success) { success = true; }

  // Blocking write
  static void write(const T & /*data*/) {
    // do nothing
  }
};

#endif  // ifndef __PIPE_DUPLICATOR_HPP__
