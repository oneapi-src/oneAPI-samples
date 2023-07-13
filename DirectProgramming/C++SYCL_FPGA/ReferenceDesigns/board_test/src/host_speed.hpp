// Header file to accompany hostspeed tests
#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>

#include "helper.hpp"

// struct used in ReadSpeed & WriteSpeed functions to store transfer speeds
struct Speed {
  float fastest;
  float slowest;
  float average;
  float total;
};

///////////////////////////////////
// **** WriteSpeed function **** //
///////////////////////////////////

// Inputs:
// 1. queue &q - queue to submit operation
// 2. buffer<char,1> &device_buffer - device buffer to write to (created in
// calling function)
// 3. char *hostbuf_wr - host memory that contains input data to be written to
// device (allocated in calling function)
// 4. size_t block_bytes - size of 1 transfer to device (in bytes)
// 5. size_t total_bytes - total number of bytes to transfer
// Returns:
// struct Speed with the transfer times

// The function does the following tasks:
// 1. Write data to device in multiple transfers as total_bytes > block_bytes
// (i.e. size of 1 transfer)
// 2. Calculate bandwidth based on measured time for each transfer

struct Speed WriteSpeed(sycl::queue &q, sycl::buffer<char, 1> &device_buffer,
                        char *hostbuf_wr, size_t block_bytes,
                        size_t total_bytes) {
  // Total number of iterations to transfer all bytes in block_bytes sizes
  size_t num_xfers = total_bytes / block_bytes;

  assert(num_xfers > 0);

  // Sycl event for each transfer
  sycl::event evt[num_xfers];

  // **** Write to device **** //

  for (size_t i = 0; i < num_xfers; i++) {
    // Submit copy operation (explicit copy from host to device)
    evt[i] = q.submit([&](sycl::handler &h) {
      // Range of buffer that needs to accessed
      auto buf_range = block_bytes / sizeof(char);
      // offset starts at 0 - incremented by transfer size each iteration (i.e.
      // block_bytes)
      auto buf_offset = (i * block_bytes) / sizeof(char);
      // Accessor to access range of device buffer at buf_offset
      sycl::accessor<char, 1, sycl::access::mode::write> mem(
          device_buffer, h, buf_range, buf_offset);
      h.copy(&hostbuf_wr[buf_offset], mem);
    });
  }
  // Wait for copy to complete
  q.wait();

  // **** Get the time for each transfer from Sycl event array **** //

  struct Speed speed_wr;
  speed_wr.average = 0.0f;
  speed_wr.fastest = 0.0f;
  speed_wr.slowest = 1.0e7f;

  for (size_t i = 0; i < num_xfers; i++) {
    float time_ns = SyclGetQStExecTimeNs(evt[i]);
    float speed_MBps = ((float)block_bytes / kMB) / ((float)time_ns * 1e-9);

    if (speed_MBps > speed_wr.fastest) speed_wr.fastest = speed_MBps;
    if (speed_MBps < speed_wr.slowest) speed_wr.slowest = speed_MBps;

    speed_wr.average += time_ns;
  }

  // Average write bandwidth
  speed_wr.average =
      ((float)total_bytes / kMB) / ((float)speed_wr.average * 1e-9);
  speed_wr.total =
      ((float)total_bytes / kMB) /
      ((float)SyclGetTotalTimeNs(evt[0], evt[num_xfers - 1]) * 1e-9);

  return speed_wr;

}  // End of WriteSpeed

//////////////////////////////////
// **** ReadSpeed function **** //
//////////////////////////////////

// Inputs:
// 1. queue &q - queue to submit operation
// 2. buffer<char,1> &device_buffer - device buffer to read from (created in
// calling function)
// 3. char *hostbuf_rd - pointer to host memory to store data that is read from
// device (allocated in calling function)
// 4. size_t block_bytes - size of 1 transfer to device (in bytes)
// 5. size_t total_bytes - total number of bytes to transfer
// Returns:
// struct Speed with the transfer times

// The function does the following tasks:
// 1. Read data from device in multiple transfers as total_bytes > block_bytes
// (i.e. size of 1 transfer)
// 2. Calculate bandwidth based on measured time for each transfer

struct Speed ReadSpeed(sycl::queue &q, sycl::buffer<char, 1> &device_buffer,
                       char *hostbuf_rd, size_t block_bytes,
                       size_t total_bytes) {
  // Total number of iterations to transfer all bytes in block_bytes sizes
  size_t num_xfers = total_bytes / block_bytes;

  assert(num_xfers > 0);

  // Sycl event for each transfer
  sycl::event evt[num_xfers];

  // **** Read from device **** //

  for (size_t i = 0; i < num_xfers; i++) {
    // Submit copy operation (explicit copy from device to host)
    evt[i] = q.submit([&](sycl::handler &h) {
      // Range of buffer that needs to accessed
      auto buf_range = block_bytes / sizeof(char);
      // offset starts at 0 - incremented by transfer size each iteration (i.e
      // block_bytes)
      auto buf_offset = (i * block_bytes) / sizeof(char);
      // Accessor to access range of device buffer at buf_offset
      sycl::accessor<char, 1, sycl::access::mode::read> mem(
          device_buffer, h, buf_range, buf_offset);
      h.copy(mem, &hostbuf_rd[buf_offset]);
    });
  }
  // Wait for copy to complete
  q.wait();

  // **** Get the time for each transfer from Sycl event array **** //

  struct Speed speed_rd;
  speed_rd.average = 0.0f;
  speed_rd.fastest = 0.0f;
  speed_rd.slowest = 1.0e7f;

  for (size_t i = 0; i < num_xfers; i++) {
    float time_ns = SyclGetQStExecTimeNs(evt[i]);
    float speed_MBps = ((float)block_bytes / kMB) / ((float)time_ns * 1e-9);

    if (speed_MBps > speed_rd.fastest) speed_rd.fastest = speed_MBps;
    if (speed_MBps < speed_rd.slowest) speed_rd.slowest = speed_MBps;

    speed_rd.average += time_ns;
  }

  // Average read bandwidth
  speed_rd.average =
      ((float)total_bytes / kMB) / ((float)speed_rd.average * 1e-9);
  speed_rd.total =
      ((float)total_bytes / kMB) /
      ((float)SyclGetTotalTimeNs(evt[0], evt[num_xfers - 1]) * 1e-9);

  return speed_rd;

}  // End of ReadSpeed

/////////////////////////////////////
// **** CheckResults function **** //
/////////////////////////////////////

// Inputs:
// 1. char *hostbuf_rd - pointer to host memory with data read back from device
// (allocated in calling function)
// 2. char *hostbuf_wr - pointer to host memory with input data written to
// device (allocated in calling function)
// 3. maxchars - size of comparison
// Returns:
// true if verification is successfull

// The function does the following tasks:
// 1. Compare maxchars elements of hostbuf_rd to hostbuf_wr
// 2. Return false if a mismatch is found
// 3. Return true if all values read back from device match the values that were
// written

bool CheckResults(char *hostbuf_rd, char *hostbuf_wr, size_t maxchars) {
  bool result = true;
  int prints = 0;

  for (auto j = 0; j < maxchars; j++) {
    if (hostbuf_rd[j] != hostbuf_wr[j]) {
      if (prints++ < 512) {  // only print 512 errors
        std::cerr << "Error! Mismatch at element " << j << ":" << std::setw(8)
                  << std::hex << std::showbase << hostbuf_rd[j]
                  << " != " << hostbuf_wr[j] << ", xor = " << std::setfill('0')
                  << (hostbuf_rd[j] ^ hostbuf_wr[j]) << "\n";
      }
      result = false;
    }
  }

  return result;

}  // End of CheckResults
