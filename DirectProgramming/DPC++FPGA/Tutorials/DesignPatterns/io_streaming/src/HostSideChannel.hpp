#ifndef __HOSTSIDECHANNEL_HPP__
#define __HOSTSIDECHANNEL_HPP__

#include <iostream>
#include <type_traits>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "FakeIOPipes.hpp"

using namespace sycl;

//
// This class provides a convenient, but low-bandwidth and relatively high
// latency, side channel to send data from the host to the device. It exposes
// a read() interface to the DEVICE code that the user can treat just like a
// SYCL pipe. It also exposes a write interface to the HOST that allows the
// user to easily write data from host to the device
//
template <typename Id, typename T, bool use_host_alloc, size_t min_capacity=0>
class HostToDeviceSideChannel {
protected:
  using MyProducer = Producer<Id, T, use_host_alloc, min_capacity>;
  static inline queue* q_{nullptr};

public:
  // disable copy constructor and operator=
  HostToDeviceSideChannel()=delete;
  HostToDeviceSideChannel(const HostToDeviceSideChannel &)=delete;
  HostToDeviceSideChannel& operator=(HostToDeviceSideChannel const &)=delete;

  static void Init(queue &q) {
    q_ = &q;
    MyProducer::Init(q, 1);
  };

  static void Destroy(queue &q) {
    q_ = nullptr;
    MyProducer::Destroy(q);
  };

  static T read() {
    // DEVICE CODE
    return MyProducer::Pipe::read();
  }

  static T read(bool &success_code) {
    // DEVICE CODE
    return MyProducer::Pipe::read(success_code);
  }

  // blocking
  static void write(const T &data) {
    // HOST CODE
    // populate the data
    MyProducer::Data()[0] = data;

    // start the kernel and wait on it to finish (blocking)
    event dma, kernel;
    std::tie(dma, kernel) = MyProducer::Start(*q_);
    dma.wait();
    kernel.wait();
  }

  // non-blocking
  // Call .wait() on the returned event to wait for the write to take place
  static event write(const T &data, bool &success_code) {
    // HOST CODE
    // always succeed
    success_code = true;

    // populate the data
    MyProducer::Data()[0] = data;

    // start the kernel and return the kernel event
    return MyProducer::Start(*q_).second;
  }
};

//
// This class provides a convenient, but not highly performing, side channel
// to send data from the device to the host. It exposes a read() interface
// to the HOST code that lets the user get updates from the device.
// It also exposes a write interface to the DEVICE that allows the user to
// easily write data from device to the host.
//
template <typename Id, typename T, bool use_host_alloc, size_t min_capacity=0>
class DeviceToHostSideChannel {
protected:
  using MyConsumer = Consumer<Id, T, use_host_alloc, min_capacity>;
  static inline queue* q_{nullptr};

public:
  // disable copy constructor and operator=
  DeviceToHostSideChannel()=delete;
  DeviceToHostSideChannel(const DeviceToHostSideChannel &)=delete;
  DeviceToHostSideChannel& operator=(DeviceToHostSideChannel const &)=delete;

  static void Init(queue &q) {
    q_ = &q;
    MyConsumer::Init(q, 1);
  };

  static void Destroy(queue &q) {
    q_ = nullptr;
    MyConsumer::Destroy(q);
  };

  // blocking
  static T read() {
    // HOST CODE
    // launch the kernel to read the data from the pipe into memory
    // and wait for it to finish (blocking)
    event dma, kernel;
    std::tie(dma, kernel) = MyConsumer::Start(*q_);
    dma.wait();
    kernel.wait();

    // the kernel has finished, so return the data
    return MyConsumer::Data()[0];
  }

  // non-blocking
  // call .wait() on the returned event to wait for the read to take place,
  // then access the data using ::Data()[0]
  static event read(bool &success_code) {
    // start the kernel and return the event
    // the user can use ::Data() later to get the data
    // return the DMA event, since it happen second
    success_code = true;
    return MyConsumer::Start(*q_).first;
  }

  static void write(const T &data) {
    // DEVICE CODE
    MyConsumer::Pipe::write(data);
  }

  static void write(const T &data, bool &success_code) {
    // DEVICE CODE
    MyConsumer::Pipe::write(data, success_code);
  }

  static T Data() {
    return MyConsumer::Data()[0];
  }
};

#endif /* __HOSTSIDECHANNEL_HPP__ */
