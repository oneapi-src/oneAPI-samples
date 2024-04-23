//==---- image.hpp --------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_IMAGE_HPP__
#define __DPCT_IMAGE_HPP__

#include <sycl/sycl.hpp>

#include "memory.hpp"

namespace dpct {

enum class image_channel_data_type {
  signed_int,
  unsigned_int,
  fp,
};

/// Image channel info, include channel number, order, data width and type
class image_channel {
  image_channel_data_type _type = image_channel_data_type::signed_int;
  /// Number of channels.
  unsigned _channel_num = 0;
  /// Total size of all channels in bytes.
  unsigned _total_size = 0;
  /// Size of each channel in bytes.
  unsigned _channel_size = 0;

public:

  image_channel() = default;

  unsigned get_total_size() { return _total_size; }

  void set_channel_num(unsigned channel_num) {
    _channel_num = channel_num;
    _total_size = _channel_size * _channel_num;
  }

  /// image_channel constructor.
  /// \param r Channel r width in bits.
  /// \param g Channel g width in bits. Should be same with \p r, or zero.
  /// \param b Channel b width in bits. Should be same with \p g, or zero.
  /// \param a Channel a width in bits. Should be same with \p b, or zero.
  /// \param data_type Image channel data type: signed_nt, unsigned_int or fp.
  image_channel(int r, int g, int b, int a, image_channel_data_type data_type) {
    _type = data_type;
    if (a) {
      assert(r == a && "SYCL doesn't support different channel size");
      assert(r == b && "SYCL doesn't support different channel size");
      assert(r == g && "SYCL doesn't support different channel size");
      set_channel_size(4, a);
    } else if (b) {
      assert(r == b && "SYCL doesn't support different channel size");
      assert(r == g && "SYCL doesn't support different channel size");
      set_channel_size(3, b);
    } else if (g) {
      assert(r == g && "SYCL doesn't support different channel size");
      set_channel_size(2, g);
    } else {
      set_channel_size(1, r);
    }
  }

  void set_channel_type(sycl::image_channel_type type) {
    switch (type) {
    case sycl::image_channel_type::unsigned_int8:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 1;
      break;
    case sycl::image_channel_type::unsigned_int16:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 2;
      break;
    case sycl::image_channel_type::unsigned_int32:
      _type = image_channel_data_type::unsigned_int;
      _channel_size = 4;
      break;
    case sycl::image_channel_type::signed_int8:
      _type = image_channel_data_type::signed_int;
      _channel_size = 1;
      break;
    case sycl::image_channel_type::signed_int16:
      _type = image_channel_data_type::signed_int;
      _channel_size = 2;
      break;
    case sycl::image_channel_type::signed_int32:
      _type = image_channel_data_type::signed_int;
      _channel_size = 4;
      break;
    case sycl::image_channel_type::fp16:
      _type = image_channel_data_type::fp;
      _channel_size = 2;
      break;
    case sycl::image_channel_type::fp32:
      _type = image_channel_data_type::fp;
      _channel_size = 4;
      break;
    default:
      break;
    }
    _total_size = _channel_size * _channel_num;
  }

  /// Set channel size.
  /// \param in_channel_num Channels number to set.
  /// \param channel_size Size for each channel in bits.
  void set_channel_size(unsigned in_channel_num,
                        unsigned channel_size) {
    if (in_channel_num < _channel_num)
      return;
    _channel_num = in_channel_num;
    _channel_size = channel_size / 8;
    _total_size = _channel_size * _channel_num;
  }

};

/// 2D or 3D matrix data for image.
class image_matrix {
  image_channel _channel;
  int _range[3] = {1, 1, 1};
  int _dims = 0;
  void *_host_data = nullptr;

  /// Set range of each dimension.
  template <int dimensions> void set_range(sycl::range<dimensions> range) {
    for (int i = 0; i < dimensions; ++i)
      _range[i] = range[i];
    _dims = dimensions;
  }

public:
  /// Constructor with channel info and dimension size info.
  template <int dimensions>
  image_matrix(image_channel channel, sycl::range<dimensions> range)
      : _channel(channel) {
    set_range(range);
    _host_data = std::malloc(range.size() * _channel.get_total_size());
  }
  image_matrix(sycl::image_channel_type channel_type, unsigned channel_num,
               size_t x, size_t y) {
    _channel.set_channel_type(channel_type);
    _channel.set_channel_num(channel_num);
    _dims = 1;
    _range[0] = x;
    if (y) {
      _dims = 2;
      _range[1] = y;
    }
    _host_data = std::malloc(_range[0] * _range[1] * _channel.get_total_size());
  }

  /// Convert to pitched data.
  pitched_data to_pitched_data() {
    return pitched_data(_host_data, _range[0], _range[0], _range[1]);
  }

  ~image_matrix() {
    if (_host_data)
      std::free(_host_data);
    _host_data = nullptr;
  }
};

} // namespace dpct

#endif // __DPCT_IMAGE_HPP__
