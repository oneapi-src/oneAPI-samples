//===--- bindless_images.hpp ----------------------*- C++ -*---------------===//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BINDLESS_IMAGE_HPP__
#define __DPCT_BINDLESS_IMAGE_HPP__

namespace dpct {
namespace experimental {

#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES

namespace detail {
struct sampled_image_handle_compare {
  bool
  operator()(sycl::ext::oneapi::experimental::sampled_image_handle L,
             sycl::ext::oneapi::experimental::sampled_image_handle R) const {
    return L.raw_handle < R.raw_handle;
  }
};

inline std::pair<image_data, sampling_info> &get_img_info_map(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::sampled_image_handle,
                  std::pair<image_data, sampling_info>,
                  sampled_image_handle_compare>
      img_info_map;
  return img_info_map[handle];
}

inline sycl::ext::oneapi::experimental::image_mem *&get_img_mem_map(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::sampled_image_handle,
                  sycl::ext::oneapi::experimental::image_mem *,
                  sampled_image_handle_compare>
      img_mem_map;
  return img_mem_map[handle];
}

static inline size_t
get_ele_size(const sycl::ext::oneapi::experimental::image_descriptor &decs) {
  size_t channel_num, channel_size;
  switch (decs.channel_order) {
  case sycl::image_channel_order::r:
    channel_num = 1;
    break;
  case sycl::image_channel_order::rg:
    channel_num = 2;
    break;
  case sycl::image_channel_order::rgb:
    channel_num = 3;
    break;
  case sycl::image_channel_order::rgba:
    channel_num = 4;
    break;
  default:
    throw std::runtime_error("Unsupported channel_order in get_ele_size!");
    break;
  }
  switch (decs.channel_type) {
  case sycl::image_channel_type::signed_int8:
  case sycl::image_channel_type::unsigned_int8:
    channel_size = 1;
    break;
  case sycl::image_channel_type::fp16:
  case sycl::image_channel_type::signed_int16:
  case sycl::image_channel_type::unsigned_int16:
    channel_size = 2;
    break;
  case sycl::image_channel_type::fp32:
  case sycl::image_channel_type::signed_int32:
  case sycl::image_channel_type::unsigned_int32:
    channel_size = 4;
    break;
  default:
    throw std::runtime_error("Unsupported channel_type in get_ele_size!");
    break;
  }
  return channel_num * channel_size;
}

static inline sycl::event
dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
            size_t w_offset_src, size_t h_offset_src, void *dest, size_t p,
            size_t w, size_t h, sycl::queue q) {
  const auto ele_size = get_ele_size(src->get_descriptor());
  const auto src_offset =
      sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
  const auto dest_offset = sycl::range<3>(0, 0, 0);
  const auto dest_extend = sycl::range<3>(p / ele_size, 0, 0);
  const auto copy_extend = sycl::range<3>(w / ele_size, h, 0);
  return q.ext_oneapi_copy(src->get_handle(), src_offset, src->get_descriptor(),
                           dest, dest_offset, dest_extend, copy_extend);
}

static inline std::vector<sycl::event>
dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
            size_t w_offset_src, size_t h_offset_src, void *dest, size_t s,
            sycl::queue q) {
  std::vector<sycl::event> event_list;
  const auto ele_size = get_ele_size(src->get_descriptor());
  const auto w = src->get_descriptor().width * ele_size;
  size_t offset_dest = 0;
  while (s - offset_dest > w - w_offset_src) {
    const auto src_offset =
        sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
    const auto dest_offset = sycl::range<3>(offset_dest / ele_size, 0, 0);
    const auto dest_extend = sycl::range<3>(0, 0, 0);
    const auto copy_extend =
        sycl::range<3>((w - w_offset_src) / ele_size, 1, 0);
    event_list.push_back(
        q.ext_oneapi_copy(src->get_handle(), src_offset, src->get_descriptor(),
                          dest, dest_offset, dest_extend, copy_extend));
    offset_dest += w - w_offset_src;
    w_offset_src = 0;
    ++h_offset_src;
  }
  const auto src_offset =
      sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
  const auto dest_offset = sycl::range<3>(offset_dest / ele_size, 0, 0);
  const auto dest_extend = sycl::range<3>(0, 0, 0);
  const auto copy_extend = sycl::range<3>((s - w_offset_src) / ele_size, 1, 0);
  event_list.push_back(
      q.ext_oneapi_copy(src->get_handle(), src_offset, src->get_descriptor(),
                        dest, dest_offset, dest_extend, copy_extend));
  return event_list;
}

static inline sycl::event
dpct_memcpy(void *src, sycl::ext::oneapi::experimental::image_mem *dest,
            size_t w_offset_dest, size_t h_offset_dest, size_t p, size_t w,
            size_t h, sycl::queue q) {
  const auto ele_size = get_ele_size(dest->get_descriptor());
  const auto src_offset = sycl::range<3>(0, 0, 0);
  const auto src_extend = sycl::range<3>(p / ele_size, 0, 0);
  const auto dest_offset =
      sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
  const auto copy_extend = sycl::range<3>(w / ele_size, h, 0);
  return q.ext_oneapi_copy(src, src_offset, src_extend, dest->get_handle(),
                           dest_offset, dest->get_descriptor(), copy_extend);
}

static inline std::vector<sycl::event>
dpct_memcpy(void *src, sycl::ext::oneapi::experimental::image_mem *dest,
            size_t w_offset_dest, size_t h_offset_dest, size_t s,
            sycl::queue q = get_default_queue()) {
  std::vector<sycl::event> event_list;
  const auto ele_size = get_ele_size(dest->get_descriptor());
  const auto w = dest->get_descriptor().width * ele_size;
  size_t offset_src = 0;
  while (s - offset_src >= w - w_offset_dest) {
    const auto src_offset = sycl::range<3>(offset_src / ele_size, 0, 0);
    const auto src_extend = sycl::range<3>(0, 0, 0);
    const auto dest_offset =
        sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
    const auto copy_extend =
        sycl::range<3>((w - w_offset_dest) / ele_size, 1, 0);
    event_list.push_back(
        q.ext_oneapi_copy(src, src_offset, src_extend, dest->get_handle(),
                          dest_offset, dest->get_descriptor(), copy_extend));
    offset_src += w - w_offset_dest;
    w_offset_dest = 0;
    ++h_offset_dest;
  }
  const auto src_offset = sycl::range<3>(offset_src / ele_size, 0, 0);
  const auto src_extend = sycl::range<3>(0, 0, 0);
  const auto dest_offset =
      sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
  const auto copy_extend =
      sycl::range<3>((s - offset_src - w_offset_dest) / ele_size, 1, 0);
  event_list.push_back(q.ext_oneapi_copy(src, src_offset, src_extend,
                                         dest->get_handle(), dest_offset,
                                         dest->get_descriptor(), copy_extend));
  return event_list;
}
} // namespace detail

/// Create bindless image according to image data and sampling info.
/// \param [in] data The image data used to create bindless image.
/// \param [in] info The image sampling info used to create bindless image.
/// \param [in] q The queue where the image creation be executed.
/// \returns The sampled image handle of created bindless image.
static inline sycl::ext::oneapi::experimental::sampled_image_handle
create_bindless_image(image_data data, sampling_info info,
                      sycl::queue q = get_default_queue()) {
  auto desc = sycl::ext::oneapi::experimental::image_descriptor();
  auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
      info.get_addressing_mode(), info.get_coordinate_normalization_mode(),
      info.get_filtering_mode());

  switch (data.get_data_type()) {
  case image_data_type::linear: {
    desc.width = data.get_x() / data.get_channel().get_total_size();
    desc.channel_order = data.get_channel().get_channel_order();
    desc.channel_type = data.get_channel_type();
    desc.num_levels = 1;
    // linear memory only use sycl::filtering_mode::nearest.
    samp.filtering = sycl::filtering_mode::nearest;
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new sycl::ext::oneapi::experimental::image_mem(desc, q);
    auto img =
        sycl::ext::oneapi::experimental::create_image(*mem, samp, desc, q);
    detail::get_img_mem_map(img) = mem;
    auto ptr = data.get_data_ptr();
#ifdef DPCT_USM_LEVEL_NONE
    ptr = get_buffer(ptr)
              .template get_access<sycl::access_mode::read_write>()
              .get_pointer();
#endif
    q.ext_oneapi_copy(ptr, mem->get_handle(), desc);
    q.wait_and_throw();
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::pitch: {
    desc.width = data.get_x();
    desc.height = data.get_y();
    desc.channel_order = data.get_channel().get_channel_order();
    desc.channel_type = data.get_channel_type();
    desc.num_levels = 1;
#ifdef DPCT_USM_LEVEL_NONE
    auto mem = new sycl::ext::oneapi::experimental::image_mem(desc, q);
    auto img =
        sycl::ext::oneapi::experimental::create_image(*mem, samp, desc, q);
    detail::get_img_mem_map(img) = mem;
    auto ptr = get_buffer(data.get_data_ptr())
                   .template get_access<sycl::access_mode::read_write>()
                   .get_pointer();
    q.ext_oneapi_copy(ptr, mem->get_handle(), desc);
    q.wait_and_throw();
#else
    auto img = sycl::ext::oneapi::experimental::create_image(
        data.get_data_ptr(), data.get_pitch(), samp, desc, q);
    detail::get_img_mem_map(img) = nullptr;
#endif
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::matrix: {
    const auto mem = static_cast<sycl::ext::oneapi::experimental::image_mem *>(
        data.get_data_ptr());
    auto img = sycl::ext::oneapi::experimental::create_image(
        *mem, samp, mem->get_descriptor(), q);
    detail::get_img_mem_map(img) = nullptr;
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  default:
    throw std::runtime_error(
        "Unsupported image_data_type in create_bindless_image!");
    break;
  }
  // Must not reach here.
  return sycl::ext::oneapi::experimental::create_image(data.get_data_ptr(), 0,
                                                       samp, desc, q);
}

/// Destroy bindless image.
/// \param [in] handle The bindless image should be destroyed.
/// \param [in] q The queue where the image destruction be executed.
static inline void destroy_bindless_image(
    sycl::ext::oneapi::experimental::sampled_image_handle handle,
    sycl::queue q = get_default_queue()) {
  auto &mem = detail::get_img_mem_map(handle);
  if (mem) {
    sycl::ext::oneapi::experimental::free_image_mem(
        mem->get_handle(),
	dpct::get_current_device());
    mem = nullptr;
  }
  sycl::ext::oneapi::experimental::destroy_image_handle(handle, q);
}

/// Get the image data according to sampled image handle.
/// \param [in] handle The bindless image handle.
/// \returns The image data of sampled image.
static inline image_data
get_data(const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  return detail::get_img_info_map(handle).first;
}

/// Get the sampling info according to sampled image handle.
/// \param [in] handle The bindless image handle.
/// \returns The sampling info of sampled image.
static inline sampling_info get_sampling_info(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  return detail::get_img_info_map(handle).second;
}

/// Get the image channel according to image memory.
/// \param [in] mem The image memory pointer.
/// \returns The image channel of image memory.
static inline image_channel
get_channel(const sycl::ext::oneapi::experimental::image_mem *mem) {
  image_channel channel;
  channel.set_channel_num(mem->get_num_channels());
  channel.set_channel_type(mem->get_channel_type());
  return channel;
}

/// The wrapper class of bindless image.
template <class T, int dimensions> class bindless_image_wrapper {
public:
  /// Create bindless image wrapper according to template argument \p T.
  bindless_image_wrapper() : _channel(image_channel::create<T>()) {
    // Make sure that singleton class dev_mgr will destruct later than this.
    dev_mgr::instance();
  }

  /// Destroy bindless image wrapper.
  ~bindless_image_wrapper() {
    destroy_bindless_image(_img, get_default_queue());
  }

  /// Attach linear data to bindless image.
  /// \param [in] data The linear data used to create bindless image.
  /// \param [in] size The size of linear data used to create bindless image.
  /// \param [in] channel The image channel used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t size, const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {size}, channel.get_channel_order(), channel.get_channel_type());
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new sycl::ext::oneapi::experimental::image_mem(desc, q);
    _img = sycl::ext::oneapi::experimental::create_image(*mem, samp, desc, q);
    detail::get_img_mem_map(_img) = mem;
    auto ptr = data;
#ifdef DPCT_USM_LEVEL_NONE
    ptr = get_buffer(data)
              .template get_access<sycl::access_mode::read_write>()
              .get_pointer();
#endif
    q.ext_oneapi_copy(ptr, mem->get_handle(), desc);
    q.wait_and_throw();
  }

  /// Attach linear data to bindless image.
  /// \param [in] data The linear data used to create bindless image.
  /// \param [in] size The size of linear data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t size, sycl::queue q = get_default_queue()) {
    attach(data, size, _channel, q);
  }

  /// Attach 2D data to bindless image.
  /// \param [in] data The 2D data used to create bindless image.
  /// \param [in] width The width of 2D data used to create bindless image.
  /// \param [in] height The height of 2D data used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] channel The image channel used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t width, size_t height, size_t pitch,
              const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {width, height}, channel.get_channel_order(),
        channel.get_channel_type());
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
#ifdef DPCT_USM_LEVEL_NONE
    auto mem = new sycl::ext::oneapi::experimental::image_mem(desc, q);
    _img = sycl::ext::oneapi::experimental::create_image(*mem, samp, desc, q);
    detail::get_img_mem_map(_img) = mem;
    auto ptr = get_buffer(data)
                   .template get_access<sycl::access_mode::read_write>()
                   .get_pointer();
    q.ext_oneapi_copy(ptr, mem->get_handle(), desc);
    q.wait_and_throw();
#else
    _img = sycl::ext::oneapi::experimental::create_image(data, pitch, samp,
                                                         desc, q);
    detail::get_img_mem_map(_img) = nullptr;
#endif
  }

  /// Attach 2D data to bindless image.
  /// \param [in] data The 2D data used to create bindless image.
  /// \param [in] width The width of 2D data used to create bindless image.
  /// \param [in] height The height of 2D data used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t width, size_t height, size_t pitch,
              sycl::queue q = get_default_queue()) {
    attach(data, width, height, pitch, _channel, q);
  }

  /// Attach image memory to bindless image.
  /// \param [in] mem The image memory used to create bindless image.
  /// \param [in] channel The image channel used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(sycl::ext::oneapi::experimental::image_mem *mem,
              const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    _img = sycl::ext::oneapi::experimental::create_image(
        *mem, samp, mem->get_descriptor(), q);
    detail::get_img_mem_map(_img) = nullptr;
  }

  /// Attach image memory to bindless image.
  /// \param [in] mem The image memory used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(sycl::ext::oneapi::experimental::image_mem *mem,
              sycl::queue q = get_default_queue()) {
    attach(mem, _channel, q);
  }

  /// Detach bindless image data.
  /// \param [in] q The queue where the image destruction be executed.
  void detach(sycl::queue q = get_default_queue()) {
    destroy_bindless_image(_img, q);
  }

  /// Set image channel of bindless image.
  /// \param [in] channel The image channel used to set.
  void set_channel(image_channel channel) { _channel = channel; }
  /// Get image channel of bindless image.
  /// \returns The image channel of bindless image.
  image_channel get_channel() { return _channel; }

  /// Set channel size of image channel.
  /// \param [in] channel_num The channels number to set.
  /// \param [in] channel_size The size for each channel in bits.
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _channel.set_channel_size(channel_num, channel_size);
  }
  /// Get channel size of image channel.
  /// \returns The size for each channel in bits.
  unsigned get_channel_size() { return _channel.get_channel_size(); }

  /// Set image channel data type of image channel.
  /// \param [in] type The image channel data type to set.
  void set_channel_data_type(image_channel_data_type type) {
    _channel.set_channel_data_type(type);
  }
  /// Get image channel data type of image channel.
  /// \returns The image channel data type of image channel.
  image_channel_data_type get_channel_data_type() {
    return _channel.get_channel_data_type();
  }

  /// Set addressing mode of bindless image.
  /// \param [in] addressing_mode The addressing mode used to set.
  void set(sycl::addressing_mode addressing_mode) {
    _addressing_mode = addressing_mode;
  }
  /// Get addressing mode of bindless image.
  /// \returns The addressing mode of bindless image.
  sycl::addressing_mode get_addressing_mode() { return _addressing_mode; }

  /// Set coordinate normalization mode of bindless image.
  /// \param [in] coordinate_normalization_mode The coordinate normalization
  /// mode used to set.
  void set(sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _coordinate_normalization_mode = coordinate_normalization_mode;
  }
  /// Get coordinate normalization mode of bindless image.
  /// \returns The coordinate normalization mode of bindless image.
  bool is_coordinate_normalized() {
    return _coordinate_normalization_mode ==
           sycl::coordinate_normalization_mode::normalized;
  }

  /// Set filtering mode of bindless image.
  /// \param [in] filtering_mode The filtering mode used to set.
  void set(sycl::filtering_mode filtering_mode) {
    _filtering_mode = filtering_mode;
  }
  /// Get filtering mode of bindless image.
  /// \returns The filtering mode of bindless image.
  sycl::filtering_mode get_filtering_mode() { return _filtering_mode; }

  /// Get bindless image handle.
  /// \returns The sampled image handle of created bindless image.
  inline sycl::ext::oneapi::experimental::sampled_image_handle get_handle() {
    return _img;
  }

private:
  image_channel _channel;
  sycl::addressing_mode _addressing_mode = sycl::addressing_mode::clamp_to_edge;
  sycl::coordinate_normalization_mode _coordinate_normalization_mode =
      sycl::coordinate_normalization_mode::unnormalized;
  sycl::filtering_mode _filtering_mode = sycl::filtering_mode::nearest;
  sycl::ext::oneapi::experimental::sampled_image_handle _img{0};
};

/// Asynchronously copies from the image memory to memory specified by a
/// pointer, The return of the function does NOT guarantee the copy is
/// completed.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] dest The destination memory address.
/// \param [in] p The pitch of destination memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void
async_dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                  size_t w_offset_src, size_t h_offset_src, void *dest,
                  size_t p, size_t w, size_t h,
                  sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, w_offset_src, h_offset_src, dest, p, w, h, q);
}

/// Synchronously copies from the image memory to memory specified by a
/// pointer, The function will return after the copy is completed.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] dest The destination memory address.
/// \param [in] p The pitch of destination memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               void *dest, size_t p, size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, w_offset_src, h_offset_src, dest, p, w, h, q).wait();
}

/// Asynchronously copies from the image memory to memory specified by a
/// pointer, The return of the function does NOT guarantee the copy is
/// completed.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] dest The destination memory address.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void
async_dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                  size_t w_offset_src, size_t h_offset_src, void *dest,
                  size_t s, sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, w_offset_src, h_offset_src, dest, s, q);
}

/// Synchronously copies from the image memory to memory specified by a
/// pointer, The function will return after the copy is completed.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] dest The destination memory address.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               void *dest, size_t s,
                               sycl::queue q = get_default_queue()) {
  sycl::event::wait(
      detail::dpct_memcpy(src, w_offset_src, h_offset_src, dest, s, q));
}

/// Asynchronously copies from memory specified by a pointer to the image
/// memory, The return of the function does NOT guarantee the copy is completed.
/// \param [in] src The source memory address.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] p The pitch of source memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void
async_dpct_memcpy(void *src, sycl::ext::oneapi::experimental::image_mem *dest,
                  size_t w_offset_dest, size_t h_offset_dest, size_t p,
                  size_t w, size_t h, sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest, w_offset_dest, h_offset_dest, p, w, h, q);
}

/// Synchronously copies from memory specified by a pointer to the image
/// memory, The function will return after the copy is completed.
/// \param [in] src The source memory address.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] p The pitch of source memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(void *src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t p, size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest, w_offset_dest, h_offset_dest, p, w, h, q)
      .wait();
}

/// Asynchronously copies from memory specified by a pointer to the image
/// memory, The return of the function does NOT guarantee the copy is completed.
/// \param [in] src The source memory address.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void
async_dpct_memcpy(void *src, sycl::ext::oneapi::experimental::image_mem *dest,
                  size_t w_offset_dest, size_t h_offset_dest, size_t s,
                  sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest, w_offset_dest, h_offset_dest, s, q);
}

/// Synchronously copies from memory specified by a pointer to the image
/// memory, The function will return after the copy is completed.
/// \param [in] src The source memory address.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(void *src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t s, sycl::queue q = get_default_queue()) {
  sycl::event::wait(
      detail::dpct_memcpy(src, dest, w_offset_dest, h_offset_dest, s, q));
}

/// Synchronously copies from image memory to the image memory, The function
/// will return after the copy is completed.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  auto temp = (void *)sycl::malloc_device(w * h, q);
  // TODO: Need change logic when sycl support image_mem to image_mem copy.
  dpct_memcpy(src, w_offset_src, h_offset_src, temp, w, w, h, q);
  dpct_memcpy(temp, dest, w_offset_dest, h_offset_dest, w, w, h, q);
  sycl::free(temp, q);
}

/// Synchronously copies from image memory to the image memory, The function
/// will return after the copy is completed.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(sycl::ext::oneapi::experimental::image_mem *src,
                               size_t w_offset_src, size_t h_offset_src,
                               sycl::ext::oneapi::experimental::image_mem *dest,
                               size_t w_offset_dest, size_t h_offset_dest,
                               size_t s, sycl::queue q = get_default_queue()) {
  auto temp = (void *)sycl::malloc_device(s, q);
  // TODO: Need change logic when sycl support image_mem to image_mem copy.
  dpct_memcpy(src, w_offset_src, h_offset_src, temp, s, q);
  dpct_memcpy(temp, dest, w_offset_dest, h_offset_dest, s, q);
  sycl::free(temp, q);
}

using image_mem_ptr = sycl::ext::oneapi::experimental::image_mem *;

#endif

} // namespace experimental
} // namespace dpct

#endif // !__DPCT_BINDLESS_IMAGE_HPP__
