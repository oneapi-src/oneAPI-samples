// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(_MSC_VER) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <rkcommon/math/vec.h>
//#include <~/intel/oneapi/rkcommon/latest/include/rkcommon/math/vec.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef USE_GPU
#include <sycl/sycl.hpp>
#endif

/* On Windows, we need to explicitly enable VT processing so that
 * the ANSI color codes we use will be supported. */
#if defined(_MSC_VER)

#include <windows.h>

void enableAnsiColor()
{
  HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hStdout) {
    DWORD outputMode = 0;
    if (GetConsoleMode(hStdout, &outputMode)) {
      outputMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
      SetConsoleMode(hStdout, outputMode);
    }
  }
}
#else
void enableAnsiColor() {}
#endif

using Color = rkcommon::math::vec4f;

/*
 * We use this transfer function to map scalar values to
 * colors with opacity.
 */
inline Color transferFunction(float v)
{
  Color p = {0.f};
  if (std::isfinite(v)) {
    const float a = std::max(0.f, std::min(std::abs(v), 1.f));
    p = (v < 0) ? Color{233.f / 255.f, 97.f / 255.f, 21.f / 255.f, a}
                : Color{0.f, 163.f / 255.f, 246.f / 255.f, a};
  }
  return p;
}

/*
 * Porter-Duff over blending. Note that we are not using
 * premultiplied colors here for simplicity.
 */
inline Color over(const Color &c1, const Color &c2)
{
  Color c;
  c.w = c1.w + (1.f - c1.w) * c2.w;
  c.x = (c1.w * (1.f - c2.w) * c1.x + c2.w * c2.x) / c.w;
  c.y = (c1.w * (1.f - c2.w) * c1.y + c2.w * c2.y) / c.w;
  c.z = (c1.w * (1.f - c2.w) * c1.z + c2.w * c2.z) / c.w;
  return c;
}

/*
 * A basic framebuffer class that can draw itself to the terminal using
 * ANSI control sequences.
 */
using Pixel = Color;

template <typename Allocator = std::allocator<Pixel>>
class Framebuffer
{
 public:
  Framebuffer() = delete;
  Framebuffer(size_t w, size_t h);

#ifdef USE_GPU
  Framebuffer(size_t w, size_t h, sycl::queue &syclQueue);
#endif

  template <class Functor>
  void generate(Functor &&f);

  void drawToTerminal() const;

 private:
  std::string toAnsi(const Pixel &p) const;

 private:
  size_t w{0};
  size_t h{0};
  Allocator allocator;
  std::vector<Pixel, Allocator> buffer;

#ifdef USE_GPU
  sycl::queue syclQueue;
#endif
};

template <typename Allocator>
inline Framebuffer<Allocator>::Framebuffer(size_t w, size_t h)
    : w(w), h(h), buffer(w * h, Pixel{0.f}, allocator)
{
  assert(w > 0);
  assert(h > 0);
  enableAnsiColor();
}

#ifdef USE_GPU
template <typename Allocator>
inline Framebuffer<Allocator>::Framebuffer(size_t w,
                                           size_t h,
                                           sycl::queue &syclQueue)
    : w(w),
      h(h),
      syclQueue(syclQueue),
      allocator(syclQueue),
      buffer(w * h, Pixel{0.f}, allocator)
{
  assert(w > 0);
  assert(h > 0);
  enableAnsiColor();
}
#endif

#ifndef USE_GPU

template <typename Allocator>
template <class Functor>
inline void Framebuffer<Allocator>::generate(Functor &&f)
{
  for (size_t y = 0; y < h; ++y) {
    const float fy = y / static_cast<float>(h);
    for (size_t x = 0; x < w; ++x) {
      const float fx    = x / static_cast<float>(w);
      buffer[y * w + x] = f(fx, fy);
    }
  }
}

#else

template <typename Allocator>
template <class Functor>
inline void Framebuffer<Allocator>::generate(Functor &&f)
{
  // cannot use `this` in a kernel function, so need local copies here
  const size_t _h = h;
  const size_t _w = w;
  Pixel *_buffer  = buffer.data();
  syclQueue
      .submit([=](sycl::handler &cgh) {
        cgh.single_task([=](sycl::kernel_handler kh) {
          for (size_t y = 0; y < _h; ++y) {
            const float fy = y / static_cast<float>(_h);
            for (size_t x = 0; x < _w; ++x) {
              const float fx      = x / static_cast<float>(_w);
              _buffer[y * _w + x] = f(fx, fy);
            }
          }
        });
      })
      .wait();
}

#endif

template <typename Allocator>
inline void Framebuffer<Allocator>::drawToTerminal() const
{
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      // Draw from top to bottom.
      const size_t idx = (h - 1 - y) * w + x;
      std::cout << toAnsi(buffer[idx]).c_str();
    }
    std::cout << std::endl;
  }
}

inline uint8_t premultAlpha(uint8_t v, uint8_t a)
{
  return static_cast<uint8_t>(v * static_cast<float>(a) / 255.f) & 0xFF;
}

const int mapTo8bit(float v, float a)
{
  v *= a;
  v *= 255.f;
  v = std::max(0.f, std::min(v, 255.f));
  return static_cast<int>(v) & 255;
}

template <typename Allocator>
inline std::string Framebuffer<Allocator>::toAnsi(const Pixel &p) const
{
  std::ostringstream os;
  os << "\x1b[48;2"
     << ";" << mapTo8bit(p.x, p.w) << ";" << mapTo8bit(p.y, p.w) << ";"
     << mapTo8bit(p.z, p.w) << "m \x1b[m";
  return os.str();
}
