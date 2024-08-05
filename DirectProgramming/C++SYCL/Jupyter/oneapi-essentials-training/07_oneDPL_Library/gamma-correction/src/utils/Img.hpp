//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _GAMMA_UTILS_IMG_HPP
#define _GAMMA_UTILS_IMG_HPP

#include "ImgPixel.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

// Image class definition
template <typename Format>
class Img {
 private:
  Format _format;
  int32_t _width;
  int32_t _height;
  vector<ImgPixel> _pixels;

  using Iterator = vector<ImgPixel>::iterator;
  using ConstIterator = vector<ImgPixel>::const_iterator;

 public:
  /////////////////////
  // SPECIAL METHODS //
  /////////////////////

  Img(int32_t width, int32_t height);

  void reset(int32_t width, int32_t height);

  ///////////////
  // ITERATORS //
  ///////////////

  Iterator begin() noexcept;
  Iterator end() noexcept;
  ConstIterator begin() const noexcept;
  ConstIterator end() const noexcept;
  ConstIterator cbegin() const noexcept;
  ConstIterator cend() const noexcept;

  /////////////
  // GETTERS //
  /////////////

  int32_t width() const noexcept;
  int32_t height() const noexcept;

  ImgPixel const* data() const noexcept;
  ImgPixel* data() noexcept;

  ///////////////////
  // FUNCTIONALITY //
  ///////////////////

  void write(string const& filename) const;

  template <typename Functor>
  void fill(Functor f);
  void fill(ImgPixel pixel);
  void fill(ImgPixel pixel, int32_t row, int32_t col);
};

///////////////////////////////////////////////
// IMG CLASS IMPLEMENTATION: SPECIAL METHODS //
///////////////////////////////////////////////

template <typename Format>
Img<Format>::Img(int32_t width, int32_t height) : _format(width, height) {
  _pixels.resize(width * height);

  _width = width;
  _height = height;
}

template <typename Format>
void Img<Format>::reset(int32_t width, int32_t height) {
  _pixels.resize(width * height);

  _width = width;
  _height = height;

  _format.reset(width, height);
}

/////////////////////////////////////////
// IMG CLASS IMPLEMENTATION: ITERATORS //
/////////////////////////////////////////

template <typename Format>
typename Img<Format>::Iterator Img<Format>::begin() noexcept {
  return _pixels.begin();
}

template <typename Format>
typename Img<Format>::Iterator Img<Format>::end() noexcept {
  return _pixels.end();
}

template <typename Format>
typename Img<Format>::ConstIterator Img<Format>::begin() const noexcept {
  return _pixels.begin();
}

template <typename Format>
typename Img<Format>::ConstIterator Img<Format>::end() const noexcept {
  return _pixels.end();
}

template <typename Format>
typename Img<Format>::ConstIterator Img<Format>::cbegin() const noexcept {
  return _pixels.begin();
}

template <typename Format>
typename Img<Format>::ConstIterator Img<Format>::cend() const noexcept {
  return _pixels.end();
}

///////////////////////////////////////
// IMG CLASS IMPLEMENTATION: GETTERS //
///////////////////////////////////////

template <typename Format>
int32_t Img<Format>::width() const noexcept {
  return _width;
}

template <typename Format>
int32_t Img<Format>::height() const noexcept {
  return _height;
}

template <typename Format>
ImgPixel const* Img<Format>::data() const noexcept {
  return _pixels.data();
}

template <typename Format>
ImgPixel* Img<Format>::data() noexcept {
  return _pixels.data();
}

/////////////////////////////////////////////
// IMG CLASS IMPLEMENTATION: FUNCTIONALITY //
/////////////////////////////////////////////

template <typename Format>
void Img<Format>::write(string const& filename) const {
  if (_pixels.empty()) {
    cerr << "Img::write:: image is empty\n";
    return;
  }

  ofstream filestream(filename, ios::binary);

  _format.write(filestream, *this);
}

template <typename Format>
template <typename Functor>
void Img<Format>::fill(Functor f) {
  if (_pixels.empty()) {
    cerr << "Img::fill(Functor): image is empty\n";
    return;
  }

  for (auto& pixel : _pixels) f(pixel);
}

template <typename Format>
void Img<Format>::fill(ImgPixel pixel) {
  if (_pixels.empty()) {
    cerr << "Img::fill(ImgPixel): image is empty\n";
    return;
  }

  fill(_pixels.begin(), _pixels.end(), pixel);
}

template <typename Format>
void Img<Format>::fill(ImgPixel pixel, int row, int col) {
  if (_pixels.empty()) {
    cerr << "Img::fill(ImgPixel): image is empty\n";
    return;
  }

  if (row >= _height || row < 0 || col >= _width || col < 0) {
    cerr << "Img::fill(ImgPixel, int, int): out of range\n";
    return;
  }

  _pixels.at(row * _width + col) = pixel;
}

#endif  // _GAMMA_UTILS_IMG_HPP
