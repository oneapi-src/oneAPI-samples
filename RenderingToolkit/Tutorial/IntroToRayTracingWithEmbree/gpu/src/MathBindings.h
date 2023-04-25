#pragma once
#ifndef FILEMATHBINDINGSSEEN
#define FILEMATHBINDINGSSEEN

#include "sycl/sycl.hpp"

/* These MathBindings are a quick wrapper for abstraction of geometric algebra operations and vector types. This wrapper functions on both host and target SYCL device.
 * In previous introduction programs, rkcommon library was used in the CPU only environment. This manual definition of grometric algebra types allows for targeting CPU or GPU devices and can be a stand-in for your own library.
 *  A standard SYCL vector type sycl::float2 and sycl::float4 and x,y,[z] union aliasing is used to be adaptable and more optimizable for application kernels.
 *
 */

/* We don't use much from Vec4ff, just in accumulation buffers. So 4 dimensional vector overloads are not needed for samples.*/
using Vec4ff = sycl::float4;

namespace MathBindings {

struct Vec3fa {
  union {
    sycl::float4 data;

    struct {
      float x, y, z, pad;
    };
  };

  Vec3fa() {}
  Vec3fa(float in_x, float in_y, float in_z) {
    this->x = in_x;
    this->y = in_y;
    this->z = in_z;
  }

  Vec3fa(const float& in) { this->x = this->y = this->z = in; }

  Vec3fa(const sycl::float4& in) { this->data = in; }

  // Do not use w... We use Vec3fa for alignment.
  inline Vec3fa operator+(const Vec3fa& second) {
    Vec3fa out(this->data + second.data);
    return out;
  }

  inline Vec3fa operator-(const Vec3fa& second) {
    Vec3fa out(this->data - second.data);
    return out;
  }

  inline Vec3fa operator/(const Vec3fa& second) {
    return Vec3fa(this->x / second.x, this->y / second.y, this->z / second.z);
  }

  inline Vec3fa operator/(float f) {
    return Vec3fa(this->x / f, this->y / f, this->z / f);
  }

  friend inline Vec3fa operator-(const Vec3fa& in) {
    return Vec3fa(-in.x, -in.y, -in.z);
  }

  friend inline Vec3fa operator*(const Vec3fa& first, const Vec3fa& second) {
    return Vec3fa(first.x * second.x, first.y * second.y, first.z * second.z);
  }

  friend inline const Vec3fa operator-(const Vec3fa& first,
                                       const Vec3fa& second) {
    return Vec3fa(first.data - second.data);
  }
};

inline const Vec3fa operator+(const Vec3fa& first, const Vec3fa& second) {
  return Vec3fa(first.data + second.data);
}

inline Vec3fa operator*(const float& f, const Vec3fa& in) {
  return Vec3fa(in.x * f, in.y * f, in.z * f);
}

inline Vec3fa operator*(const Vec3fa& in, const float& f) {
  return Vec3fa(in.x * f, in.y * f, in.z * f);
}

inline float dot(const Vec3fa& first, const Vec3fa& second) {
  return first.data[0] * second.data[0] + first.data[1] * second.data[1] +
         first.data[2] * second.data[2];
}

inline float dot(const sycl::float3& first,
                 const sycl::float3& second) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

inline Vec3fa normalize(const Vec3fa& in) {
  return in * sycl::rsqrt(dot(in, in));
}

inline Vec3fa cross(const Vec3fa& first, const Vec3fa& second) {
  return Vec3fa(first.y * second.z - first.z * second.y,
                first.z * second.x - first.x * second.z,
                first.x * second.y - first.y * second.x);
}

struct LinearSpace3fa {
  /*! default matrix constructor */
  inline LinearSpace3fa() = default;
  inline LinearSpace3fa(const LinearSpace3fa& other) {
    vx = other.vx;
    vy = other.vy;
    vz = other.vz;
  }
  inline LinearSpace3fa& operator=(const LinearSpace3fa& other) {
    vx = other.vx;
    vy = other.vy;
    vz = other.vz;
    return *this;
  }

  inline LinearSpace3fa(Vec3fa a, Vec3fa b, Vec3fa c) {
    vx = a;
    vy = b;
    vz = c;
  }
  Vec3fa vx, vy, vz;
};

struct AffineSpace3fa {
  inline AffineSpace3fa() = default;
  LinearSpace3fa l;
  Vec3fa p;
};

inline LinearSpace3fa operator*(const float& a, const LinearSpace3fa& b) {
  return LinearSpace3fa(a * b.vx, a * b.vy, a * b.vz);
}

inline Vec3fa operator*(const LinearSpace3fa& a, const Vec3fa& b) {
  return b.x * a.vx + b.y * a.vy + b.z * a.vz;
}

/* Wrapping sycl::float2 here. This makes a pathtracer kernel with 2D random sampling on a
 * hemisphere a little bit more adaptable.*/
struct Vec2f {
  union {
    sycl::float2 data;
    struct {
      float x, y;
    };
  };

  Vec2f() {}
  Vec2f(float in_x, float in_y) {
    this->x = x;
    this->y = y;
  }

  Vec2f(const Vec2f& in) {
    this->x = in.x;
    this->y = in.y;
  }

  Vec2f(const float& in) { this->x = this->y = in; }

  inline Vec2f operator+(const Vec2f& second) {
    Vec2f out;
    out.data = this->data + second.data;
    return out;
  }

  inline Vec2f operator-(const Vec2f& second) {
    Vec2f out;
    out.data = this->data - second.data;
    return out;
  }

  inline Vec2f operator/(const Vec2f& second) {
    return Vec2f(this->x / second.x, this->y / second.y);
  }

  inline Vec2f operator/(float f) { return Vec2f(this->x / f, this->y / f); }

  friend inline Vec2f operator-(const Vec2f& in) { return Vec2f(-in.x, -in.y); }

  friend inline Vec2f operator*(const Vec2f& first, const Vec2f& second) {
    return Vec2f(first.x * second.x, first.y * second.y);
  }
};

inline Vec2f operator*(const float& f, const Vec2f& in) {
  return Vec2f(in.x * f, in.y * f);
}

inline Vec2f operator*(const Vec2f& in, const float& f) {
  return Vec2f(in.x * f, in.y * f);
}

inline float dot(const Vec2f& first, const Vec2f& second) {
  return first.data[0] * second.data[0] + first.data[1] * second.data[1];
}

inline float dot(const sycl::float2& first,
                 const sycl::float2& second) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

inline Vec2f normalize(const Vec2f& in) {
  return in * sycl::rsqrt(dot(in, in));
}

inline float deg2rad(const float& x) {
  return x * float(1.745329251994329576923690768489e-2);
}

inline float clamp(const float& x) { return sycl::clamp(x, 0.f, 1.f); }

inline float clamp(const float& x, const float& y, const float& z) {
  return sycl::clamp(x, y, z);
}

}  // namespace MathBindings

#endif  // FILEMATHBINDINGSSEEN
