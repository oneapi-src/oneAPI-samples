// Copyright 2017 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArcballCamera.h"

ArcballCamera::ArcballCamera(const rkcommon::math::box3f &worldBounds,
    const rkcommon::math::vec2i &windowSize)
    : zoomSpeed(1),
      invWindowSize(
          rkcommon::math::vec2f(1.0) / rkcommon::math::vec2f(windowSize)),
      centerTranslation(rkcommon::math::one),
      translation(rkcommon::math::one),
      rotation(rkcommon::math::one)
{
  rkcommon::math::vec3f diag = worldBounds.size();
  zoomSpeed = rkcommon::math::max(length(diag) / 150.0, 0.001);
  diag = rkcommon::math::max(diag, rkcommon::math::vec3f(0.3f * length(diag)));

  centerTranslation =
      rkcommon::math::AffineSpace3f::translate(-worldBounds.center());
  translation = rkcommon::math::AffineSpace3f::translate(
      rkcommon::math::vec3f(0, 0, length(diag)));
  updateCamera();
}

void ArcballCamera::rotate(
    const rkcommon::math::vec2f &from, const rkcommon::math::vec2f &to)
{
  rotation = screenToArcball(to) * screenToArcball(from) * rotation;
  updateCamera();
}

void ArcballCamera::zoom(float amount)
{
  amount *= zoomSpeed;
  translation = rkcommon::math::AffineSpace3f::translate(
                    rkcommon::math::vec3f(0, 0, amount))
      * translation;
  updateCamera();
}

void ArcballCamera::pan(const rkcommon::math::vec2f &delta)
{
  const rkcommon::math::vec3f t = rkcommon::math::vec3f(
      -delta.x * invWindowSize.x, delta.y * invWindowSize.y, 0);
  const rkcommon::math::vec3f worldt =
      translation.p.z * xfmVector(invCamera, t);
  centerTranslation =
      rkcommon::math::AffineSpace3f::translate(worldt) * centerTranslation;
  updateCamera();
}

rkcommon::math::vec3f ArcballCamera::eyePos() const
{
  return xfmPoint(invCamera, rkcommon::math::vec3f(0, 0, 1));
}

rkcommon::math::vec3f ArcballCamera::center() const
{
  return -centerTranslation.p;
}

rkcommon::math::vec3f ArcballCamera::lookDir() const
{
  return xfmVector(invCamera, rkcommon::math::vec3f(0, 0, 1));
}

rkcommon::math::vec3f ArcballCamera::upDir() const
{
  return xfmVector(invCamera, rkcommon::math::vec3f(0, 1, 0));
}

rkcommon::math::AffineSpace3f ArcballCamera::transform() const
{
  return invCamera;
}

void ArcballCamera::updateCamera()
{
  const rkcommon::math::AffineSpace3f rot =
      rkcommon::math::LinearSpace3f(rotation);
  const rkcommon::math::AffineSpace3f camera =
      translation * rot * centerTranslation;
  invCamera = rcp(camera);
}

void ArcballCamera::setRotation(rkcommon::math::quatf q)
{
  rotation = q;
  updateCamera();
}

void ArcballCamera::updateWindowSize(const rkcommon::math::vec2i &windowSize)
{
  invWindowSize = rkcommon::math::vec2f(1) / rkcommon::math::vec2f(windowSize);
}

rkcommon::math::quatf ArcballCamera::screenToArcball(
    const rkcommon::math::vec2f &p)
{
  const float dist = dot(p, p);
  // If we're on/in the sphere return the point on it
  if (dist <= 1.f) {
    return rkcommon::math::quatf(0, p.x, p.y, std::sqrt(1.f - dist));
  } else {
    // otherwise we project the point onto the sphere
    const rkcommon::math::vec2f unitDir = normalize(p);
    return rkcommon::math::quatf(0, unitDir.x, unitDir.y, 0);
  }
}
