// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct Planes : public detail::Builder
{
  Planes() = default;
  ~Planes() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
};

// Inlined definitions ////////////////////////////////////////////////////

void Planes::commit()
{
  Builder::commit();

  addPlane = false;
}

cpp::Group Planes::buildGroup() const
{
  // Create plane geometry
  cpp::Geometry planeGeometry("plane");
  std::vector<vec4f> coeffs;
  std::vector<box3f> bounds;
  const box3f b = box3f(vec3f(-2.f, -2.f, -1.f), vec3f(2.f, 2.f, 1.f));
  const uint32_t count = 28;
  for (uint32_t i = 0; i < count; i++) {
    affine3f rotate = affine3f::rotate(
        vec3f(.3f, 0.f, 1.f), (i / float(count)) * 2.0f * M_PI);
    vec3f p = xfmNormal(rotate, vec3f(1.f, 0.f, 0.f));
    coeffs.push_back(vec4f(p.x, p.y, p.z, 0.f));
    bounds.push_back(b);
  }
  planeGeometry.setParam("plane.coefficients", cpp::CopiedData(coeffs));
  planeGeometry.setParam("plane.bounds", cpp::CopiedData(bounds));
  planeGeometry.commit();

  // Create geometric model
  cpp::GeometricModel model(planeGeometry);
  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    cpp::Material material(rendererType, "obj");
    material.setParam("kd", vec3f(.1f, .4f, .8f));
    material.commit();
    model.setParam("material", material);
  }
  model.commit();

  // Create group
  cpp::Group group;
  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  // Done
  return group;
}

OSP_REGISTER_TESTING_BUILDER(Planes, planes);

} // namespace testing
} // namespace ospray
