// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct PtLuminous : public detail::Builder
{
  PtLuminous() = default;
  ~PtLuminous() override = default;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;
};

// Inlined definitions ////////////////////////////////////////////////////

cpp::Group PtLuminous::buildGroup() const
{
  cpp::Geometry sphereGeometry("sphere");

  sphereGeometry.setParam("sphere.position", cpp::CopiedData(vec3f(0.f)));
  sphereGeometry.setParam("radius", 1.f);
  sphereGeometry.commit();

  cpp::GeometricModel model(sphereGeometry);

  cpp::Material material(rendererType, "luminous");
  material.setParam("color", vec3f(0.7f, 0.7f, 1.f));
  material.commit();

  model.setParam("material", material);
  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

cpp::World PtLuminous::buildWorld() const
{
  auto world = Builder::buildWorld();
  world.removeParam("light");
  return world;
}

OSP_REGISTER_TESTING_BUILDER(PtLuminous, test_pt_luminous);

} // namespace testing
} // namespace ospray
