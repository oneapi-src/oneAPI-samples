// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct Boxes : public detail::Builder
{
  Boxes(bool lights = false) : useLights(lights) {}
  ~Boxes() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;


 private:
  vec3i dimensions{4};
  bool useLights{false};
};

// Inlined definitions ////////////////////////////////////////////////////

void Boxes::commit()
{
  Builder::commit();

  dimensions = getParam<vec3i>("dimensions", vec3i(4));

  addPlane = false;
}

cpp::Group Boxes::buildGroup() const
{
  cpp::Geometry boxGeometry("box");

  index_sequence_3D numBoxes(dimensions);

  std::vector<box3f> boxes;
  std::vector<vec4f> color;

  auto dim = reduce_max(dimensions);

  for (auto i : numBoxes) {
    auto i_f = static_cast<vec3f>(i);

    auto lower = i_f * 5.f;
    auto upper = lower + (0.75f * 5.f);
    boxes.emplace_back(lower, upper);

    auto box_color = (0.8f * i_f / dim) + 0.2f;
    color.emplace_back(box_color.x, box_color.y, box_color.z, 1.f);
  }

  boxGeometry.setParam("box", cpp::CopiedData(boxes));
  boxGeometry.commit();

  cpp::GeometricModel model(boxGeometry);

  model.setParam("color", cpp::CopiedData(color));

  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    cpp::Material material(rendererType, "obj");
    if (rendererType == "pathtracer" || rendererType == "scivis") {
      material.setParam("ks", vec3f(0.3f));
      material.setParam("ns", 10.f);
    }
    material.commit();
    model.setParam("material", material);
  }

  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

cpp::World Boxes::buildWorld() const
{
  auto world = Builder::buildWorld();
  if (!useLights)
    return world;
  cpp::Light light("distant");
  light.setParam("color", vec3f(0.78f, 0.551f, 0.483f));
  light.setParam("intensity", 3.14f);
  light.setParam("direction", vec3f(-0.8f, -0.6f, 0.3f));
  light.commit();
  cpp::Light ambient("ambient");
  ambient.setParam("intensity", 0.35f);
  ambient.setParam("visible", false);
  ambient.commit();
  std::vector<cpp::Light> lights{light, ambient};
  world.setParam("light", cpp::CopiedData(lights));
  return world;
}

OSP_REGISTER_TESTING_BUILDER(Boxes, boxes);
OSP_REGISTER_TESTING_BUILDER(Boxes(true), boxes_lit);

} // namespace testing
} // namespace ospray
