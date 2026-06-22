// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/random.h"
// stl
#include <random>

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct Curves : public detail::Builder
{
  Curves() = default;
  ~Curves() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;

  std::string curveVariant;
};

static std::vector<vec4f> points = {{-1.0f, 0.0f, -2.f, 0.2f},
    {0.0f, -1.0f, 0.0f, 0.2f},
    {1.0f, 0.0f, 2.f, 0.2f},
    {-1.0f, 0.0f, 2.f, 0.2f},
    {0.0f, 1.0f, 0.0f, 0.6f},
    {1.0f, 0.0f, -2.f, 0.2f},
    {-1.0f, 0.0f, -2.f, 0.2f},
    {0.0f, -1.0f, 0.0f, 0.2f},
    {1.0f, 0.0f, 2.f, 0.2f}};
static std::vector<unsigned int> indices = {0, 1, 2, 3, 4, 5};

void Curves::commit()
{
  Builder::commit();

  curveVariant = getParam<std::string>("curveVariant", "bspline");
}

cpp::Group Curves::buildGroup() const
{
  std::vector<cpp::GeometricModel> geometricModels;
  std::mt19937 gen(randomSeed);
  utility::uniform_real_distribution<float> colorDistribution(0.1f, 1.0f);
  std::vector<vec4f> s_colors(points.size());

  cpp::Geometry geom("curve");
  // defaults, "linear"
  geom.setParam("type", OSP_ROUND);
  geom.setParam("basis", OSP_LINEAR);
  geom.setParam("vertex.position_radius", cpp::CopiedData(points));

  if (curveVariant == "hermite") {
    geom.setParam("basis", OSP_HERMITE);
    std::vector<vec4f> tangents;
    for (auto iter = points.begin(); iter != points.end() - 1; ++iter) {
      const vec4f pointTangent = *(iter + 1) - *iter;
      tangents.push_back(pointTangent);
    }
    geom.setParam("vertex.tangent", cpp::CopiedData(tangents));
  } else if (curveVariant == "catmull-rom") {
    geom.setParam("basis", OSP_CATMULL_ROM);
  } else if (curveVariant == "cones") {
    geom.setParam("type", OSP_DISJOINT);
  } else if (curveVariant == "linear_deprecated") {
    geom.removeParam("vertex.position_radius");
    geom.setParam("radius", 0.1f);
    geom.setParam("vertex.position",
        cpp::CopiedData((vec3f *)points.data(), points.size(), sizeof(vec4f)));
  } else if (curveVariant == "bspline") {
    geom.setParam("basis", OSP_BSPLINE);
  }

  for (auto &c : s_colors) {
    c.x = colorDistribution(gen);
    c.y = colorDistribution(gen);
    c.z = colorDistribution(gen);
    c.w = colorDistribution(gen);
  }

  geom.setParam("vertex.color", cpp::CopiedData(s_colors));

  geom.setParam("index", cpp::CopiedData(indices));
  geom.commit();

  cpp::GeometricModel model(geom);

  if (rendererType == "pathtracer") {
    // create glass material and assign to geometry
    cpp::Material glassMaterial(rendererType.c_str(), "thinGlass");
    glassMaterial.setParam("attenuationDistance", 1.f);
    glassMaterial.commit();
    model.setParam("material", glassMaterial);
  } else if (rendererType == "scivis" || rendererType == "ao") {
    cpp::Material glassMaterial(rendererType.c_str(), "obj");
    glassMaterial.commit();
    model.setParam("material", glassMaterial);
  }

  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

OSP_REGISTER_TESTING_BUILDER(Curves, curves);

} // namespace testing
} // namespace ospray
