// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct PtGlass : public detail::Builder
{
  PtGlass() = default;
  ~PtGlass() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
};

static cpp::Material makeGlassMaterial(const std::string &rendererType,
    float eta,
    vec3f color,
    float attenuationDistance)
{
  cpp::Material mat(rendererType, "glass");
  mat.setParam("eta", eta);
  mat.setParam("attenuationColor", color);
  mat.setParam("attenuationDistance", attenuationDistance);
  mat.commit();

  return mat;
}

// Inlined definitions ////////////////////////////////////////////////////

void PtGlass::commit()
{
  Builder::commit();
  addPlane = false;
}

cpp::Group PtGlass::buildGroup() const
{
  cpp::Geometry sphereGeometry("sphere");

  constexpr int dimSize = 5;

  index_sequence_2D numSpheres(dimSize);

  std::vector<vec3f> spheres;
  std::vector<uint8_t> index;
  std::vector<cpp::Material> materials;

  for (auto i : numSpheres) {
    auto i_f = static_cast<vec2f>(i);
    spheres.emplace_back(i_f.x, i_f.y, 0.f);
    materials.push_back(makeGlassMaterial(rendererType,
        lerp(i_f.x / (dimSize - 1), 1.f, 5.f),
        vec3f(1.f, 0.25f, 0.f),
        lerp(i_f.y / (dimSize - 1), 0.5f, 5.f)));
    index.push_back(numSpheres.flatten(i));
  }

  sphereGeometry.setParam("sphere.position", cpp::CopiedData(spheres));
  sphereGeometry.setParam("radius", 0.4f);
  sphereGeometry.commit();

  cpp::GeometricModel model(sphereGeometry);

  model.setParam("material", cpp::CopiedData(materials));
  model.setParam("index", cpp::CopiedData(index));
  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

OSP_REGISTER_TESTING_BUILDER(PtGlass, test_pt_glass);

} // namespace testing
} // namespace ospray
