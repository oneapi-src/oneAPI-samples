// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct PtMetallicFlakes : public detail::Builder
{
  PtMetallicFlakes() = default;
  ~PtMetallicFlakes() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
};

static cpp::Material makeMetallicFlakesMaterial(
    const std::string &rendererType, float flakeAmount, float flakeSpread)
{
  cpp::Material mat(rendererType, "metallicPaint");
  mat.setParam("baseColor", vec3f(1.f, 0.f, 0.f));
  mat.setParam("flakeAmount", flakeAmount);
  mat.setParam("flakeSpread", flakeSpread);
  mat.commit();

  return mat;
}

// Inlined definitions ////////////////////////////////////////////////////

void PtMetallicFlakes::commit()
{
  Builder::commit();
  addPlane = false;
}

cpp::Group PtMetallicFlakes::buildGroup() const
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
    materials.push_back(makeMetallicFlakesMaterial(rendererType,
        lerp(i_f.x / (dimSize - 1), 1.f, 0.f),
        lerp(i_f.y / (dimSize - 1), 1.f, 0.f)));
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

OSP_REGISTER_TESTING_BUILDER(PtMetallicFlakes, test_pt_metallic_flakes);

} // namespace testing
} // namespace ospray
