// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct PtMetalRoughness : public detail::Builder
{
  PtMetalRoughness() = default;
  ~PtMetalRoughness() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
};

static cpp::Material makeMetalMaterial(
    const std::string &rendererType, vec3f eta, vec3f k, float roughness)
{
  cpp::Material mat(rendererType, "metal");
  mat.setParam("eta", eta);
  mat.setParam("k", k);
  mat.setParam("roughness", roughness);
  mat.commit();

  return mat;
}

// Inlined definitions ////////////////////////////////////////////////////

void PtMetalRoughness::commit()
{
  Builder::commit();
  addPlane = false;
}

cpp::Group PtMetalRoughness::buildGroup() const
{
  cpp::Geometry sphereGeometry("sphere");

  constexpr int dimSize = 5;

  index_sequence_2D numSpheres(dimSize);

  std::vector<vec3f> spheres;
  std::vector<uint8_t> index;

  for (auto i : numSpheres) {
    auto i_f = static_cast<vec2f>(i);
    spheres.emplace_back(i_f.x, i_f.y, 0.f);
    index.push_back(numSpheres.flatten(i));
  }

  sphereGeometry.setParam("sphere.position", cpp::CopiedData(spheres));
  sphereGeometry.setParam("radius", 0.4f);
  sphereGeometry.commit();

  cpp::GeometricModel model(sphereGeometry);

  std::vector<cpp::Material> materials;

  // Silver
  for (int i = 0; i < dimSize; ++i) {
    materials.push_back(makeMetalMaterial(rendererType,
        vec3f(0.051, 0.043, 0.041),
        vec3f(5.3, 3.6, 2.3),
        float(i) / 4));
  }

  // Aluminium
  for (int i = 0; i < dimSize; ++i) {
    materials.push_back(makeMetalMaterial(rendererType,
        vec3f(1.5, 0.98, 0.6),
        vec3f(7.6, 6.6, 5.4),
        float(i) / 4));
  }

  // Gold
  for (int i = 0; i < dimSize; ++i) {
    materials.push_back(makeMetalMaterial(rendererType,
        vec3f(0.07, 0.37, 1.5),
        vec3f(3.7, 2.3, 1.7),
        float(i) / 4));
  }

  // Chromium
  for (int i = 0; i < dimSize; ++i) {
    materials.push_back(makeMetalMaterial(rendererType,
        vec3f(3.2, 3.1, 2.3),
        vec3f(3.3, 3.3, 3.1),
        float(i) / 4));
  }

  // Copper
  for (int i = 0; i < dimSize; ++i) {
    materials.push_back(makeMetalMaterial(rendererType,
        vec3f(0.1, 0.8, 1.1),
        vec3f(3.5, 2.5, 2.4),
        float(i) / 4));
  }

  model.setParam("material", cpp::CopiedData(materials));
  model.setParam("index", cpp::CopiedData(index));
  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

OSP_REGISTER_TESTING_BUILDER(PtMetalRoughness, test_pt_metal_roughness);

} // namespace testing
} // namespace ospray
