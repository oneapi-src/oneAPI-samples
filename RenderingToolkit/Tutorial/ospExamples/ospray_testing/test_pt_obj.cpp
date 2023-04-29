// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct PtObj : public detail::Builder
{
  PtObj() = default;
  ~PtObj() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;
};

static cpp::Material makeObjMaterial(
    const std::string &rendererType, vec3f Kd, vec3f Ks)
{
  cpp::Material mat(rendererType, "obj");
  mat.setParam("kd", Kd);
  mat.setParam("ks", Ks);
  mat.commit();

  return mat;
}

// Inlined definitions ////////////////////////////////////////////////////

void PtObj::commit()
{
  Builder::commit();
  addPlane = false;
}

cpp::Group PtObj::buildGroup() const
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

    auto l = i_f / (dimSize - 1);
    materials.push_back(makeObjMaterial(rendererType,
        lerp(l.x, vec3f(0.1f), vec3f(1.f, 0.f, 0.f)),
        lerp(l.y, vec3f(0.f), vec3f(1.f))));

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

cpp::World PtObj::buildWorld() const
{
  auto world = Builder::buildWorld();

  std::vector<cpp::Light> lightHandles;

  cpp::Light dirLight("distant");
  dirLight.setParam("direction", vec3f(1.f, -1.f, 1.f));
  dirLight.commit();

  cpp::Light ambientLight("ambient");
  ambientLight.setParam("intensity", 0.4f);
  ambientLight.setParam("color", vec3f(1.f));
  ambientLight.commit();

  lightHandles.push_back(dirLight);
  lightHandles.push_back(ambientLight);

  world.setParam("light", cpp::CopiedData(lightHandles));

  return world;
}

OSP_REGISTER_TESTING_BUILDER(PtObj, test_pt_obj);

} // namespace testing
} // namespace ospray
