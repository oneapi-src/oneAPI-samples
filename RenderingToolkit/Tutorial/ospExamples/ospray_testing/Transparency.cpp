// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct Transparency : public detail::Builder
{
  Transparency() = default;
  ~Transparency() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;
};

// Inlined definitions ////////////////////////////////////////////////////

void Transparency::commit()
{
  Builder::commit();

  addPlane = true;
}

cpp::Group Transparency::buildGroup() const
{
  // Create plane geometry
  cpp::Geometry planeGeometry("plane");
  {
    std::vector<vec4f> coeffs = {{0.f, 0.f, 1.f, 0.f},
        {0.f, 0.f, 1.f, 0.f},
        {0.f, 0.f, 1.f, 0.f},
        {0.f, 0.f, 1.f, 0.f},
        {0.f, 0.f, 1.f, -.4f},
        {0.f, 0.f, 1.f, -.45f},
        {0.f, 0.f, 1.f, -.5f}};
    std::vector<box3f> bounds = {{{-2.f, 1.8f, -2.f}, {2.f, 2.f, 2.f}},
        {{-2.f, 1.6f, -2.f}, {2.f, 1.8f, 2.f}},
        {{-2.f, 1.4f, -2.f}, {2.f, 1.6f, 2.f}},
        {{-2.f, 0.2f, -2.f}, {2.f, 1.4f, 2.f}},
        {{-1.7f, 1.f, -1.f}, {-.3f, 2.f, 1.f}},
        {{-.7f, 1.f, -1.f}, {.7f, 2.f, 1.f}},
        {{.3f, 1.f, -1.f}, {1.7f, 2.f, 1.f}}};
    planeGeometry.setParam("plane.coefficients", cpp::CopiedData(coeffs));
    planeGeometry.setParam("plane.bounds", cpp::CopiedData(bounds));
    planeGeometry.commit();
  }

  // Create geometric model
  cpp::GeometricModel gModel(planeGeometry);
  {
    if (rendererType == "pathtracer" || rendererType == "scivis"
        || rendererType == "ao") {
      std::vector<vec3f> colors = {{.8f, 0.f, 0.f},
          {0.f, .8f, 0.f},
          {0.f, 0.f, .8f},
          {.8f, .8f, .8f},
          {.5f, 0.f, 0.f},
          {0.f, .5f, 0.f},
          {0.f, 0.f, .5f}};
      std::vector<vec3f> transmissions = {{0.f, 0.f, 0.f},
          {0.f, 0.f, 0.f},
          {0.f, 0.f, 0.f},
          {0.f, 0.f, 0.f},
          {.8f, 0.f, 0.f},
          {0.f, .8f, 0.f},
          {0.f, 0.f, .8f}};
      std::vector<cpp::Material> materials;
      for (uint32_t i = 0; i < std::min(colors.size(), transmissions.size());
           i++) {
        materials.emplace_back(cpp::Material(rendererType, "obj"));
        materials[i].setParam("kd", colors[i]);
        if (rendererType == "pathtracer" || rendererType == "scivis")
          materials[i].setParam("tf", transmissions[i]);
        materials[i].commit();
      }
      gModel.setParam("material", cpp::CopiedData(materials));
    }
    gModel.commit();
  }

  // Create volume object with values being shortest distance to surface for
  // gradient shading working correctly
  cpp::Volume volume("structuredRegular");
  {
    volume.setParam("gridOrigin", vec3f(-2.f, -.7f, -.1f));
    volume.setParam("gridSpacing", vec3f(.1f));
    const vec3i volumeCells{41, 10, 5};
    std::vector<float> voxels(volumeCells.long_product());
    index_sequence_3D numCell(volumeCells);
    for (vec3i i : numCell) {
      voxels[numCell.flatten(i)] = (float)std::min(
          {i.x < (volumeCells.x / 2) ? i.x : volumeCells.x - i.x - 1,
              i.y < (volumeCells.y / 2) ? i.y : volumeCells.y - i.y - 1,
              i.z < (volumeCells.z / 2) ? i.z : volumeCells.z - i.z - 1});
    }
    volume.setParam("data", cpp::CopiedData(voxels.data(), volumeCells));
    volume.commit();
  }

  // Create volume model
  cpp::VolumetricModel vModel(volume);
  {
    // Create transfer function
    cpp::TransferFunction transferFunction("piecewiseLinear");
    {
      std::vector<vec3f> colors = {{1.f, 1.f, 1.f}};
      std::vector<float> opacities = {1.f};
      transferFunction.setParam("color", cpp::CopiedData(colors));
      transferFunction.setParam("opacity", cpp::CopiedData(opacities));
      transferFunction.setParam("value", range1f(0.f, 10.f));
      transferFunction.commit();
    }

    vModel.setParam("transferFunction", transferFunction);
    vModel.setParam("densityScale", 10.f);
    vModel.setParam("gradientShadingScale", 1.f);
    vModel.commit();
  }

  // Create group
  cpp::Group group;
  group.setParam("geometry", cpp::CopiedData(gModel));
  group.setParam("volume", cpp::CopiedData(vModel));
  group.commit();

  // Done
  return group;
}

cpp::World Transparency::buildWorld() const
{
  auto world = Builder::buildWorld();

  cpp::Light light("distant");
  light.setParam("color", vec3f(0.78f, 0.551f, 0.483f));
  light.setParam("intensity", 3.14f);
  light.setParam("direction", vec3f(0.f, -0.75f, 0.25f));
  light.commit();
  cpp::Light ambient("ambient");
  ambient.setParam("intensity", 0.5f);
  ambient.setParam("visible", false);
  ambient.commit();
  std::vector<cpp::Light> lights{light, ambient};
  world.setParam("light", cpp::CopiedData(lights));
  return world;
}

OSP_REGISTER_TESTING_BUILDER(Transparency, transparency);

} // namespace testing
} // namespace ospray
