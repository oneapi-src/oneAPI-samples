// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"
#include "rkcommon/utility/random.h"
// stl
#include <random>

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct Instancing : public detail::Builder
{
  Instancing() = default;
  ~Instancing() override = default;

  void commit() override;

  cpp::Group buildGroup() const override
  {
    return cpp::Group();
  }
  cpp::Group buildGroupA() const;
  cpp::Group buildGroupB() const;
  cpp::Group buildGroupC() const;
  cpp::World buildWorld() const override;

 private:
  vec2ui numInstances{10, 10};
  bool useIDs{false};
};

// Inlined definitions ////////////////////////////////////////////////////

// Helper functions //
namespace {

cpp::Geometry makeBoxGeometry(const box3f &box)
{
  cpp::Geometry ospGeometry("box");
  ospGeometry.setParam("box", cpp::CopiedData(box));
  ospGeometry.commit();
  return ospGeometry;
}

cpp::GeometricModel makeGeometricModel(cpp::Geometry geo,
    const std::string &rendererType,
    const vec3f &kd,
    bool useIDs)
{
  cpp::GeometricModel geometricModel(geo);

  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    cpp::Material objMaterial(rendererType, "obj");
    objMaterial.setParam("kd", kd);
    objMaterial.commit();
    geometricModel.setParam("material", objMaterial);
  }
  if (useIDs)
    geometricModel.setParam("id", 44u);
  geometricModel.commit();

  return geometricModel;
}

cpp::Volume makeVolume(const vec3f &center)
{
  cpp::Volume volume("structuredRegular");
  {
    const vec3i volumeCells{10, 10, 10};
    const vec3f spacing{.25f};
    volume.setParam("gridOrigin", center - volumeCells * spacing / 2.f);
    volume.setParam("gridSpacing", spacing);
    std::vector<float> voxels(volumeCells.long_product());
    index_sequence_3D numCell(volumeCells);
    for (const vec3i i : numCell)
      voxels[numCell.flatten(i)] = length(volumeCells - i / 2.f);
    volume.setParam("data", cpp::CopiedData(voxels.data(), volumeCells));
    volume.commit();
  }
  return volume;
}

cpp::VolumetricModel makeVolumetricModel(cpp::Volume volume, bool useIDs)
{
  cpp::VolumetricModel vModel(volume);
  {
    // Create transfer function
    cpp::TransferFunction transferFunction("piecewiseLinear");
    {
      std::vector<vec3f> colors = {{.1f, .4f, .8f}};
      std::vector<float> opacities = {1.f};
      transferFunction.setParam("color", cpp::CopiedData(colors));
      transferFunction.setParam("opacity", cpp::CopiedData(opacities));
      transferFunction.setParam("value", range1f(0.f, 10.f));
      transferFunction.commit();
    }

    vModel.setParam("transferFunction", transferFunction);
    vModel.setParam("densityScale", .5f);
    vModel.setParam("gradientShadingScale", 1.f);
    if (useIDs)
      vModel.setParam("id", 3u);
    vModel.commit();
  }
  return vModel;
}

} // namespace

// Instancing definitions //

void Instancing::commit()
{
  Builder::commit();

  numInstances = getParam<vec2ui>("numInstances", numInstances);
  useIDs = getParam<bool>("useIDs", useIDs);
}

cpp::Group Instancing::buildGroupA() const
{
  cpp::Group group;

  cpp::Geometry box = makeBoxGeometry(box3f(vec3f(-1.f), vec3f(1.f)));
  cpp::GeometricModel geometricModel =
      makeGeometricModel(box, rendererType, vec3f(.1f, .4f, .8f), useIDs);
  group.setParam("geometry", cpp::CopiedData(geometricModel));

  cpp::Light light("quad");
  light.setParam("position", vec3f(0.f, 6.f, 0.f));
  light.setParam("edge1", vec3f(0.f, 0.f, -1.f));
  light.setParam("edge2", vec3f(1.f, 0.f, 0.f));
  light.setParam("intensity", 2.0f);
  light.setParam("color", vec3f(2.6f, 2.5f, 2.3f));
  light.commit();
  group.setParam("light", cpp::CopiedData(light));

  group.commit();
  return group;
}

cpp::Group Instancing::buildGroupB() const
{
  cpp::Group group;
  std::vector<cpp::GeometricModel> models;

  cpp::Geometry box = makeBoxGeometry(box3f(vec3f(-1.f), vec3f(0.f, 1.f, 1.f)));
  models.emplace_back(
      makeGeometricModel(box, rendererType, vec3f(.1f, .4f, .8f), useIDs));
  box = makeBoxGeometry(box3f(vec3f(0.f, -1.f, -1.f), vec3f(1.f)));
  models.emplace_back(
      makeGeometricModel(box, rendererType, vec3f(.1f, .4f, .8f), useIDs));
  group.setParam("geometry", cpp::CopiedData(models));

  cpp::Light light("spot");
  light.setParam("position", vec3f(0.f, 6.f, 0.f));
  light.setParam("direction", vec3f(0.f, -1.f, 0.f));
  light.setParam("openingAngle", 180.f);
  light.setParam("penumbraAngle", 0.f);
  light.setParam("radius", 0.3f);
  light.setParam("intensity", 2.0f);
  float lid1d[] = {0.f, 2.4f, 0.2f, 0.1f, 0.03f, 0.01f, 0.01f};
  light.setParam("intensityDistribution", cpp::CopiedData(lid1d, 7));
  light.setParam("color", vec3f(2.6f, 2.5f, 2.3f));
  light.commit();
  group.setParam("light", cpp::CopiedData(light));

  group.commit();
  return group;
}

cpp::Group Instancing::buildGroupC() const
{
  cpp::Group group;

  // Create volumetric model
  group.setParam("volume",
      cpp::CopiedData(makeVolumetricModel(makeVolume(vec3f(0.f)), useIDs)));

  cpp::Light light("sphere");
  light.setParam("position", vec3f(0.f, 2.f, 0.f));
  light.setParam("radius", .25f);
  light.setParam("intensity", 2.0f);
  light.setParam("color", vec3f(2.6f, 2.5f, 2.3f));
  light.commit();
  group.setParam("light", cpp::CopiedData(light));

  group.commit();
  return group;
}

cpp::World Instancing::buildWorld() const
{
  // Create all groups
  std::vector<cpp::Group> groups;
  groups.push_back(buildGroupA());
  groups.push_back(buildGroupB());
  groups.push_back(buildGroupC());

  // Create instances
  std::vector<cpp::Instance> instances;
  {
    const float spacing = 2.5f;
    const vec2f totalSize = spacing * (numInstances - 1);
    box3f sceneBounds;
    index_sequence_2D indexInstances(numInstances);
    for (const vec2ui i : indexInstances) {
      const float h = .3f * (sin(.7f * i.x) + sin(.7f * i.y));
      const vec3f position = spacing * vec3f(i.x, h, i.y)
          - 0.5f * vec3f(totalSize.x, 0.0f, totalSize.y);
      const float d = length(vec2f(position.x, position.z));
      const int gId = int(.18f * d) % groups.size();
      cpp::Instance instance(groups[gId]);
      if (useIDs)
        instance.setParam("id", i.y + 1);
      const affine3f xfm = affine3f::translate(position)
          * affine3f::rotate(vec3f(0.f, 1.f, 0.f), .03f * (15.f - d));
      instance.setParam("transform",
          affine3f::rotate(vec3f(-1.f, 0.f, 0.f), .3f * float(pi)) * xfm);
      instance.commit();
      instances.push_back(instance);
    }
  }

  // Create world
  cpp::World world;
  world.setParam("instance", cpp::CopiedData(instances));

  // Done
  return world;
}

OSP_REGISTER_TESTING_BUILDER(Instancing, instancing);

} // namespace testing
} // namespace ospray
