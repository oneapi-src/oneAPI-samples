// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/tasking/parallel_for.h"
#include "rkcommon/utility/random.h"
// stl
#include <random>
#include <vector>
// raw_to_amr
#include "rawToAMR.h"

using namespace rkcommon;
using namespace rkcommon::math;

namespace ospray {
namespace testing {

using VoxelArray = std::vector<float>;

struct GravitySpheres : public detail::Builder
{
  GravitySpheres(bool addVolume = true,
      bool asAMR = false,
      bool addIsosurface = false,
      bool clip = false,
      bool multipleIsosurfaces = false);
  ~GravitySpheres() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;

 private:
  VoxelArray generateVoxels() const;

  cpp::Volume createStructuredVolume(const VoxelArray &voxels) const;
  cpp::Volume createAMRVolume(const VoxelArray &voxels) const;

  // Data //

  vec3i volumeDimensions{128};
  int numPoints{10};
  bool withVolume{true};
  bool createAsAMR{false};
  bool withIsosurface{false};
  float isovalue{2.5f};
  bool withClipping{false};
  bool multipleIsosurfaces{false};
};

// Inlined definitions ////////////////////////////////////////////////////

GravitySpheres::GravitySpheres(bool addVolume,
    bool asAMR,
    bool addIsosurface,
    bool clip,
    bool multipleIsosurfaces)
    : withVolume(addVolume),
      createAsAMR(asAMR),
      withIsosurface(addIsosurface),
      withClipping(clip),
      multipleIsosurfaces(multipleIsosurfaces)
{}

void GravitySpheres::commit()
{
  Builder::commit();

  volumeDimensions = getParam<vec3i>("volumeDimensions", vec3i(128));
  numPoints = getParam<int>("numPoints", 10);
  withVolume = getParam<bool>("withVolume", withVolume);
  createAsAMR = getParam<bool>("asAMR", createAsAMR);
  withIsosurface = getParam<bool>("withIsosurface", withIsosurface);
  isovalue = getParam<float>("isovalue", 2.5f);
  withClipping = getParam<bool>("withClipping", withClipping);

  addPlane = false;
}

cpp::Group GravitySpheres::buildGroup() const
{
  auto voxels = generateVoxels();
  auto voxelRange = range1f(0.f, 10.f);

  cpp::Volume volume =
      createAsAMR ? createAMRVolume(voxels) : createStructuredVolume(voxels);

  cpp::VolumetricModel model(volume);
  model.setParam("transferFunction", makeTransferFunction(voxelRange));
  model.commit();

  cpp::Group group;

  if (withVolume)
    group.setParam("volume", cpp::CopiedData(model));

  if (withIsosurface) {
    cpp::Geometry isoGeom("isosurface");
    std::vector<float> isovalues = {isovalue};
    if (multipleIsosurfaces) {
      isovalues.push_back(isovalue + 1.f);
    }

    isoGeom.setParam("isovalue", cpp::CopiedData(isovalues));
    isoGeom.setParam("volume", volume);
    isoGeom.commit();

    cpp::GeometricModel isoModel(isoGeom);

    if (rendererType == "pathtracer" || rendererType == "scivis"
        || rendererType == "ao") {
      cpp::Material mat(rendererType, "obj");
      mat.setParam("kd", vec3f(1.f));
      mat.setParam("d", 0.5f);
      if (rendererType == "pathtracer" || rendererType == "scivis")
        mat.setParam("ks", vec3f(0.2f));
      mat.commit();

      if (multipleIsosurfaces) {
        std::vector<vec4f> colors = {
            vec4f(0.2f, 0.2f, 0.8f, 1.f), vec4f(0.8f, 0.2f, 0.2f, 1.f)};
        isoModel.setParam("color", cpp::CopiedData(colors));
      }
      isoModel.setParam("material", mat);
    }

    isoModel.commit();

    group.setParam("geometry", cpp::CopiedData(isoModel));
  }

  group.commit();

  return group;
}

cpp::World GravitySpheres::buildWorld() const
{
  // Skip clipping if not enabled
  std::vector<cpp::Instance> instances;
  if (withClipping) {
    // Create clipping plane
    std::vector<cpp::GeometricModel> geometricModels;
    {
      cpp::Geometry planeGeometry("plane");
      std::vector<vec4f> coefficients = {vec4f(1.f, -1.f, 1.f, 0.f)};
      planeGeometry.setParam(
          "plane.coefficients", cpp::CopiedData(coefficients));
      planeGeometry.commit();

      cpp::GeometricModel model(planeGeometry);
      model.commit();
      geometricModels.emplace_back(model);
    }

    // Create clipping sphere
    {
      cpp::Geometry sphereGeometry("sphere");
      std::vector<vec3f> position = {vec3f(.2f, -.2f, .2f)};
      sphereGeometry.setParam("sphere.position", cpp::CopiedData(position));
      sphereGeometry.setParam("radius", .5f);
      sphereGeometry.commit();

      cpp::GeometricModel model(sphereGeometry);
      model.commit();
      geometricModels.emplace_back(model);
    }

    cpp::Group group;
    group.setParam("clippingGeometry", cpp::CopiedData(geometricModels));
    group.commit();

    cpp::Instance inst(group);
    inst.commit();
    instances.push_back(inst);
  }

  return Builder::buildWorld(instances);
}

std::vector<float> GravitySpheres::generateVoxels() const
{
  struct Point
  {
    vec3f center;
    float weight;
  };

  // create random number distributions for point center and weight
  std::mt19937 gen(randomSeed);

  utility::uniform_real_distribution<float> centerDistribution(-1.f, 1.f);
  utility::uniform_real_distribution<float> weightDistribution(0.1f, 0.3f);

  // populate the points
  std::vector<Point> points(numPoints);

  for (auto &p : points) {
    p.center.x = centerDistribution(gen);
    p.center.y = centerDistribution(gen);
    p.center.z = centerDistribution(gen);

    p.weight = weightDistribution(gen);
  }

  // get world coordinate in [-1.f, 1.f] from logical coordinates in [0,
  // volumeDimension)
  auto logicalToWorldCoordinates = [&](int i, int j, int k) {
    return vec3f(-1.f + float(i) / float(volumeDimensions.x - 1) * 2.f,
        -1.f + float(j) / float(volumeDimensions.y - 1) * 2.f,
        -1.f + float(k) / float(volumeDimensions.z - 1) * 2.f);
  };

  // generate voxels
  std::vector<float> voxels(volumeDimensions.long_product());

  tasking::parallel_for(volumeDimensions.z, [&](int k) {
    for (int j = 0; j < volumeDimensions.y; j++) {
      for (int i = 0; i < volumeDimensions.x; i++) {
        // index in array
        size_t index = size_t(k) * volumeDimensions.z * volumeDimensions.y
            + size_t(j) * volumeDimensions.x + size_t(i);

        // compute volume value
        float value = 0.f;

        for (auto &p : points) {
          vec3f pointCoordinate = logicalToWorldCoordinates(i, j, k);
          const float distance = length(pointCoordinate - p.center);

          // contribution proportional to weighted inverse-square distance
          // (i.e. gravity)
          value += p.weight / (distance * distance);
        }

        voxels[index] = value;
      }
    }
  });

  return voxels;
}

cpp::Volume GravitySpheres::createStructuredVolume(
    const VoxelArray &voxels) const
{
  cpp::Volume volume("structuredRegular");

  volume.setParam("gridOrigin", vec3f(-1.f));
  volume.setParam("gridSpacing", vec3f(2.f / reduce_max(volumeDimensions)));
  volume.setParam("data", cpp::CopiedData(voxels.data(), volumeDimensions));
  volume.commit();
  return volume;
}

cpp::Volume GravitySpheres::createAMRVolume(const VoxelArray &voxels) const
{
  const int numLevels = 2;
  const int blockSize = 16;
  const int refinementLevel = 4;
  const float threshold = 1.0;

  std::vector<box3i> blockBounds;
  std::vector<int> refinementLevels;
  std::vector<float> cellWidths;
  std::vector<std::vector<float>> blockDataVectors;
  std::vector<cpp::CopiedData> blockData;

  // convert the structured volume to AMR
  ospray::amr::makeAMR(voxels,
      volumeDimensions,
      numLevels,
      blockSize,
      refinementLevel,
      threshold,
      blockBounds,
      refinementLevels,
      cellWidths,
      blockDataVectors);

  for (const std::vector<float> &bd : blockDataVectors)
    blockData.emplace_back(bd.data(), OSP_FLOAT, bd.size());

  // create an AMR volume and assign attributes
  cpp::Volume volume("amr");

  int toplevelVolDim =
      reduce_max(volumeDimensions) / std::pow(refinementLevel, numLevels - 1);
  volume.setParam("gridOrigin", vec3f(-1.f));
  volume.setParam("gridSpacing", vec3f(2.f / toplevelVolDim));
  volume.setParam("block.data", cpp::CopiedData(blockData));
  volume.setParam("block.bounds", cpp::CopiedData(blockBounds));
  volume.setParam("block.level", cpp::CopiedData(refinementLevels));
  volume.setParam("cellWidth", cpp::CopiedData(cellWidths));

  volume.commit();

  return volume;
}

OSP_REGISTER_TESTING_BUILDER(GravitySpheres, gravity_spheres_volume);

OSP_REGISTER_TESTING_BUILDER(
    GravitySpheres(true, true, false), gravity_spheres_amr);

OSP_REGISTER_TESTING_BUILDER(GravitySpheres(false, false, true, false, true),
    gravity_spheres_isosurface);

OSP_REGISTER_TESTING_BUILDER(
    GravitySpheres(true, false, false, true), clip_gravity_spheres_volume);

} // namespace testing
} // namespace ospray
