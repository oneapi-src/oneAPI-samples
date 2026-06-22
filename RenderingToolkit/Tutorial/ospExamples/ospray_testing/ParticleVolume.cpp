// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
// stl
#include <random>
#include <vector>

// file IO
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

// rkcommon
#include "rkcommon/math/range.h"
#include "rkcommon/math/vec.h"
#include "rkcommon/tasking/parallel_for.h"
#include "rkcommon/utility/random.h"

using namespace rkcommon;
using namespace rkcommon::math;

namespace ospray {
namespace testing {

using VoxelArray = std::vector<float>;

struct ParticleVolume : public detail::Builder
{
  ParticleVolume(int numParticles,
      bool withVolume,
      bool withIsosurface,
      bool withClipping,
      bool multipleIsosurfaces,
      bool provideWeights,
      float clampMaxCumulativeValue,
      float radiusSupportFactor);

  ~ParticleVolume() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;

 private:
  // Data //

  int numParticles{1000};
  bool withVolume{true};
  bool withIsosurface{false};
  bool withClipping{false};
  bool multipleIsosurfaces{false};
  bool provideWeights{true};
  float clampMaxCumulativeValue{0.f};
  float radiusSupportFactor{4.f};

  float isovalue{0.5f};

  range1f valueRange{0, 1.5f};
  box3f bounds{{0, 0, 0}, {10, 10, 10}};
};

// Inlined definitions ////////////////////////////////////////////////////

ParticleVolume::ParticleVolume(int numParticles,
    bool withVolume,
    bool withIsosurface,
    bool withClipping,
    bool multipleIsosurfaces,
    bool provideWeights,
    float clampMaxCumulativeValue,
    float radiusSupportFactor)
    : numParticles(numParticles),
      withVolume(withVolume),
      withIsosurface(withIsosurface),
      withClipping(withClipping),
      multipleIsosurfaces(multipleIsosurfaces),
      provideWeights(provideWeights),
      clampMaxCumulativeValue(clampMaxCumulativeValue),
      radiusSupportFactor(radiusSupportFactor)
{}

void ParticleVolume::commit()
{
  Builder::commit();

  numParticles = getParam<int>("numParticles", numParticles);
  withVolume = getParam<bool>("withVolume", withVolume);
  withIsosurface = getParam<bool>("withIsosurface", withIsosurface);
  isovalue = getParam<float>("isovalue", 0.5f);
  withClipping = getParam<bool>("withClipping", withClipping);
  multipleIsosurfaces =
      getParam<bool>("multipleIsosurfaces", multipleIsosurfaces);
  clampMaxCumulativeValue =
      getParam<float>("clampMaxCumulativeValue", clampMaxCumulativeValue);
  radiusSupportFactor =
      getParam<float>("radiusSuportFactor", radiusSupportFactor);

  addPlane = false;
}

cpp::Group ParticleVolume::buildGroup() const
{
  // AMK: Very similar to ProceduralParticleVolume.h in OpenVKL. Any way to
  // combine?

  std::vector<vec3f> particles(numParticles);
  std::vector<float> radius(numParticles);
  std::vector<float> weights(numParticles);

  int32_t randomSeed = 0;

  // create random number distributions for point center and weight
  std::mt19937 gen(randomSeed);

  range1f weightRange(0.5f, 1.5f);
  const float radiusScale = 1.f / powf(numParticles, 1.f / 3.f);

  utility::uniform_real_distribution<float> centerDistribution_x(
      bounds.lower.x, bounds.upper.x);
  utility::uniform_real_distribution<float> centerDistribution_y(
      bounds.lower.y, bounds.upper.y);
  utility::uniform_real_distribution<float> centerDistribution_z(
      bounds.lower.z, bounds.upper.z);
  utility::uniform_real_distribution<float> radiusDistribution(
      radiusScale, 2.f * radiusScale);
  utility::uniform_real_distribution<float> weightDistribution(
      weightRange.lower, weightRange.upper);

  // populate the points
  for (int i = 0; i < numParticles; i++) {
    auto &p = particles[i];
    p.x = centerDistribution_x(gen);
    p.y = centerDistribution_y(gen);
    p.z = centerDistribution_z(gen);
    radius[i] = radiusDistribution(gen);
    weights[i] = provideWeights ? weightDistribution(gen) : 1.f;
  }

  cpp::Volume volume("particle");
  volume.setParam("particle.position", cpp::CopiedData(particles));
  volume.setParam("particle.radius", cpp::CopiedData(radius));
  volume.setParam("particle.weight", cpp::CopiedData(weights));
  volume.setParam("clampMaxCumulativeValue", clampMaxCumulativeValue);
  volume.setParam("radiusSupportFactor", radiusSupportFactor);

  volume.commit();

  cpp::VolumetricModel model(volume);

  // AMK: VKL will set the correct valueRange, but we don't have a good way to
  // get that to OSPRay yet. Use the weightRange instead.
  model.setParam("transferFunction",
      makeTransferFunction(range1f(0.f, weightRange.upper)));
  model.commit();

  cpp::Group group;

  if (withVolume)
    group.setParam("volume", cpp::CopiedData(model));

  if (withIsosurface) {
    cpp::Geometry isoGeom("isosurface");
    std::vector<float> isovalues = {isovalue};
    if (multipleIsosurfaces) {
      isovalues.push_back(isovalue + .25f);
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

cpp::World ParticleVolume::buildWorld() const
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

  auto world = Builder::buildWorld(instances);

  return world;
}

OSP_REGISTER_TESTING_BUILDER(
    ParticleVolume(1000, true, false, false, false, true, 0.f, 4.f),
    particle_volume);

OSP_REGISTER_TESTING_BUILDER(
    ParticleVolume(1000, false, true, false, false, true, 0.f, 4.f),
    particle_volume_isosurface);

OSP_REGISTER_TESTING_BUILDER(
    ParticleVolume(1000, true, false, true, false, true, 0.f, 4.f),
    clip_particle_volume);

} // namespace testing
} // namespace ospray
