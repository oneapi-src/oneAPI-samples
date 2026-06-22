// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "Noise.h"
#include "ospray_testing.h"
#include "rkcommon/math/AffineSpace.h"
#include "rkcommon/tasking/parallel_for.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct VdbVolume : public detail::Builder
{
  VdbVolume(bool packed = false) : packed(packed) {}
  ~VdbVolume() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;

 private:
  float densityScale{1.f};
  float anisotropy{0.f};
  bool packed{false};
  static constexpr uint32_t domainRes = 128;
};

// Inlined definitions ////////////////////////////////////////////////////

void VdbVolume::commit()
{
  Builder::commit();

  densityScale = getParam<float>("densityScale", 20.f);
  anisotropy = getParam<float>("anisotropy", 0.f);
}

cpp::Group VdbVolume::buildGroup() const
{
  cpp::Volume volume("vdb");

  std::vector<vec3i> origin;
  std::vector<cpp::CopiedData> data;
  std::vector<float> dataPacked;

  constexpr uint32_t leafRes = 8;
  constexpr uint32_t N = domainRes / leafRes;
  constexpr size_t numLeaves = N * static_cast<size_t>(N) * N;

  origin.reserve(numLeaves);
  if (!packed)
    data.reserve(numLeaves);

  std::vector<float> leaf(leafRes * (size_t)leafRes * leafRes, 0.f);
  for (uint32_t z = 0; z < N; ++z)
    for (uint32_t y = 0; y < N; ++y)
      for (uint32_t x = 0; x < N; ++x) {
        std::vector<range1f> leafValueRange(leafRes);
        tasking::parallel_for(leafRes, [&](uint32_t vz) {
          for (uint32_t vy = 0; vy < leafRes; ++vy)
            for (uint32_t vx = 0; vx < leafRes; ++vx) {
              float v = 0.f;
              const vec3f p((x * leafRes + vx + 0.5f) / (float)domainRes,
                  (y * leafRes + vy + 0.5f) / (float)domainRes,
                  (z * leafRes + vz + 0.5f) / (float)domainRes);
              if (turbulentTorus(p, 0.75f, 0.375f))
                v = 0.5f + 0.5f * PerlinNoise::noise(p, 12);

              const size_t idx =
                  vx * (size_t)leafRes * leafRes + vy * (size_t)leafRes + vz;
              leafValueRange[vz].extend(v);
              leaf.at(idx) = v;
            }
        });
        // reduction
        for (auto &vr : leafValueRange)
          leafValueRange[0].extend(vr);

        if (leafValueRange[0].lower != 0.f || leafValueRange[0].upper != 0.f) {
          origin.push_back(vec3i(x * leafRes, y * leafRes, z * leafRes));
          if (packed)
            std::copy(leaf.begin(), leaf.end(), std::back_inserter(dataPacked));
          else
            data.emplace_back(cpp::CopiedData(
                leaf.data(), vec3ul(leafRes, leafRes, leafRes)));
        }
      }

  if (origin.empty())
    throw std::runtime_error("vdb volume is empty.");

  volume.setParam("filter", (int)OSP_VOLUME_FILTER_TRILINEAR);
  volume.setParam("gradientFilter", (int)OSP_VOLUME_FILTER_TRILINEAR);
  volume.setParam(
      "node.level", cpp::CopiedData(std::vector<uint32_t>(origin.size(), 3)));
  volume.setParam("node.origin", cpp::CopiedData(origin));
  if (packed) {
    volume.setParam("nodesPackedDense", cpp::CopiedData(dataPacked));
    volume.setParam("node.format",
        cpp::CopiedData(std::vector<OSPVolumeFormat>(
            origin.size(), OSP_VOLUME_FORMAT_DENSE_ZYX)));
  } else
    volume.setParam("node.data", cpp::CopiedData(data));
  volume.commit();

  cpp::VolumetricModel model(volume);
  model.setParam("transferFunction", makeTransferFunction({0.f, 1.f}));
  model.setParam("densityScale", densityScale);
  model.setParam("anisotropy", anisotropy);
  model.commit();

  cpp::Group group;
  group.setParam("volume", cpp::CopiedData(model));
  group.commit();

  return group;
}

cpp::World VdbVolume::buildWorld() const
{
  auto group = buildGroup();
  cpp::Instance instance(group);
  rkcommon::math::AffineSpace3f xform(
      rkcommon::math::LinearSpace3f::scale(8.f / domainRes),
      vec3f(-4.f, 0, -4.f));
  instance.setParam("transform", xform);
  instance.commit();

  std::vector<cpp::Instance> inst;
  inst.push_back(instance);

  if (addPlane)
    inst.push_back(makeGroundPlane(instance.getBounds<box3f>()));

  cpp::Light quadLight("quad");
  quadLight.setParam("position", vec3f(-4.f, 8.0f, 4.f));
  quadLight.setParam("edge1", vec3f(0.f, 0.0f, -8.0f));
  quadLight.setParam("edge2", vec3f(2.f, 1.0f, 0.0f));
  quadLight.setParam("intensity", 5.0f);
  quadLight.setParam("color", vec3f(2.8f, 2.2f, 1.9f));
  quadLight.commit();

  std::vector<cpp::Light> lightHandles;
  lightHandles.push_back(quadLight);

  cpp::World world;
  world.setParam("instance", cpp::CopiedData(inst));
  world.setParam("light", cpp::CopiedData(lightHandles));

  return world;
}

OSP_REGISTER_TESTING_BUILDER(VdbVolume, vdb_volume);
OSP_REGISTER_TESTING_BUILDER(VdbVolume(true), vdb_volume_packed);

} // namespace testing
} // namespace ospray
