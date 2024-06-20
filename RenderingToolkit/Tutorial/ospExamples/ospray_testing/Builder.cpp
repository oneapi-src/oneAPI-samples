// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"

namespace ospray {
namespace testing {
namespace detail {

std::unique_ptr<Builder::BuilderFactory> Builder::factory;

void Builder::commit()
{
  rendererType = getParam<std::string>("rendererType", "scivis");
  tfColorMap = getParam<std::string>("tf.colorMap", "jet");
  tfOpacityMap = getParam<std::string>("tf.opacityMap", "linear");
  randomSeed = getParam<unsigned int>("randomSeed", 0);
}

cpp::World Builder::buildWorld() const
{
  return buildWorld({});
}

cpp::World Builder::buildWorld(
    const std::vector<cpp::Instance> &instances) const
{
  cpp::World world;

  auto group = buildGroup();

  cpp::Instance instance(group);
  instance.commit();

  std::vector<cpp::Instance> inst = instances;
  inst.push_back(instance);

  if (addPlane)
    inst.push_back(makeGroundPlane(group.getBounds<box3f>()));

  world.setParam("instance", cpp::CopiedData(inst));

  cpp::Light light("ambient");
  light.setParam("visible", false);
  light.commit();

  world.setParam("light", cpp::CopiedData(light));

  return world;
}

cpp::TransferFunction Builder::makeTransferFunction(
    const range1f &valueRange) const
{
  cpp::TransferFunction transferFunction("piecewiseLinear");

  std::vector<vec3f> colors;
  std::vector<float> opacities;

  if (tfColorMap == "jet") {
    colors.emplace_back(0, 0, 0.562493);
    colors.emplace_back(0, 0, 1);
    colors.emplace_back(0, 1, 1);
    colors.emplace_back(0.500008, 1, 0.500008);
    colors.emplace_back(1, 1, 0);
    colors.emplace_back(1, 0, 0);
    colors.emplace_back(0.500008, 0, 0);
  } else if (tfColorMap == "rgb") {
    colors.emplace_back(0, 0, 1);
    colors.emplace_back(0, 1, 0);
    colors.emplace_back(1, 0, 0);
  } else {
    colors.emplace_back(0.f, 0.f, 0.f);
    colors.emplace_back(1.f, 1.f, 1.f);
  }

  if (tfOpacityMap == "linear") {
    opacities.emplace_back(0.f);
    opacities.emplace_back(1.f);
  } else if (tfOpacityMap == "linearInv") {
    opacities.emplace_back(1.f);
    opacities.emplace_back(0.f);
  } else if (tfOpacityMap == "opaque") {
    opacities.emplace_back(1.f);
  }

  transferFunction.setParam("color", cpp::CopiedData(colors));
  transferFunction.setParam("opacity", cpp::CopiedData(opacities));
  transferFunction.setParam("value", valueRange);
  transferFunction.commit();

  return transferFunction;
}

cpp::Instance Builder::makeGroundPlane(const box3f &bounds) const
{
  auto planeExtent = 0.8f * length(bounds.center() - bounds.lower);

  cpp::Geometry planeGeometry("mesh");

  std::vector<vec3f> v_position;
  std::vector<vec3f> v_normal;
  std::vector<vec4f> v_color;
  std::vector<vec4ui> indices;

  unsigned int startingIndex = 0;

  const vec3f up = vec3f{0.f, 1.f, 0.f};
  const vec4f gray = vec4f{0.9f, 0.9f, 0.9f, 0.75f};

  v_position.emplace_back(-planeExtent, -1.f, -planeExtent);
  v_position.emplace_back(planeExtent, -1.f, -planeExtent);
  v_position.emplace_back(planeExtent, -1.f, planeExtent);
  v_position.emplace_back(-planeExtent, -1.f, planeExtent);

  v_normal.push_back(up);
  v_normal.push_back(up);
  v_normal.push_back(up);
  v_normal.push_back(up);

  v_color.push_back(gray);
  v_color.push_back(gray);
  v_color.push_back(gray);
  v_color.push_back(gray);

  indices.emplace_back(
      startingIndex, startingIndex + 1, startingIndex + 2, startingIndex + 3);

  // stripes on ground plane
  const float stripeWidth = 0.025f;
  const float paddedExtent = planeExtent + stripeWidth;
  const size_t numStripes = 10;

  const vec4f stripeColor = vec4f{1.0f, 0.1f, 0.1f, 1.f};

  for (size_t i = 0; i < numStripes; i++) {
    // the center coordinate of the stripe, either in the x or z
    // direction
    const float coord =
        -planeExtent + float(i) / float(numStripes - 1) * 2.f * planeExtent;

    // offset the stripes by an epsilon above the ground plane
    const float yLevel = -1.f + 1e-3f;

    // x-direction stripes
    startingIndex = v_position.size();

    v_position.emplace_back(-paddedExtent, yLevel, coord - stripeWidth);
    v_position.emplace_back(paddedExtent, yLevel, coord - stripeWidth);
    v_position.emplace_back(paddedExtent, yLevel, coord + stripeWidth);
    v_position.emplace_back(-paddedExtent, yLevel, coord + stripeWidth);

    v_normal.push_back(up);
    v_normal.push_back(up);
    v_normal.push_back(up);
    v_normal.push_back(up);

    v_color.push_back(stripeColor);
    v_color.push_back(stripeColor);
    v_color.push_back(stripeColor);
    v_color.push_back(stripeColor);

    indices.emplace_back(
        startingIndex, startingIndex + 1, startingIndex + 2, startingIndex + 3);

    // z-direction stripes
    startingIndex = v_position.size();

    // offset another epsilon to avoid z-figthing for primID AOV
    const float yLevel2 = yLevel + 1e-4f;

    v_position.emplace_back(coord - stripeWidth, yLevel2, -paddedExtent);
    v_position.emplace_back(coord + stripeWidth, yLevel2, -paddedExtent);
    v_position.emplace_back(coord + stripeWidth, yLevel2, paddedExtent);
    v_position.emplace_back(coord - stripeWidth, yLevel2, paddedExtent);

    v_normal.push_back(up);
    v_normal.push_back(up);
    v_normal.push_back(up);
    v_normal.push_back(up);

    v_color.push_back(stripeColor);
    v_color.push_back(stripeColor);
    v_color.push_back(stripeColor);
    v_color.push_back(stripeColor);

    indices.emplace_back(
        startingIndex, startingIndex + 1, startingIndex + 2, startingIndex + 3);
  }

  planeGeometry.setParam("vertex.position", cpp::CopiedData(v_position));
  planeGeometry.setParam("vertex.normal", cpp::CopiedData(v_normal));
  planeGeometry.setParam("vertex.color", cpp::CopiedData(v_color));
  planeGeometry.setParam("index", cpp::CopiedData(indices));

  planeGeometry.commit();

  cpp::GeometricModel plane(planeGeometry);

  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    cpp::Material material(rendererType, "obj");
    material.commit();
    plane.setParam("material", material);
  }

  plane.commit();

  cpp::Group planeGroup;
  planeGroup.setParam("geometry", cpp::CopiedData(plane));
  planeGroup.commit();

  cpp::Instance planeInst(planeGroup);
  planeInst.commit();
  return planeInst;
}

void Builder::registerBuilder(const std::string &name, BuilderFcn fcn)
{
  if (factory.get() == nullptr)
    factory = make_unique<BuilderFactory>();

  (*factory)[name] = fcn;
}

Builder *Builder::createBuilder(const std::string &name)
{
  if (factory.get() == nullptr)
    return nullptr;
  else {
    auto &fcn = (*factory)[name];
    return fcn();
  }
}

} // namespace detail
} // namespace testing
} // namespace ospray
