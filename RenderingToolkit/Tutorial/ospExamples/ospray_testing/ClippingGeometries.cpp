// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct ClippingGeometries : public detail::Builder
{
  ClippingGeometries(
      const std::string &geometryType, const std::string &geometrySubType = "");
  ~ClippingGeometries() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;

 private:
  vec3i dimensions{19};
  std::string geomType;
  std::string geomSubType;
};

// Inlined definitions ////////////////////////////////////////////////////

ClippingGeometries::ClippingGeometries(
    const std::string &geometryType, const std::string &geometrySubType)
    : geomType(geometryType), geomSubType(geometrySubType)
{}

void ClippingGeometries::commit()
{
  Builder::commit();
  dimensions = getParam<vec3i>("dimensions", dimensions);
  geomType = getParam<std::string>("geometryType", geomType);
  geomSubType = getParam<std::string>("geometrySubType", geomSubType);
  addPlane = false;
}

cpp::Group ClippingGeometries::buildGroup() const
{
  cpp::Geometry spheresGeometry("sphere");

  index_sequence_3D numSpheres(dimensions);
  std::vector<vec3f> positions;

  auto dim = reduce_max(dimensions) - 1;
  float size = 2.f;
  for (auto i : numSpheres) {
    vec3f i_f = static_cast<vec3f>(i);
    vec3f p = size * (i_f / dim - .5f);
    positions.emplace_back(p);
  }

  spheresGeometry.setParam("sphere.position", cpp::CopiedData(positions));
  spheresGeometry.setParam("radius", size / dim / 2.f);
  spheresGeometry.commit();

  cpp::GeometricModel model(spheresGeometry);
  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    cpp::Material material(rendererType, "obj");
    material.setParam("kd", vec3f(.1f, .4f, .8f));
    material.commit();
    model.setParam("material", material);
  }
  model.commit();

  cpp::Group group;
  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

cpp::World ClippingGeometries::buildWorld() const
{
  // Create selected geometry
  vec3f cPos;
  cpp::Geometry geometry(geomType);
  if (geomType == "curve") {
    static std::vector<vec4f> vertices = {{-1.f, +0.f, -1.f, 0.2f},
        {+0.f, -1.f, +0.f, 0.2f},
        {+1.f, +0.f, +1.f, 0.2f},
        {-1.f, +0.f, +1.f, 0.2f},
        {+0.f, +1.f, +0.f, 0.3f},
        {+1.f, +0.f, -1.f, 0.2f},
        {-1.f, +0.f, -1.f, 0.2f},
        {+0.f, -1.f, +0.f, 0.2f},
        {+1.f, +0.f, +1.f, 0.2f}};
    if (geomSubType == "bspline") {
      geometry.setParam("vertex.position_radius", cpp::CopiedData(vertices));
      geometry.setParam("basis", OSP_BSPLINE);
      cPos = vec3f(-.1f, 0.f, -.1f);
    } else if (geomSubType == "linear") {
      geometry.setParam("vertex.position",
          cpp::CopiedData((vec3f *)vertices.data(), vertices.size(), sizeof(vec4f)));
      geometry.setParam("radius", 0.2f);
      geometry.setParam("basis", OSP_LINEAR);
      cPos = vec3f(-.2f, 0.f, -.2f);
    }
    std::vector<unsigned int> indices = {0, 1, 2, 3, 4, 5};
    geometry.setParam("index", cpp::CopiedData(indices));
    geometry.setParam("type", OSP_ROUND);
    geometry.commit();
  } else if (geomType == "subdivision") {
    const float size = .9f;
    std::vector<vec3f> vertices = {{-size, -size, -size},
        {+size, -size, -size},
        {+size, -size, +size},
        {-size, -size, +size},
        {-size, +size, -size},
        {+size, +size, -size},
        {+size, +size, +size},
        {-size, +size, +size}};
    geometry.setParam("vertex.position", cpp::CopiedData(vertices));
    std::vector<unsigned int> faces = {4, 4, 4, 4, 4, 4};
    geometry.setParam("face", cpp::CopiedData(faces));
    std::vector<unsigned int> indices = {
        0,
        4,
        5,
        1,
        1,
        5,
        6,
        2,
        2,
        6,
        7,
        3,
        0,
        3,
        7,
        4,
        4,
        7,
        6,
        5,
        0,
        1,
        2,
        3,
    };
    geometry.setParam("index", cpp::CopiedData(indices));
    std::vector<unsigned int> vertexCreaseIndices = {0, 1, 2, 3, 4, 5, 6, 7};
    geometry.setParam("vertexCrease.index", cpp::CopiedData(vertexCreaseIndices));
    std::vector<float> vertexCreaseWeights = {
        2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
    geometry.setParam("vertexCrease.weight", cpp::CopiedData(vertexCreaseWeights));
    std::vector<vec2ui> edgeCreaseIndices = {{0, 1},
        {1, 2},
        {2, 3},
        {3, 0},
        {4, 5},
        {5, 6},
        {6, 7},
        {7, 4},
        {0, 4},
        {1, 5},
        {2, 6},
        {3, 7}};
    geometry.setParam("edgeCrease.index", cpp::CopiedData(edgeCreaseIndices));
    std::vector<float> edgeCreaseWeights = {
        2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
    geometry.setParam("edgeCrease.weight", cpp::CopiedData(edgeCreaseWeights));
    geometry.setParam("level", 10.f);
    geometry.commit();
    cPos = vec3f(-.3f, .3f, -.3f);
  } else if (geomType == "mesh") {
    const vec3f pos = vec3f(0.f);
    const vec3f size = vec3f(1.1f);
    std::vector<vec3f> vertices = {{pos.x, pos.y - size.y, pos.z},
        {pos.x - size.x, pos.y, pos.z - size.z},
        {pos.x + size.x, pos.y, pos.z - size.z},
        {pos.x + size.x, pos.y, pos.z + size.z},
        {pos.x - size.x, pos.y, pos.z + size.z},
        {pos.x, pos.y + size.y, pos.z}};
    geometry.setParam("vertex.position", cpp::CopiedData(vertices));
    std::vector<vec3ui> indices = {{0, 1, 2},
        {0, 2, 3},
        {0, 3, 4},
        {0, 4, 1},
        {5, 2, 1},
        {5, 3, 2},
        {5, 4, 3},
        {5, 1, 4}};
    geometry.setParam("index", cpp::CopiedData(indices));
    geometry.commit();
    cPos = vec3f(-.3f, 0.f, -.3f);
  } else if (geomType == "plane") {
    std::vector<vec4f> coefficients = {vec4f(1.0f, 1.0f, 1.0f, 0.0f)};
    geometry.setParam("plane.coefficients", cpp::CopiedData(coefficients));
    geometry.commit();
    cPos = vec3f(-.3f, .2f, -.3f);
  } else if (geomType == "box") {
    std::vector<box3f> boxes = {
        box3f(vec3f(-.9f, -.9f, -.9f), vec3f(.9f, .9f, .9f))};
    geometry.setParam("box", cpp::CopiedData(boxes));
    geometry.commit();
    cPos = vec3f(-.3f, .3f, -.3f);
  } else if (geomType == "sphere") {
    std::vector<vec3f> position = {vec3f(0.f, 0.f, 0.f)};
    geometry.setParam("sphere.position", cpp::CopiedData(position));
    geometry.setParam("radius", 1.f);
    geometry.commit();
    cPos = vec3f(-.3f, .2f, -.3f);
  } else
    return Builder::buildWorld();

  std::vector<cpp::Instance> inst;

  // Create clipping instance with model that has inverted normals
  {
    cpp::GeometricModel model(geometry);
    model.setParam("invertNormals", true);
    model.commit();

    cpp::Group group;
    group.setParam("clippingGeometry", cpp::CopiedData(model));
    group.commit();

    cpp::Instance instance(group);
    instance.commit();
    inst.push_back(instance);
  }

  // Create second clipping instance but keep original normals
  {
    cpp::GeometricModel model(geometry);
    model.setParam("invertNormals", false);
    model.commit();

    cpp::Group group;
    group.setParam("clippingGeometry", cpp::CopiedData(model));
    group.commit();

    cpp::Instance instance(group);
    instance.setParam("transform", affine3f::translate(cPos));
    instance.commit();

    inst.push_back(instance);
  }

  return Builder::buildWorld(inst);
}

OSP_REGISTER_TESTING_BUILDER(ClippingGeometries("sphere"), clip_with_spheres);
OSP_REGISTER_TESTING_BUILDER(ClippingGeometries("box"), clip_with_boxes);
OSP_REGISTER_TESTING_BUILDER(ClippingGeometries("plane"), clip_with_planes);
OSP_REGISTER_TESTING_BUILDER(ClippingGeometries("mesh"), clip_with_meshes);
OSP_REGISTER_TESTING_BUILDER(
    ClippingGeometries("subdivision"), clip_with_subdivisions);
OSP_REGISTER_TESTING_BUILDER(
    ClippingGeometries("curve", "linear"), clip_with_linear_curves);
OSP_REGISTER_TESTING_BUILDER(
    ClippingGeometries("curve", "bspline"), clip_with_bspline_curves);
} // namespace testing
} // namespace ospray
