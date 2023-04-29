// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct SubdivisionCube : public detail::Builder
{
  SubdivisionCube() = default;
  ~SubdivisionCube() override = default;

  cpp::Group buildGroup() const override;
};

// Inlined definitions ////////////////////////////////////////////////////

cpp::Group SubdivisionCube::buildGroup() const
{
  // vertices of a cube
  std::vector<vec3f> vertices = {{-1.0f, -1.0f, -1.0f},
      {1.0f, -1.0f, -1.0f},
      {1.0f, -1.0f, 1.0f},
      {-1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f},
      {1.0f, 1.0f, -1.0f},
      {1.0f, 1.0f, 1.0f},
      {-1.0f, 1.0f, 1.0f}};

  // color per vertex
  std::vector<vec4f> colors = {{0.0f, 0.0f, 0.0f, 1.f},
      {1.0f, 0.0f, 0.0f, 1.f},
      {1.0f, 0.0f, 1.0f, 1.f},
      {0.0f, 0.0f, 1.0f, 1.f},
      {0.0f, 1.0f, 0.0f, 1.f},
      {1.0f, 1.0f, 0.0f, 1.f},
      {1.0f, 1.0f, 1.0f, 1.f},
      {0.0f, 1.0f, 1.0f, 1.f}};

  // number of vertex indices per face
  std::vector<unsigned int> faces = {4, 4, 4, 4, 4, 4};

  // vertex indices for all faces
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

  // vertex crease indices and weights
  std::vector<unsigned int> vertexCreaseIndices = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> vertexCreaseWeights = {
      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};

  // edge crease indices and weights
  std::vector<vec2i> edgeCreaseIndices = {{0, 1},
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
  std::vector<float> edgeCreaseWeights = {
      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};

  // global level of tessellation
  float level = 5.f;

  // create the OSPRay geometry and set all parameters
  cpp::Geometry geometry("subdivision");

  geometry.setParam("vertex.position", cpp::CopiedData(vertices));
  geometry.setParam("vertex.color", cpp::CopiedData(colors));
  geometry.setParam("face", cpp::CopiedData(faces));
  geometry.setParam("index", cpp::CopiedData(indices));
  geometry.setParam("vertexCrease.index", cpp::CopiedData(vertexCreaseIndices));
  geometry.setParam("vertexCrease.weight", cpp::CopiedData(vertexCreaseWeights));
  geometry.setParam("edgeCrease.index", cpp::CopiedData(edgeCreaseIndices));
  geometry.setParam("edgeCrease.weight", cpp::CopiedData(edgeCreaseWeights));
  geometry.setParam("level", level);
  geometry.setParam("mode", OSP_SUBDIVISION_PIN_CORNERS);

  geometry.commit();

  cpp::GeometricModel model(geometry);

  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    cpp::Material material(rendererType, "obj");
    material.commit();
    model.setParam("material", material);
  }

  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

OSP_REGISTER_TESTING_BUILDER(SubdivisionCube, subdivision_cube);

} // namespace testing
} // namespace ospray
