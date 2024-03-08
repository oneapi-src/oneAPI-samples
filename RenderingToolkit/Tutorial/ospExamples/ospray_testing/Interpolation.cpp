// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
// stl
#include <random>

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct Interpolation : public detail::Builder
{
  Interpolation(
      bool subd = false, unsigned int attrib = 0, unsigned int interp = 0)
      : useSubd(subd), attribute(attrib), interpolation(interp)
  {}
  ~Interpolation() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;

  // TODO: support triangles
  bool useSubd{false}; // otherwise Mesh
  unsigned int attribute; // color, texcoord, normal
  unsigned int interpolation; // face varying, vertex varying, uniform, constant
};

// quad mesh data
static std::vector<vec3f> vertices = {
    // Left
    {0.f, 0.f, 0.f},
    {1.f, 0.f, 0.f},
    {1.f, 1.f, 0.f},
    {0.f, 1.f, 0.f},
    // Right
    {2.f, 0.f, 0.f},
    {2.f, 1.f, 0.f}};

static std::vector<vec4ui> quad_indices = {
    {0, 1, 2, 3}, // Left
    {1, 4, 5, 2} // Right
};

static std::vector<unsigned int> subd_indices = {0, 1, 2, 3, 1, 4, 5, 2};

static std::vector<vec2f> texcoords = {{0.0f, 0.0f},
    {0.1f, 0.0f},
    {0.2f, 0.0f},
    {0.3f, 0.0f},
    {0.4f, 0.0f},
    {0.5f, 0.0f},
    {0.6f, 0.0f},
    {0.7f, 0.0f}};

static std::vector<vec4f> colors = {{1.f, 0.f, 0.f, 1.f},
    {0.f, 1.f, 0.f, 1.f},
    {0.f, 0.f, 1.f, 1.f},
    {1.f, 1.f, 0.f, 1.f},
    {1.f, 0.f, 1.f, 1.f},
    {1.f, 0.f, 1.f, 1.f}};

static std::vector<vec4f> colors_fv = {{1.f, 0.f, 0.f, 1.f},
    {0.f, 1.f, 0.f, 1.f},
    {0.f, 0.f, 1.f, 1.f},
    {1.f, 1.f, 0.f, 1.f},
    {1.f, 0.f, 1.f, 1.f},
    {1.f, 0.f, 1.f, 1.f},
    {1.f, 0.f, 1.f, 1.f},
    {1.f, 0.f, 1.f, 1.f}};

static std::vector<vec3f> normals = {{1.f, 0.f, 0.f},
    {0.f, 1.f, 0.f},
    {0.f, 0.f, 1.f},
    {1.f, 1.f, 0.f},
    {1.f, 0.f, 1.f},
    {1.f, 0.f, 1.f},
    {1.f, 0.f, 1.f},
    {1.f, 0.f, 1.f}};

// number of vertex indices per face
static std::vector<unsigned int> faces = {4, 4};

// Inlined definitions ////////////////////////////////////////////////////

void Interpolation::commit()
{
  Builder::commit();
  useSubd = getParam<bool>("useSubd", useSubd);
  attribute = getParam<unsigned int>("attribute", attribute);
  interpolation = getParam<unsigned int>("interpolation", interpolation);
  addPlane = false;
}

cpp::Group Interpolation::buildGroup() const
{
  cpp::Geometry mesh;
  std::vector<vec4f> colors_mod = colors;
  if (interpolation == 0)
    colors_mod = colors_fv;
  else if (interpolation == 2)
    colors_mod.resize(2);
  else if (interpolation == 3)
    colors_mod.resize(1);

  cpp::Texture tex("texture2d");
  tex.setParam("format", OSP_TEXTURE_RGBA32F);
  tex.setParam(
      "data", cpp::CopiedData(colors_mod.data(), vec2ul(colors_mod.size(), 1)));
  tex.commit();
  cpp::Material mat(rendererType, "obj");
  if (attribute == 1) // texture
    mat.setParam("map_kd", tex);
  mat.commit();
  if (useSubd) {
    mesh = cpp::Geometry("subdivision");
    mesh.setParam("vertex.position", cpp::CopiedData(vertices));
    mesh.setParam("face", cpp::CopiedData(faces));
    mesh.setParam("level", 1.0f); // global level
    mesh.setParam("index", cpp::CopiedData(subd_indices));
    mesh.setParam("mode", OSP_SUBDIVISION_PIN_CORNERS);
  } else {
    mesh = cpp::Geometry("mesh");
    mesh.setParam("vertex.position", cpp::CopiedData(vertices));
    mesh.setParam("index", cpp::CopiedData(quad_indices));
  }
  if (interpolation == 0) {
    if (attribute == 1)
      mesh.setParam("texcoord", cpp::CopiedData(texcoords));
    else if (attribute == 2)
      mesh.setParam("normal", cpp::CopiedData(normals));
    else
      mesh.setParam("color", cpp::CopiedData(colors_mod));
  } else if (interpolation == 1) {
    if (attribute == 1)
      mesh.setParam("vertex.texcoord", cpp::CopiedData(texcoords));
    else if (attribute == 2)
      mesh.setParam("vertex.normal", cpp::CopiedData(normals));
    else
      mesh.setParam("vertex.color", cpp::CopiedData(colors_mod));
  }
  mesh.commit();

  cpp::GeometricModel model(mesh);

  // create and setup a material
  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    model.setParam("material", mat);
  }

  if ((attribute == 0) && interpolation == 2)
    model.setParam("color", cpp::CopiedData(colors_mod));
  else if ((attribute == 0) && interpolation == 3)
    model.setParam("color", colors_mod[0]);

  // Put the mesh and material into a model
  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

cpp::World Interpolation::buildWorld() const
{
  auto world = Builder::buildWorld();

  cpp::Light light("distant");
  light.setParam("color", vec3f(0.78f, 0.551f, 0.183f));
  light.setParam("intensity", 3.14f);
  light.setParam("direction", vec3f(-0.8f, -0.6f, 0.3f));
  light.commit();
  cpp::Light ambient("ambient");
  ambient.setParam("intensity", 0.35f);
  ambient.setParam("visible", false);
  ambient.commit();
  std::vector<cpp::Light> lights{light, ambient};
  world.setParam("light", cpp::CopiedData(lights));

  return world;
}

OSP_REGISTER_TESTING_BUILDER(Interpolation, interpolation);

} // namespace testing
} // namespace ospray
