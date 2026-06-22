// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/multidim_index_sequence.h"

using namespace rkcommon::math;

namespace {
// List degenerate tetrahedrons
std::vector<std::vector<uint32_t>> tetraNotAllowed = {{0, 1, 2, 3},
    {4, 5, 6, 7},
    {0, 3, 4, 7},
    {1, 2, 5, 6},
    {0, 1, 4, 5},
    {2, 3, 6, 7}};

// Allowed pyramid indices order
std::vector<std::vector<uint32_t>> pyramidAllowed = {{0, 1, 2, 3, 4},
    {0, 1, 2, 3, 5},
    {0, 1, 2, 3, 6},
    {0, 1, 2, 3, 7},

    {4, 5, 6, 7, 0},
    {4, 5, 6, 7, 1},
    {4, 5, 6, 7, 2},
    {4, 5, 6, 7, 3},

    {0, 4, 7, 3, 1},
    {0, 4, 7, 3, 2},
    {0, 4, 7, 3, 5},
    {0, 4, 7, 3, 6},

    {1, 5, 6, 2, 0},
    {1, 5, 6, 2, 3},
    {1, 5, 6, 2, 4},
    {1, 5, 6, 2, 7},

    {0, 1, 5, 4, 2},
    {0, 1, 5, 4, 3},
    {0, 1, 5, 4, 6},
    {0, 1, 5, 4, 7},

    {2, 6, 7, 3, 0},
    {2, 6, 7, 3, 1},
    {2, 6, 7, 3, 4},
    {2, 6, 7, 3, 5}};

// Allowed wedge indices order
std::vector<std::vector<uint32_t>> wedgeAllowed = {{0, 1, 2, 4, 5, 6},
    {1, 2, 3, 5, 6, 7},
    {0, 2, 3, 4, 6, 7},
    {0, 1, 3, 4, 5, 7},

    {0, 4, 3, 1, 5, 2},
    {0, 4, 7, 1, 5, 6},
    {4, 7, 3, 5, 6, 2},
    {0, 7, 3, 1, 6, 2},

    {0, 1, 5, 3, 2, 6},
    {1, 5, 4, 2, 6, 7},
    {5, 4, 0, 6, 7, 3},
    {0, 1, 4, 3, 2, 7}};

// Allowed wedge+pyramid indices order
std::vector<std::vector<uint32_t>> wedgePyramidAllowed = {
    {0, 1, 2, 4, 5, 6, 0, 2, 6, 4, 3},
    {0, 1, 2, 4, 5, 6, 0, 2, 6, 4, 7},

    {1, 2, 3, 5, 6, 7, 1, 5, 7, 3, 0},
    {1, 2, 3, 5, 6, 7, 1, 5, 7, 3, 4},

    {0, 2, 3, 4, 6, 7, 0, 2, 6, 4, 1},
    {0, 2, 3, 4, 6, 7, 0, 2, 6, 4, 5},

    {0, 1, 3, 4, 5, 7, 1, 5, 7, 3, 2},
    {0, 1, 3, 4, 5, 7, 1, 5, 7, 3, 6}};

// Indices order of all allowed primitives, indexed with 8-vertices mask
std::array<std::vector<uint32_t>, 256> primsAllowed;

// Not allowed primitives indexed with 8-vertices mask
std::array<bool, 256> primsNotAllowed;

void initializeFromArray(const std::vector<std::vector<uint32_t>> &array)
{
  for (auto order : array) {
    uint32_t mask = 0;
    for (uint32_t i : order)
      mask |= (1 << i);

    std::vector<uint32_t> &ids = primsAllowed[mask];
    ids.insert(ids.begin(), order.begin(), order.end());
  }
}

void initializePrims()
{
  // Initialize degenerate tetrahedrons
  for (auto order : tetraNotAllowed) {
    uint32_t mask = 0;
    for (uint32_t i : order)
      mask |= (1 << i);

    primsNotAllowed[mask] = true;
  }

  // Initialize allowed primitives with its indices orders
  initializeFromArray(pyramidAllowed);
  initializeFromArray(wedgeAllowed);
  initializeFromArray(wedgePyramidAllowed);
}

float sphere(const vec3f &p, const vec3f &center, float radius)
{
  vec3f t = p - center;
  return sqrt(t.x * t.x + t.y * t.y + t.z * t.z) - radius;
}

float ground(const vec3f &p, const vec3f &center)
{
  vec3f t = p - center;
  return t.y + .1f * sin(10.f * t.x) + .1f * sin(10.f * t.z);
}

float sdf(const vec3f &p)
{
  return std::min({ground(p, vec3f(0.f, -.7f, 0.f)),
      sphere(p, vec3f(0.f, .2f, 0.f), .6f),
      sphere(p, vec3f(.7f, 0.f, .7f), .3f),
      sphere(p, vec3f(-.6f, -.3f, -.6f), .25f)});
}
} // namespace

namespace ospray {
namespace testing {

struct UnstructuredVolumeGen : public detail::Builder
{
  UnstructuredVolumeGen(bool transform = false, bool isosurface = false)
      : transform(transform), isosurface(isosurface)
  {}
  ~UnstructuredVolumeGen() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
  cpp::World buildWorld() const override;

 private:
  bool showCells{false};
  float densityScale{1.f};
  bool transform{false};
  bool isosurface{false};
};

// Inlined definitions ////////////////////////////////////////////////////

void UnstructuredVolumeGen::commit()
{
  Builder::commit();

  showCells = getParam<bool>("showCells", false);
  densityScale = getParam<float>("densityScale", showCells ? 100.f : 3.f);
  tfOpacityMap = getParam<std::string>(
      "tf.opacityMap", showCells ? "opaque" : "linearInv");

  addPlane = false;
}

cpp::Group UnstructuredVolumeGen::buildGroup() const
{
  // Prepare 3D grid
  vec3i dimensions = vec3i(30, 30, 30);
  index_sequence_3D numVertices(dimensions);
  std::vector<vec3f> positions;
  std::vector<float> values;
  float size = 2.f;
  positions.resize(numVertices.total_indices());
  values.resize(numVertices.total_indices());
  for (auto i : numVertices) {
    vec3f i_f = static_cast<vec3f>(i);
    vec3f p = size * (i_f / (dimensions - 1) - .5f);
    uint32_t id = numVertices.flatten(i);
    values[id] = sdf(p);
    if (transform)
      p.z *= 0.25f;
    positions[id] = p;
  }

  // Initialize primitive arrays
  initializePrims();

  // Iterate through all cubic cells in the created grid
  // and build appropriate unstructured cells
  index_sequence_3D numHex(dimensions - 1);
  std::vector<uint32_t> indices;
  std::vector<uint32_t> cells;
  std::vector<uint8_t> cellTypes;
  std::vector<float> cellValues;
  for (auto i : numHex) {
    std::vector<uint32_t> ids = {
        (uint32_t)numVertices.flatten(i + vec3i(1, 0, 0)),
        (uint32_t)numVertices.flatten(i + vec3i(0, 0, 0)),
        (uint32_t)numVertices.flatten(i + vec3i(0, 0, 1)),
        (uint32_t)numVertices.flatten(i + vec3i(1, 0, 1)),
        (uint32_t)numVertices.flatten(i + vec3i(1, 1, 0)),
        (uint32_t)numVertices.flatten(i + vec3i(0, 1, 0)),
        (uint32_t)numVertices.flatten(i + vec3i(0, 1, 1)),
        (uint32_t)numVertices.flatten(i + vec3i(1, 1, 1))};

    // Calculate mask for cubic cell consisting of 8 vertices,
    uint32_t mask = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < 8; i++) {
      uint32_t bit = 1 << i;
      if (values[ids[i]] < 0.f) {
        mask |= bit;
        count++;
      }
    }

    // Depending on number of bits set
    switch (count) {
    // Build tetrahedron
    case 4: {
      if (primsNotAllowed[mask])
        break;

      cells.push_back(indices.size());
      cellTypes.push_back(OSP_TETRAHEDRON);
      cellValues.push_back(0.8f);
      for (uint32_t i = 0; i < 8; i++)
        if ((mask >> i) & 1)
          indices.push_back(ids[i]);
    } break;

    // Build pyramid
    case 5: {
      std::vector<uint32_t> &pa = primsAllowed[mask];
      if (pa.empty())
        break;

      cells.push_back(indices.size());
      cellTypes.push_back(OSP_PYRAMID);
      for (uint32_t i : pa)
        indices.push_back(ids[i]);
      cellValues.push_back(0.6f);
    } break;

    // Build wedge
    case 6: {
      std::vector<uint32_t> &pa = primsAllowed[mask];
      if (pa.empty())
        break;

      cells.push_back(indices.size());
      cellTypes.push_back(OSP_WEDGE);
      for (uint32_t i : pa)
        indices.push_back(ids[i]);
      cellValues.push_back(0.4f);
    } break;

    // Build wedge and pyramid
    case 7: {
      std::vector<uint32_t> &pa = primsAllowed[mask];
      if (pa.empty())
        break;

      cells.push_back(indices.size());
      cells.push_back(indices.size() + 6);
      cellTypes.push_back(OSP_WEDGE);
      cellTypes.push_back(OSP_PYRAMID);
      for (uint32_t i : pa)
        indices.push_back(ids[i]);
      cellValues.push_back(0.4f);
      cellValues.push_back(0.6f);
    } break;

    // Build hexahedron
    case 8: {
      cells.push_back(indices.size());
      cellTypes.push_back(OSP_HEXAHEDRON);
      indices.insert(indices.end(), ids.begin(), ids.end());
      cellValues.push_back(0.2f);
    } break;
    }
  }

  // Create and configure volume object
  cpp::Volume volume("unstructured");
  volume.setParam("vertex.position", cpp::CopiedData(positions));
  if (showCells)
    volume.setParam("cell.data", cpp::CopiedData(cellValues));
  else
    volume.setParam("vertex.data", cpp::CopiedData(values));
  volume.setParam("index", cpp::CopiedData(indices));
  volume.setParam("cell.index", cpp::CopiedData(cells));
  volume.setParam("cell.type", cpp::CopiedData(cellTypes));
  volume.setParam("hexIterative", true);
  volume.commit();

  cpp::Group group;

  if (isosurface) {
    cpp::Geometry isoGeom("isosurface");

    std::vector<float> isovalues = {-0.2f};
    isoGeom.setParam("isovalue", cpp::CopiedData(isovalues));
    isoGeom.setParam("volume", volume);
    isoGeom.commit();

    cpp::GeometricModel isoModel(isoGeom);

    if (rendererType == "pathtracer" || rendererType == "scivis"
        || rendererType == "ao") {
      cpp::Material mat(rendererType, "obj");
      mat.setParam("kd", vec3f(0.8f));
      if (rendererType == "pathtracer" || rendererType == "scivis")
        mat.setParam("ks", vec3f(0.2f));
      mat.commit();

      isoModel.setParam("material", mat);
    }

    isoModel.commit();

    group.setParam("geometry", cpp::CopiedData(isoModel));
  } else {
    cpp::VolumetricModel model(volume);
    model.setParam("transferFunction",
        makeTransferFunction(
            showCells ? range1f{0.f, 1.f} : range1f{-.4f, -.05f}));
    model.setParam("densityScale", densityScale);
    model.commit();

    group.setParam("volume", cpp::CopiedData(model));
  }

  group.commit();

  return group;
}

cpp::World UnstructuredVolumeGen::buildWorld() const
{
  if (!transform)
    return Builder::buildWorld();

  auto group = buildGroup();
  cpp::Instance instance(group);

  AffineSpace3f xform(LinearSpace3f::rotate(vec3f(2.2f, 1.0f, -0.35f), 0.4f)
          * LinearSpace3f::scale(vec3f(1.0f, 1.0f, 4.0f)),
      vec3f(0.1f, -0.3, 2.f));
  instance.setParam("transform", xform);
  instance.commit();

  std::vector<cpp::Instance> inst;
  inst.push_back(instance);

  cpp::Light light("distant");
  light.setParam("direction", vec3f(-0.8f, -0.6f, 0.3f));
  light.setParam("color", vec3f(0.78f, 0.551f, 0.483f));
  light.setParam("intensity", 3.14f);
  light.setParam("angularDiameter", 1.f);
  light.commit();
  cpp::Light ambient("ambient");
  ambient.setParam("intensity", 0.35f);
  ambient.setParam("visible", false);
  ambient.commit();
  std::vector<cpp::Light> lights{light, ambient};

  cpp::World world;
  world.setParam("instance", cpp::CopiedData(inst));
  world.setParam("light", cpp::CopiedData(lights));

  return world;
}

OSP_REGISTER_TESTING_BUILDER(UnstructuredVolumeGen, unstructured_volume);
OSP_REGISTER_TESTING_BUILDER(
    UnstructuredVolumeGen(true), unstructured_volume_transformed);
OSP_REGISTER_TESTING_BUILDER(
    UnstructuredVolumeGen(true, true), unstructured_volume_isosurface);

} // namespace testing
} // namespace ospray
