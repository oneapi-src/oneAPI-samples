// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct UnstructuredVolumeSimple : public detail::Builder
{
  UnstructuredVolumeSimple() = default;
  ~UnstructuredVolumeSimple() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;

 private:
  bool sharedVertices{true};
  bool valuesPerCell{false};
};

// Inlined definitions ////////////////////////////////////////////////////

void UnstructuredVolumeSimple::commit()
{
  Builder::commit();

  sharedVertices = getParam<bool>("sharedVertices", true);
  valuesPerCell = getParam<bool>("cellCenteredValues", false);
}

cpp::Group UnstructuredVolumeSimple::buildGroup() const
{
  // define hexahedron parameters
  const float hSize = .4f;
  const float hX = -.5f, hY = -.5f, hZ = 0.f;

  // define wedge parameters
  const float wSize = .4f;
  const float wX = .5f, wY = -.5f, wZ = 0.f;

  // define tetrahedron parameters
  const float tSize = .4f;
  const float tX = .5f, tY = .5f, tZ = 0.f;

  // define pyramid parameters
  const float pSize = .4f;
  const float pX = -.5f, pY = .5f, pZ = 0.f;

  // define vertex positions
  std::vector<vec3f> vertices = {// hexahedron
      {-hSize + hX, -hSize + hY, hSize + hZ}, // bottom quad
      {hSize + hX, -hSize + hY, hSize + hZ},
      {hSize + hX, -hSize + hY, -hSize + hZ},
      {-hSize + hX, -hSize + hY, -hSize + hZ},
      {-hSize + hX, hSize + hY, hSize + hZ}, // top quad
      {hSize + hX, hSize + hY, hSize + hZ},
      {hSize + hX, hSize + hY, -hSize + hZ},
      {-hSize + hX, hSize + hY, -hSize + hZ},

      // wedge
      {-wSize + wX, -wSize + wY, wSize + wZ}, // botom triangle
      {wSize + wX, -wSize + wY, 0.f + wZ},
      {-wSize + wX, -wSize + wY, -wSize + wZ},
      {-wSize + wX, wSize + wY, wSize + wZ}, // top triangle
      {wSize + wX, wSize + wY, 0.f + wZ},
      {-wSize + wX, wSize + wY, -wSize + wZ},

      // tetrahedron
      {-tSize + tX, -tSize + tY, tSize + tZ},
      {tSize + tX, -tSize + tY, 0.f + tZ},
      {-tSize + tX, -tSize + tY, -tSize + tZ},
      {-tSize + tX, tSize + tY, 0.f + tZ},

      // pyramid
      {-pSize + pX, -pSize + pY, pSize + pZ},
      {pSize + pX, -pSize + pY, pSize + pZ},
      {pSize + pX, -pSize + pY, -pSize + pZ},
      {-pSize + pX, -pSize + pY, -pSize + pZ},
      {pSize + pX, pSize + pY, 0.f + pZ}};

  // define per-vertex values
  std::vector<float> vertexValues = {// hexahedron
      0.f,
      0.f,
      0.f,
      0.f,
      0.f,
      1.f,
      1.f,
      0.f,

      // wedge
      0.f,
      0.f,
      0.f,
      1.f,
      0.f,
      1.f,

      // tetrahedron
      1.f,
      0.f,
      1.f,
      0.f,

      // pyramid
      0.f,
      1.f,
      1.f,
      0.f,
      0.f};

  // define vertex indices for both shared and separate case
  std::vector<uint32_t> indicesSharedVert = {// hexahedron
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,

      // wedge
      1,
      9,
      2,
      5,
      12,
      6,

      // tetrahedron
      5,
      12,
      6,
      17,

      // pyramid
      4,
      5,
      6,
      7,
      17};
  std::vector<uint32_t> indicesSeparateVert = {// hexahedron
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,

      // wedge
      8,
      9,
      10,
      11,
      12,
      13,

      // tetrahedron
      14,
      15,
      16,
      17,

      // pyramid
      18,
      19,
      20,
      21,
      22};
  std::vector<uint32_t> &indices =
      sharedVertices ? indicesSharedVert : indicesSeparateVert;

  // define cell offsets in indices array
  std::vector<uint32_t> cells = {0, 8, 14, 18};

  // define cell types
  std::vector<uint8_t> cellTypes = {
      OSP_HEXAHEDRON, OSP_WEDGE, OSP_TETRAHEDRON, OSP_PYRAMID};

  // define per-cell values
  std::vector<float> cellValues = {0.1f, .3f, .7f, 1.f};

  cpp::Volume volume("unstructured");

  // set data objects for volume object
  volume.setParam("vertex.position", cpp::CopiedData(vertices));

  if (valuesPerCell)
    volume.setParam("cell.data", cpp::CopiedData(cellValues));
  else
    volume.setParam("vertex.data", cpp::CopiedData(vertexValues));

  volume.setParam("index", cpp::CopiedData(indices));
  volume.setParam("cell.index", cpp::CopiedData(cells));
  volume.setParam("cell.type", cpp::CopiedData(cellTypes));

  volume.commit();

  cpp::VolumetricModel model(volume);
  model.setParam("transferFunction", makeTransferFunction({0.f, 1.f}));
  model.commit();

  cpp::Group group;

  group.setParam("volume", cpp::CopiedData(model));
  group.commit();

  return group;
}

OSP_REGISTER_TESTING_BUILDER(UnstructuredVolumeSimple, unstructured_volume_simple);

} // namespace testing
} // namespace ospray
