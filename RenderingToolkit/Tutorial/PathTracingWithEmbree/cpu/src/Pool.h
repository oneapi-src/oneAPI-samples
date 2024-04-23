#pragma once
#ifndef FILE_POOL_SEEN
#define FILE_POOL_SEEN

#include <embree4/rtcore.h>

#include <vector>

#include "Geometry.h"
#include "Lights.h"
#include "Materials.h"
#include "definitions.h"

class Pool : public Geometry {
 public:
  Pool(const RTCScene& scene, const RTCDevice& device,
       std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
       std::map<unsigned int, size_t>& mapGeomToLightIdx,
       std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
       unsigned int width, unsigned int height);
  ~Pool();

 private:
  void add_geometry(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);
  unsigned int addPool(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);
  unsigned int addWater(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);

  void setup_camera_and_lights(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
      std::map<unsigned int, size_t>& mapGeomToLightIdx,
      std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
      unsigned int width, unsigned int height);
  void clean_geometry();
  void cleanPool();
  void cleanWater();

  static const std::vector<Vertex> m_poolBoxVertices;
  static const std::vector<Quad> m_poolBoxIndices;
  static const std::vector<Vec3fa> m_poolBoxColors;
  static const std::vector<MaterialType> m_poolBoxMats;
  std::vector<MaterialType> m_waterMats;

  /* Added for pathtracer */
  Vec3fa* m_pool_face_colors = nullptr;
  Vec3fa* m_pool_vertex_colors = nullptr;
  Vec3fa* m_water_face_colors = nullptr;
  Vec3fa* m_water_vertex_colors = nullptr;
};

// mesh data
const std::vector<Vertex> Pool::m_poolBoxVertices = {
    // South Wall
    {1.00f, -1.00f, -10.00f, 0.0f},
    {-1.00f, -1.00f, -10.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    // North Wall
    {1.00f, 1.00f, -10.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, -10.00f, 0.0f},

    // Foundation
    {1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    // East Wall
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, -10.00f, 0.0f},
    {-1.00f, 1.00f, -10.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    // West Wall
    {1.00f, -1.00f, -10.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, -10.00f, 0.0f},

    // Unseen wall
    {1.00f, -1.00f, -10.00f, 0.0f},
    {-1.00f, -1.00f, -10.00f, 0.0f},
    {-1.00f, 1.00f, -10.00f, 0.0f},
    {1.00f, 1.00f, -10.00f, 0.0f}

};

const std::vector<Quad> Pool::m_poolBoxIndices = {
    {0, 1, 2, 3},      // Floor
    {4, 5, 6, 7},      // Ceiling
    {8, 9, 10, 11},    // Backwall
    {12, 13, 14, 15},  // RightWall
    {16, 17, 18, 19},  // LeftWall
    {20, 21, 22, 23}   // Unseen wall
};

const std::vector<Vec3fa> Pool::m_poolBoxColors = {
    // South Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // North Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // Foundation "Egyptian Blue"
    {0.078f, 0.204f, 0.643f},
    {0.078f, 0.204f, 0.643f},
    {0.078f, 0.204f, 0.643f},
    {0.078f, 0.204f, 0.643f},
    // East Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // West Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // West Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},

};

const std::vector<MaterialType> Pool::m_poolBoxMats = {
    // Floor
    MaterialType::MATERIAL_MATTE,
    // Ceiling
    MaterialType::MATERIAL_MATTE,
    // Backwall
    MaterialType::MATERIAL_MATTE,
    // RightWall
    MaterialType::MATERIAL_MATTE,
    // LeftWall
    MaterialType::MATERIAL_MATTE,
    // Unseen Wall
    MaterialType::MATERIAL_MATTE

};

inline float square(const float& in) { return in * in; }

Pool::Pool(const RTCScene& scene, const RTCDevice& device,
           std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
           std::map<unsigned int, size_t>& mapGeomToLightIdx,
           std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
           unsigned int width, unsigned int height) {
  add_geometry(scene, device, mapGeomToPrim);
  setup_camera_and_lights(scene, device, mapGeomToPrim, mapGeomToLightIdx,
                          lights, camera, width, height);
}

void Pool::add_geometry(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  addPool(scene, device, mapGeomToPrim);
  addWater(scene, device, mapGeomToPrim);
}

unsigned int Pool::addPool(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  /* create a mesh for all the quads in the pool Box scene */
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD);
  m_pool_face_colors =
      (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * m_poolBoxIndices.size(), 16);
  m_pool_vertex_colors =
      (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * m_poolBoxVertices.size(), 16);
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
      m_poolBoxVertices.size());
  for (auto i = 0; i < m_poolBoxVertices.size(); i++) {
    vertices[i] = m_poolBoxVertices[i];
    m_pool_vertex_colors[i] = m_poolBoxColors[i];
  }

  /* set quads */
  Quad* quads = (Quad*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0,
                                               RTC_FORMAT_UINT4, sizeof(Quad),
                                               m_poolBoxIndices.size());

  for (auto i = 0; i < m_poolBoxIndices.size(); i++) {
    quads[i] = m_poolBoxIndices[i];
    m_pool_face_colors[i] = m_poolBoxColors[i * 4];
  }

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, m_pool_vertex_colors, 0,
                             sizeof(Vec3fa), m_poolBoxVertices.size());

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = m_poolBoxMats;
  mpTable.primColorTable = m_pool_face_colors;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));

  return geomID;
}

unsigned int Pool::addWater(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  /* create a triangulated water with 12 triangles and 8 vertices */
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

  size_t latticeWidth = 200;
  int numTriangles = 2 * (latticeWidth - 1) * (latticeWidth - 1);
  /* create face and vertex color arrays */
  m_water_face_colors =
      (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * numTriangles, 16);
  m_water_vertex_colors =
      (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * latticeWidth * latticeWidth, 16);

  /* set vertices and vertex colors */
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
      latticeWidth * latticeWidth);

  // Bubbles color
  Vec3fa waterColor = Vec3fa(0.906f, 0.9961f, 1.0f);

  // Top of the water will be z = FarPlaneZ - waterdepth... modulated up or down
  // by some sin waves
  float waterDepth = 1.4f;
  float horizScale = 0.03f;
  // float vertScale = 0.03f;
  int horizPeriods = 4;
  // int vertPeriods = 6;

  for (int j = 0; j < latticeWidth; j++) {
    for (int i = 0; i < latticeWidth; i++) {
      m_water_vertex_colors[j * latticeWidth + i] = waterColor;
      // Assigning vertices from bottom right to top left
      vertices[j * latticeWidth + i].x =
          1.0f - (float(i) / float(latticeWidth - 1)) * 2.0f;
      vertices[j * latticeWidth + i].y =
          1.0f - (float(j) / float(latticeWidth - 1)) * 2.0f;
      vertices[j * latticeWidth + i].z =
          1.0f - waterDepth +
          horizScale * waterDepth *
              sinf((horizPeriods *
                    (square(float(i) / (latticeWidth - 1)) +
                     square(float(j) / (latticeWidth - 1))) *
                    2.f * float(M_PI)));
      //+  vertScale * waterDepth * sinf(vertPeriods*(float(j) /
      // float(latticeWidth - 1)) * 2.f * float(M_PI));
    }
  }

  /* set triangles and face colors */

  int tri = 0;
  Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
      numTriangles);

  for (int j = 0; j < latticeWidth - 1; j++) {
    for (int i = 0; i < latticeWidth - 1; i++) {
      m_water_face_colors[tri] = waterColor;
      m_waterMats.push_back(MaterialType::MATERIAL_WATER);
      triangles[tri].v0 = j * latticeWidth + i;
      triangles[tri].v1 = j * latticeWidth + (i + 1);
      triangles[tri].v2 = (j + 1) * latticeWidth + i;
      tri++;

      m_water_face_colors[tri] = waterColor;
      m_waterMats.push_back(MaterialType::MATERIAL_WATER);
      triangles[tri].v0 = j * latticeWidth + (i + 1);
      triangles[tri].v1 = (j + 1) * latticeWidth + i;
      triangles[tri].v2 = (j + 1) * latticeWidth + (i + 1);
      tri++;
    }
  }

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, m_water_vertex_colors, 0,
                             sizeof(Vec3fa), latticeWidth * latticeWidth);

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = m_waterMats;
  mpTable.primColorTable = m_water_face_colors;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));

  return geomID;
}

void Pool::setup_camera_and_lights(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
    std::map<unsigned int, size_t>& mapGeomToLightIdx,
    std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
    unsigned int width, unsigned int height) {
  /* A camera position that connects the field of vision angle of the camera to
   * the bounds of the pool box */
  float fov = 30.0f;
  float fovrad = fov * M_PI / 180.0f;
  float half_fovrad = fovrad * 0.5f;
  // Careful to not move the camera outside the box! :-D
  camera = positionCamera(Vec3fa(0.0, 0.0, -1.0f - 1.f / tanf(half_fovrad)),
                          Vec3fa(0, 0, 0), Vec3fa(0, 1, 0), fov, width, height);

  /* The magnitude of the light can be tricky. Lights such as the point light
   * fall off at the inverse square of the distance. When designing a sandbox
   * renderer, you may need to scale your light up or down to see your scene. */
  Vec3fa spotPow = 20.f * Vec3fa(0.9922, 0.9843, 0.8275);

  /* A somewhat central position for the point light within the box. This is
   * similar to the position for the interactive pathtracer program shipped with
   * Intel Embree */

  /* Here we have a light as a disc geometry */
  Vec3fa spotPos(0.f, 0.f, -5.0f);
  Vec3fa spotDir(0.f, 0.f, 1.f);

  float spotCosAngleMax = cosf(80.f * M_PI / 180.f);
  float spotCosAngleScale = 50.f;
  float spotRadius = 0.4f;
  lights.push_back(std::make_shared<SpotLight>(spotPos, spotDir, spotPow,
                                               spotCosAngleMax,
                                               spotCosAngleScale, spotRadius));
  /* Add geometry if you want it! */
  if (spotRadius > 0.f) {
    std::shared_ptr<SpotLight> pSpotLight =
        std::dynamic_pointer_cast<SpotLight>(lights.back());
    unsigned int geomID =
        pSpotLight->add_geometry(scene, device, mapGeomToPrim);
    mapGeomToLightIdx.insert(std::make_pair(geomID, lights.size() - 1));
  }
}

void Pool::cleanPool() {
  if (m_pool_face_colors) alignedFree(m_pool_face_colors);
  m_pool_face_colors = nullptr;
  if (m_pool_vertex_colors) alignedFree(m_pool_vertex_colors);
  m_pool_vertex_colors = nullptr;
}

void Pool::cleanWater() {
  if (m_water_face_colors) alignedFree(m_water_face_colors);
  m_water_face_colors = nullptr;
  if (m_water_vertex_colors) alignedFree(m_water_vertex_colors);
  m_water_vertex_colors = nullptr;
}

void Pool::clean_geometry() {
  cleanPool();
  cleanWater();
}

Pool::~Pool() { clean_geometry(); }

#endif /* !FILE_POOL_SEEN */
