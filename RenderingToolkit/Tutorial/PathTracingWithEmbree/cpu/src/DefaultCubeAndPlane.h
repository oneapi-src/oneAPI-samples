#pragma once
#ifndef FILE_DEFAULTCUBEANDPLANE_SEEN
#define FILE_DEFAULTCUBEANDPLANE_SEEN

#include <embree4/rtcore.h>

#include <vector>

#include "Lights.h"
#include "Materials.h"
#include "definitions.h"

class CubeAndPlane : public Geometry {
 public:
  CubeAndPlane(const RTCScene& scene, const RTCDevice& device,
               std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
               std::map<unsigned int, size_t>& mapGeomToLightIdx,
               std::vector<std::shared_ptr<Light>>& lights,
               AffineSpace3fa& camera, unsigned int width, unsigned int height);
  ~CubeAndPlane();

 private:
  void add_geometry(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);
  void setup_camera_and_lights(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
      std::map<unsigned int, size_t>& mapGeomToLightIdx,
      std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
      unsigned int width, unsigned int height);

  unsigned int addCube(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);
  unsigned int addGroundPlane(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);
  void clean_geometry();

  Vec3fa* m_cube_face_colors = nullptr;
  Vec3fa* m_cube_vertex_colors = nullptr;
  Vec3fa* m_ground_face_colors = nullptr;
  Vec3fa* m_ground_vertex_colors = nullptr;

  static const std::vector<MaterialType> m_cubeMats;
  static const std::vector<MaterialType> m_groundMats;
};

CubeAndPlane::CubeAndPlane(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
    std::map<unsigned int, size_t>& mapGeomToLightIdx,
    std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
    unsigned int width, unsigned int height) {
  add_geometry(scene, device, mapGeomToPrim);
  setup_camera_and_lights(scene, device, mapGeomToPrim, mapGeomToLightIdx,
                          lights, camera, width, height);
};

void CubeAndPlane::add_geometry(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  addCube(scene, device, mapGeomToPrim);
  addGroundPlane(scene, device, mapGeomToPrim);
}

const std::vector<MaterialType> CubeAndPlane::m_cubeMats = {
    /* Two tris per face and six faces */
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE

};

const std::vector<MaterialType> CubeAndPlane::m_groundMats = {
    MaterialType::MATERIAL_MATTE, MaterialType::MATERIAL_MATTE};

/* adds a cube to the scene */
unsigned int CubeAndPlane::addCube(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  /* create a triangulated cube with 12 triangles and 8 vertices */
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  m_cube_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 12, 16);
  m_cube_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 8, 16);

  /* set vertices and vertex colors */
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 8);
  m_cube_vertex_colors[0] = Vec3fa(0, 0, 0);
  vertices[0].x = -1;
  vertices[0].y = -1;
  vertices[0].z = -1;
  m_cube_vertex_colors[1] = Vec3fa(0, 0, 1);
  vertices[1].x = -1;
  vertices[1].y = -1;
  vertices[1].z = +1;
  m_cube_vertex_colors[2] = Vec3fa(0, 1, 0);
  vertices[2].x = -1;
  vertices[2].y = +1;
  vertices[2].z = -1;
  m_cube_vertex_colors[3] = Vec3fa(0, 1, 1);
  vertices[3].x = -1;
  vertices[3].y = +1;
  vertices[3].z = +1;
  m_cube_vertex_colors[4] = Vec3fa(1, 0, 0);
  vertices[4].x = +1;
  vertices[4].y = -1;
  vertices[4].z = -1;
  m_cube_vertex_colors[5] = Vec3fa(1, 0, 1);
  vertices[5].x = +1;
  vertices[5].y = -1;
  vertices[5].z = +1;
  m_cube_vertex_colors[6] = Vec3fa(1, 1, 0);
  vertices[6].x = +1;
  vertices[6].y = +1;
  vertices[6].z = -1;
  m_cube_vertex_colors[7] = Vec3fa(1, 1, 1);
  vertices[7].x = +1;
  vertices[7].y = +1;
  vertices[7].z = +1;

  /* set triangles and face colors */
  int tri = 0;
  Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 12);

  // left side
  m_cube_face_colors[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 1;
  triangles[tri].v2 = 2;
  tri++;
  m_cube_face_colors[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 2;
  tri++;

  // right side
  m_cube_face_colors[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 4;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 5;
  tri++;
  m_cube_face_colors[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 5;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 7;
  tri++;

  // bottom side
  m_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 1;
  tri++;
  m_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 5;
  tri++;

  // top side
  m_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 6;
  tri++;
  m_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 7;
  triangles[tri].v2 = 6;
  tri++;

  // front side
  m_cube_face_colors[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 2;
  triangles[tri].v2 = 4;
  tri++;
  m_cube_face_colors[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 4;
  tri++;

  // back side
  m_cube_face_colors[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 3;
  tri++;
  m_cube_face_colors[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 7;
  tri++;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, m_cube_vertex_colors, 0,
                             sizeof(Vec3fa), 8);

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = m_cubeMats;
  mpTable.primColorTable = m_cube_face_colors;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));

  return geomID;
}

/* adds a ground plane to the scene */
unsigned int CubeAndPlane::addGroundPlane(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  /* create a triangulated plane with 2 triangles and 4 vertices */
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  m_ground_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 2, 16);
  m_ground_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 4, 16);

  /* Moving the plane up to the bottom of the cube shows more global
  illumination color bleed Try y = -1 to see it!
  */
  /* The color of the ground plane is changed to white to see global
   * illumination effects */
  /* set vertices */
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 4);
  m_ground_vertex_colors[0] = Vec3fa(1, 0, 0);
  vertices[0].x = -10;
  vertices[0].y = -2;
  vertices[0].z = -10;
  m_ground_vertex_colors[1] = Vec3fa(1, 0, 1);
  vertices[1].x = -10;
  vertices[1].y = -2;
  vertices[1].z = +10;
  m_ground_vertex_colors[2] = Vec3fa(1, 1, 0);
  vertices[2].x = +10;
  vertices[2].y = -2;
  vertices[2].z = -10;
  m_ground_vertex_colors[3] = Vec3fa(1, 1, 1);
  vertices[3].x = +10;
  vertices[3].y = -2;
  vertices[3].z = +10;

  /* set triangles */
  Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 2);

  m_ground_face_colors[0] = Vec3fa(1, 0, 0);
  triangles[0].v0 = 0;
  triangles[0].v1 = 1;
  triangles[0].v2 = 2;
  m_ground_face_colors[1] = Vec3fa(1, 0, 0);
  triangles[1].v0 = 1;
  triangles[1].v1 = 3;
  triangles[1].v2 = 2;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, m_ground_vertex_colors, 0,
                             sizeof(Vec3fa), 4);

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = m_groundMats;
  mpTable.primColorTable = m_ground_face_colors;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));

  return geomID;
}

void CubeAndPlane::clean_geometry() {
  if (m_cube_face_colors) alignedFree(m_cube_face_colors);
  m_cube_face_colors = nullptr;
  if (m_cube_vertex_colors) alignedFree(m_cube_vertex_colors);
  m_cube_vertex_colors = nullptr;
  if (m_ground_face_colors) alignedFree(m_ground_face_colors);
  m_ground_face_colors = nullptr;
  if (m_ground_vertex_colors) alignedFree(m_ground_vertex_colors);
  m_ground_vertex_colors = nullptr;
}

CubeAndPlane::~CubeAndPlane() { clean_geometry(); }

void CubeAndPlane::setup_camera_and_lights(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
    std::map<unsigned int, size_t>& mapGeomToLightIdx,
    std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
    unsigned int width, unsigned int height) {
  Vec3fa defaultLightIntensity = {1.0f, 1.0f, 1.0f};

  camera = positionCamera(Vec3fa(1.5f, 1.5f, -1.5f), Vec3fa(0, 0, 0),
                          Vec3fa(0, 1, 0), 90.0f, width, height);
  /* Zoomed out camera position below: */
  // camera = positionCamera(Vec3fa(3.f, 3.f, -3.f), Vec3fa(0, 0, 0),
  //     Vec3fa(0, 1, 0), 90.0f, width, height);

  /* We've implemented a directional light... Try it with this direction :

    Vec3fa defaultLightDirection = normalize(Vec3fa(-1.0f, -1.0f, -1.0f));

    This will give results even more similar to the original triangle geometry
    sample. You may want to change the power as well.
  */
  /* Picking the magnitude of the light can be tricky. Lights such as the point
   * light fall off at the inverse square of the distance. When building a
   * renderer, you may need to scale your light up or down to see your scene. */
  Vec3fa pow = 800.f * Vec3fa(1.f, 1.f, 1.f);

  /* The point light that mimicks the direction of the directional light */
  Vec3fa pos = Vec3fa(10.0f, 10.0f, 10.0f);
  float radius = 0.f;
  lights.push_back(std::make_shared<PointLight>(pos, pow, radius));

  /* If radius is greater than 0 lets try to build a geometry */
  if (radius > 0.f) {
    std::shared_ptr<PointLight> pPointLight =
        std::dynamic_pointer_cast<PointLight>(lights.back());
    unsigned int geomID =
        pPointLight->add_geometry(scene, device, mapGeomToPrim);
    mapGeomToLightIdx.insert(std::make_pair(geomID, lights.size() - 1));
  }
}

#endif /* !FILE_DEFAULTCUBEANDPLANE_SEEN */
