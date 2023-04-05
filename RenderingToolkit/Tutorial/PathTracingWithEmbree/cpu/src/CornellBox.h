#pragma once
#ifndef FILE_CORNELLBOX_SEEN
#define FILE_CORNELLBOX_SEEN

#include <embree4/rtcore.h>

#include <vector>

#include "Geometry.h"
#include "Lights.h"
#include "Materials.h"
#include "definitions.h"

class CornellBoxGeometry : public Geometry {
 public:
  CornellBoxGeometry(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
      std::map<unsigned int, size_t>& mapGeomToLightIdx,
      std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
      unsigned int width, unsigned int height);
  ~CornellBoxGeometry();

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
  void clean_geometry();
  /* Added for pathtracer */
  Vec3fa* m_cornell_face_colors = nullptr;
  Vec3fa* m_cornell_vertex_colors = nullptr;

  // mesh data
  static const std::vector<Vertex> m_cornellBoxVertices;

  static const std::vector<Vec3fa> m_cornellBoxColors;

  static const std::vector<MaterialType> m_cornellBoxMats;

  static const std::vector<Quad> m_cornellBoxIndices;
};

CornellBoxGeometry::CornellBoxGeometry(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
    std::map<unsigned int, size_t>& mapGeomToLightIdx,
    std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
    unsigned int width, unsigned int height) {
  add_geometry(scene, device, mapGeomToPrim);
  setup_camera_and_lights(scene, device, mapGeomToPrim, mapGeomToLightIdx,
                          lights, camera, width, height);
}

void CornellBoxGeometry::add_geometry(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  /* create a mesh for all the quads in the Cornell Box scene */
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD);
  m_cornell_face_colors =
      (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * m_cornellBoxIndices.size(), 16);
  m_cornell_vertex_colors =
      (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * m_cornellBoxVertices.size(), 16);
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
      m_cornellBoxVertices.size());
  for (auto i = 0; i < m_cornellBoxVertices.size(); i++) {
    vertices[i] = m_cornellBoxVertices[i];
    m_cornell_vertex_colors[i] = m_cornellBoxColors[i];
  }

  /* set quads */
  Quad* quads = (Quad*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0,
                                               RTC_FORMAT_UINT4, sizeof(Quad),
                                               m_cornellBoxIndices.size());

  for (auto i = 0; i < m_cornellBoxIndices.size(); i++) {
    quads[i] = m_cornellBoxIndices[i];
    m_cornell_face_colors[i] = m_cornellBoxColors[i * 4];
  }

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, *m_cornell_vertex_colors, 0,
                             sizeof(Vec3fa), m_cornellBoxVertices.size());

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = m_cornellBoxMats;
  mpTable.primColorTable = m_cornell_face_colors;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));
}

void CornellBoxGeometry::clean_geometry() {
  if (m_cornell_face_colors) alignedFree(m_cornell_face_colors);
  m_cornell_face_colors = nullptr;
  if (m_cornell_vertex_colors) alignedFree(m_cornell_vertex_colors);
  m_cornell_vertex_colors = nullptr;
}

void CornellBoxGeometry::setup_camera_and_lights(
    const RTCScene& scene, const RTCDevice& device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
    std::map<unsigned int, size_t>& mapGeomToLightIdx,
    std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
    unsigned int width, unsigned int height) {
  /* A default camera view as specified from Cornell box presets given input
   * from Intel OSPRay*/
  // camera = positionCamera(Vec3fa(0.0, 0.0, -2.0f), Vec3fa(0, 0, 0),
  //     Vec3fa(0, 1, 0), 90.0f, width, height);

  /* A camera position that connects the field of vision angle of the camera to
   * the bounds of the cornell box */
  float fov = 30.0f;
  float fovrad = fov * M_PI / 180.0f;
  float half_fovrad = fovrad * 0.5f;
  camera = positionCamera(Vec3fa(0.0, 0.0, -1.0f - 1.f / tanf(half_fovrad)),
                          Vec3fa(0, 0, 0), Vec3fa(0, 1, 0), fov, width, height);

  /* The magnitude of the light can be tricky. Lights such as the point light
   * fall off at the inverse square of the distance. When designing a sandbox
   * renderer, you may need to scale your light up or down to see your scene. */
  // Vec3fa pow = 3.f * Vec3fa(0.78f, 0.551f, 0.183f);
  /* An interesting position for an overhead light in the Cornell Box scene.
   * Notice increased noise when lights are near objects */
  // Vec3fa pos = Vec3fa(0.0f, 0.95f, 0.0f);

  /* A somewhat central position for the point light within the box. This is
   * similar to the position for the interactive pathtracer program shipped with
   * Intel Embree */
  // Vec3fa pos =
  //     Vec3fa(2.f * 213.0f / 556.0f - 1.f, 2.f * 300.f / 558.8f - 1.f,
  //            2.f * 227.f / 559.2f - 1.f);

  /* Below is a setup for a delta point light or a spherical geometric light */

  /* Delta is 0 radius */
  // float radius = 0.f;
  // float radius = 0.075f;
  // lights.push_back(std::make_shared<PointLight>(pos, pow, radius));
  // Place holder to toggle light geometries
  // if (radius > 0.f && true) {
  //     std::shared_ptr<PointLight> newPointLight =
  //     std::dynamic_pointer_cast<PointLight>(lights.back());
  //    unsigned int geomID = newPointLight->add_geometry(scene, device);
  //    mapGeomToLightIdx.insert(std::make_pair(geomID, lights.size() - 1));
  //}

  /* Here we have a light as a disc geometry */
  Vec3fa spotPos(0.f, 0.95f, 0.0f);
  Vec3fa spotDir(0.f, -1.f, 0.f);
  Vec3fa spotPow = 10.f * Vec3fa(0.78f, 0.551f, 0.183f);
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

CornellBoxGeometry::~CornellBoxGeometry() { clean_geometry(); }

const std::vector<Vertex> CornellBoxGeometry::m_cornellBoxVertices = {
    // Floor
    {1.00f, -1.00f, -1.00f, 0.0f},
    {-1.00f, -1.00f, -1.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    // Ceiling
    {1.00f, 1.00f, -1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, -1.00f, 0.0f},

    // Backwall
    {1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    // RightWall
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, -1.00f, 0.0f},
    {-1.00f, 1.00f, -1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    // LeftWall
    {1.00f, -1.00f, -1.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, -1.00f, 0.0f},
    // ShortBox Top Face
    {-0.53f, -0.40f, -0.75f, 0.0f},
    {-0.70f, -0.40f, -0.17f, 0.0f},
    {-0.13f, -0.40f, -0.00f, 0.0f},
    {0.05f, -0.40f, -0.57f, 0.0f},
    // ShortBox Left Face
    {0.05f, -1.00f, -0.57f, 0.0f},
    {0.05f, -0.40f, -0.57f, 0.0f},
    {-0.13f, -0.40f, -0.00f, 0.0f},
    {-0.13f, -1.00f, -0.00f, 0.0f},
    // ShortBox Front Face
    {-0.53f, -1.00f, -0.75f, 0.0f},
    {-0.53f, -0.40f, -0.75f, 0.0f},
    {0.05f, -0.40f, -0.57f, 0.0f},
    {0.05f, -1.00f, -0.57f, 0.0f},
    // ShortBox Right Face
    {-0.70f, -1.00f, -0.17f, 0.0f},
    {-0.70f, -0.40f, -0.17f, 0.0f},
    {-0.53f, -0.40f, -0.75f, 0.0f},
    {-0.53f, -1.00f, -0.75f, 0.0f},
    // ShortBox Back Face
    {-0.13f, -1.00f, -0.00f, 0.0f},
    {-0.13f, -0.40f, -0.00f, 0.0f},
    {-0.70f, -0.40f, -0.17f, 0.0f},
    {-0.70f, -1.00f, -0.17f, 0.0f},
    // ShortBox Bottom Face
    {-0.53f, -1.00f, -0.75f, 0.0f},
    {-0.70f, -1.00f, -0.17f, 0.0f},
    {-0.13f, -1.00f, -0.00f, 0.0f},
    {0.05f, -1.00f, -0.57f, 0.0f},
    // TallBox Top Face
    {0.53f, 0.20f, -0.09f, 0.0f},
    {-0.04f, 0.20f, 0.09f, 0.0f},
    {0.14f, 0.20f, 0.67f, 0.0f},
    {0.71f, 0.20f, 0.49f, 0.0f},
    // TallBox Left Face
    {0.53f, -1.00f, -0.09f, 0.0f},
    {0.53f, 0.20f, -0.09f, 0.0f},
    {0.71f, 0.20f, 0.49f, 0.0f},
    {0.71f, -1.00f, 0.49f, 0.0f},
    // TallBox Front Face
    {0.71f, -1.00f, 0.49f, 0.0f},
    {0.71f, 0.20f, 0.49f, 0.0f},
    {0.14f, 0.20f, 0.67f, 0.0f},
    {0.14f, -1.00f, 0.67f, 0.0f},
    // TallBox Right Face
    {0.14f, -1.00f, 0.67f, 0.0f},
    {0.14f, 0.20f, 0.67f, 0.0f},
    {-0.04f, 0.20f, 0.09f, 0.0f},
    {-0.04f, -1.00f, 0.09f, 0.0f},
    // TallBox Back Face
    {-0.04f, -1.00f, 0.09f, 0.0f},
    {-0.04f, 0.20f, 0.09f, 0.0f},
    {0.53f, 0.20f, -0.09f, 0.0f},
    {0.53f, -1.00f, -0.09f, 0.0f},
    // TallBox Bottom Face
    {0.53f, -1.00f, -0.09f, 0.0f},
    {-0.04f, -1.00f, 0.09f, 0.0f},
    {0.14f, -1.00f, 0.67f, 0.0f},
    {0.71f, -1.00f, 0.49f, 0.0f}};

const std::vector<Quad> CornellBoxGeometry::m_cornellBoxIndices = {
    {0, 1, 2, 3},      // Floor
    {4, 5, 6, 7},      // Ceiling
    {8, 9, 10, 11},    // Backwall
    {12, 13, 14, 15},  // RightWall
    {16, 17, 18, 19},  // LeftWall
    {20, 21, 22, 23},  // ShortBox Top Face
    {24, 25, 26, 27},  // ShortBox Left Face
    {28, 29, 30, 31},  // ShortBox Front Face
    {32, 33, 34, 35},  // ShortBox Right Face
    {36, 37, 38, 39},  // ShortBox Back Face
    {40, 41, 42, 43},  // ShortBox Bottom Face
    {44, 45, 46, 47},  // TallBox Top Face
    {48, 49, 50, 51},  // TallBox Left Face
    {52, 53, 54, 55},  // TallBox Front Face
    {56, 57, 58, 59},  // TallBox Right Face
    {60, 61, 62, 63},  // TallBox Back Face
    {64, 65, 66, 67}   // TallBox Bottom Face
};

const std::vector<Vec3fa> CornellBoxGeometry::m_cornellBoxColors = {
    // Floor
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // Ceiling
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // Backwall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // RightWall
    {0.140f, 0.450f, 0.091f},
    {0.140f, 0.450f, 0.091f},
    {0.140f, 0.450f, 0.091f},
    {0.140f, 0.450f, 0.091f},
    // LeftWall
    {0.630f, 0.065f, 0.05f},
    {0.630f, 0.065f, 0.05f},
    {0.630f, 0.065f, 0.05f},
    {0.630f, 0.065f, 0.05f},
    // ShortBox Top Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Left Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Front Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Right Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Back Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Bottom Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    /* 0.8f intensity of reflectance gives a decent proxy for a great real life
   mirror */
    // TallBox Top Face
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    // TallBox Left Face
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    // TallBox Front Face
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    // TallBox Right Face
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    // TallBox Back Face
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    // TallBox Bottom Face
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f},
    {0.8f, 0.8f, 0.8f}

    /* Original colors of TallBox */
    // TallBox Top Face
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //// TallBox Left Face
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //// TallBox Front Face
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //// TallBox Right Face
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //// TallBox Back Face
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //// TallBox Bottom Face
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f},
    //{0.725f, 0.710f, 0.68f}

};

const std::vector<MaterialType>
    CornellBoxGeometry::m_cornellBoxMats = {
        // Floor
        MaterialType::MATERIAL_MATTE,
        /* Swap in thr below material to make the ceiling a matte material*/
        // Ceiling
        MaterialType::MATERIAL_MATTE,
        /* Swap in the below material to make the ceiling a mirror */
        /*
        //Ceiling
        MaterialType::MATERIAL_MIRROR,
        */
        // Backwall
        MaterialType::MATERIAL_MATTE,
        // RightWall
        MaterialType::MATERIAL_MATTE,
        // LeftWall
        MaterialType::MATERIAL_MATTE,
        /* Small box configuration for a matte material. Swap this section in
           for the glass (thin dielectric) material as desired */

        // ShortBox Top Face
        MaterialType::MATERIAL_MATTE,
        // ShortBox Left Face
        MaterialType::MATERIAL_MATTE,
        // ShortBox Front Face
        MaterialType::MATERIAL_MATTE,
        // ShortBox Right Face
        MaterialType::MATERIAL_MATTE,
        // ShortBox Back Face
        MaterialType::MATERIAL_MATTE,
        // ShortBox Bottom Face
        MaterialType::MATERIAL_MATTE,

        /* Small box configuration for glass material. Swap this section in for
           the matte above. */
        /*
        // ShortBox Top Face
        MaterialType::MATERIAL_GLASS,
        // ShortBox Left Face
        MaterialType::MATERIAL_GLASS,
        // ShortBox Front Face
        MaterialType::MATERIAL_GLASS,
        // ShortBox Right Face
        MaterialType::MATERIAL_GLASS,
        // ShortBox Back Face
        MaterialType::MATERIAL_GLASS,
        // ShortBox Bottom Face
        MaterialType::MATERIAL_GLASS,
        */
        /* Tall Box configuration for a matte material. Swap this section in for
           the mirror tall box (below) as desired*/

        // TallBox Top Face
        //MaterialType::MATERIAL_MATTE,
        //// TallBox Left Face
        //MaterialType::MATERIAL_MATTE,
        //// TallBox Front Face
        //MaterialType::MATERIAL_MATTE,
        //// TallBox Right Face
        //MaterialType::MATERIAL_MATTE,
        //// TallBox Back Face
        //MaterialType::MATERIAL_MATTE,
        //// TallBox Bottom Face
        //MaterialType::MATERIAL_MATTE

        /* Tall box configuration for a mirror material. Swap this section in to
           see behind the short cube */
        // TallBox Top Face
        MaterialType::MATERIAL_MIRROR,
        // TallBox Left Face
        MaterialType::MATERIAL_MIRROR,
        // TallBox Front Face
        MaterialType::MATERIAL_MIRROR,
        // TallBox Right Face
        MaterialType::MATERIAL_MIRROR,
        // TallBox Back Face
        MaterialType::MATERIAL_MIRROR,
        // TallBox Bottom Face
        MaterialType::MATERIAL_MIRROR

};

#endif /* !FILE_CORNELLBOX_SEEN */
