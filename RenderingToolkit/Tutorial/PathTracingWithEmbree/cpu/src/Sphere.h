#pragma once
#ifndef FILE_SPHERE_SEEN
#define FILE_SPHERE_SEEN

#include <embree4/rtcore.h>

#include <vector>

#include "Geometry.h"
#include "Materials.h"
#include "definitions.h"

class Sphere : public Geometry {
 public:
  Sphere(RTCScene scene, RTCDevice device,
         std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
         MaterialType sphereMat, const Vec3fa& pos,
         const Vec3fa& color, float radius,
         std::map<unsigned int, size_t>& mapGeomToLightIdx,
         std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
         unsigned int width, unsigned int height);
  ~Sphere();

  unsigned int add_geometry(
      RTCScene scene, RTCDevice device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
      MaterialType sphereMat, const Vec3fa& pos, const Vec3fa& color,
      float radius);

  /* Place holder ... Could put a default sphere in here but we will leave it
   * blank for now */
  void add_geometry(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {}
  /* Place holder ... Could put a default light or camera in here but we will
   * leave it blank for now */
  void setup_camera_and_lights(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
      std::map<unsigned int, size_t>& mapGeomToLightIdx,
      std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
      unsigned int width, unsigned int height) {}
  /* Place holder */
  void clean_geometry();

  // Just one material for our sphere primitive
  MaterialType m_sphereMat;
  Vec3fa m_sphere_face_colors;
  Vec3fa m_pos;
  Vec3fa m_radius;
};

Sphere::Sphere(RTCScene scene, RTCDevice device,
               std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
               MaterialType sphereMat, const Vec3fa& pos,
               const Vec3fa& color, float radius,
               std::map<unsigned int, size_t>& mapGeomToLightIdx,
               std::vector<std::shared_ptr<Light>>& lights,
               AffineSpace3fa& camera, unsigned int width,
               unsigned int height) {
  /* The unused variables are in case you would like to try lights with this
   * sphere. See the setup_camera_and_lights(..) functions from the other
   * Geometry objects */
  m_sphereMat = sphereMat;
  m_sphere_face_colors = color;
  m_pos = pos;
  m_radius = radius;
  add_geometry(scene, device, mapGeomToPrim, sphereMat, pos, color, radius);
}

unsigned int Sphere::add_geometry(
    RTCScene scene, RTCDevice device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
    MaterialType sphereMat, const Vec3fa& pos, const Vec3fa& color,
    float radius) {
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(Vertex), 1);

  // Sphere primitive defined as singular Vec4 point for embree
  Vertex p = {pos.x, pos.y, pos.z, radius};
  vertices[0] = p;

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = std::vector<MaterialType>({m_sphereMat});
  mpTable.primColorTable = &m_sphere_face_colors;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));

  return geomID;
}

Sphere::~Sphere() { clean_geometry(); }

/* Only a place holder. Nothing is here because we do not attach any attributes
 * for the sphere.  */
void Sphere::clean_geometry() {}

#endif /* FILE_SPHERE_SEEN */
