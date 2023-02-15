#pragma once

#ifndef FILE_GEOMETRYH_SEEN
#define FILE_GEOMETRYH_SEEN

#include <embree4/rtcore.h>

#include <map>
#include <vector>

#include "Lights.h"
#include "Materials.h"
#include "definitions.h"

/* We'll use this 'geometries' container to automatically clean up the data
 * arrays created that are used to create embree geometries */
class Geometry {
 public:
  Geometry(){};
  virtual ~Geometry(){/* Empty */};

  // Derived Geometrys will use a custom constructor for Geometries to give
  // Geometry specific parameters needed
 private:
  virtual void add_geometry(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) = 0;

  virtual void setup_camera_and_lights(
      const RTCScene& scene, const RTCDevice& device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim,
      std::map<unsigned int, size_t>& mapGeomToLightIdx,
      std::vector<std::shared_ptr<Light>>& lights, AffineSpace3fa& camera,
      unsigned int width, unsigned int height) = 0;

  virtual void clean_geometry() = 0;
};

#endif /* FILE_GEOMETRYH_SEEN */