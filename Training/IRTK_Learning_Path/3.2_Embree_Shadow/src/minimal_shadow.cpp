// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <embree4/rtcore.h>
#include <limits>
#include <math.h>
#include <stdio.h>

#if defined(_WIN32)
#include <conio.h>
#include <windows.h>
#endif

/*
 * A minimal tutorial.
 *
 * It demonstrates how to intersect a ray with a single triangle. It is
 * meant to get you started as quickly as possible, and does not output
 * an image.
 *
 */

/*
 * This is only required to make the tutorial compile even when
 * a custom namespace is set.
 */
#if defined(RTC_NAMESPACE_OPEN)
RTC_NAMESPACE_OPEN
#endif

/*
 * We will register this error handler with the device in f_initializeDevice(),
 * so that we are automatically informed on errors.
 * This is extremely helpful for finding bugs in your code, prevents you
 * from having to add explicit error checking to each Embree API call.
 */
void errorFunction(void *userPtr, enum RTCError error, const char *str) {
  printf("error %d: %s\n", error, str);
}


// **************************************************
// Step 1 - Create a device
/*
 * Embree has a notion of devices, which are entities that can run
 * raytracing kernels.
 * We initialize our device here, and then register the error handler so that
 * we don't miss any errors.
 *
 * rtcNewDevice() takes a configuration string as an argument. See the API docs
 * for more information.
 *
 * Note that RTCDevice is reference-counted.
 */

// fnct_initializeDevice is the function called from main to create device
RTCDevice fnct_initializeDevice() {
  // create RTCDevice-type device named my_device
  RTCDevice my_device = rtcNewDevice(NULL);

  if (!my_device)
    printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

  rtcSetDeviceErrorFunction(my_device, errorFunction, NULL);
  return my_device;
}


// **************************************************
// Step 2 - Create scene and geometry and put geometry inside the scene
/*
 * Create a scene, which is a collection of geometry objects. Scenes are
 * what the intersect / occluded functions work on. You can think of a
 * scene as an acceleration structure, e.g. a bounding-volume hierarchy.
 *
 * Scenes, like devices, are reference-counted.
 */
// fnct_geometry_inside_scene is the function called from main to create the scene
RTCScene fnct_geometry_inside_scene(RTCDevice my_device) {
  // create RTCScene-type scene named my_scene
  RTCScene my_scene = rtcNewScene(my_device);

  /*
   * Create a triangle mesh geometry, and initialize a single triangle.
   * You can look up geometry types in the API documentation to
   * find out which type expects which buffers.
   *
   * We create buffers directly on the device, but you can also use
   * shared buffers. For shared buffers, special care must be taken
   * to ensure proper alignment and padding. This is described in
   * more detail in the API documentation.
   */
  // create RTCRTCGeometry-type geometry named my_geom
  RTCGeometry my_geom = rtcNewGeometry(my_device, RTC_GEOMETRY_TYPE_TRIANGLE);
  // create vertices buffer within the geometry
  float *vertices = (float *)rtcSetNewGeometryBuffer(
      my_geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 3);

  // create indices buffer within the geometry
  unsigned *indices = (unsigned *)rtcSetNewGeometryBuffer(
      my_geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned),
      1);

  // assign vertices and indices values to the buffers we just created
  if (vertices && indices) {
    vertices[0] = 0.f;
    vertices[1] = 0.f;
    vertices[2] = 0.f;
    vertices[3] = 1.f;
    vertices[4] = 0.f;
    vertices[5] = 0.f;
    vertices[6] = 0.f;
    vertices[7] = 1.f;
    vertices[8] = 0.f;

    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
  }

  // we commit the geometry
  rtcCommitGeometry(my_geom);

  // and attach the geometry into the scene
  rtcAttachGeometry(my_scene, my_geom);
  /*
  rtcAttachGeometry attaches a geometry to a scene and assigns an ID to that geometry.
  A geometry can get attached to multiplee scenes. The geometry ID is unique for the
  scene, and is used to identify the geometry when hit by a ray during ray queries.
  This function is thread-safe, thus multiple threads can attach geometries to
  a scene in parallel.
  */

  // and we release geom
  rtcReleaseGeometry(my_geom);

  // Like geometry objects, scenes must be committed as well.
  // This lets Embree know that it may start building an acceleration structure.
  rtcCommitScene(my_scene);

  return my_scene;
}


// **************************************************
// Step 3
// fnct_castRay is the function called from main to cast rays (with origin and directions as inputs)
void fnct_castRay(RTCScene my_scene, float ox, float oy, float oz, float dx, float dy,
             float dz) {
  /*
   * The intersect context can be used to set intersection
   * filters or flags, and it also contains the instance ID stack
   * used in multi-level instancing.
   */
  //struct RTCIntersectContext context;
  //rtcInitIntersectContext(&context);

  // Create the RTCRayHit structure named rayhit. It holds the ray information and the hit information
  struct RTCRayHit rayhit;
  rayhit.ray.org_x = ox;
  rayhit.ray.org_y = oy;
  rayhit.ray.org_z = oz;
  rayhit.ray.dir_x = dx;
  rayhit.ray.dir_y = dy;
  rayhit.ray.dir_z = dz;
  rayhit.ray.tnear = 0;
  rayhit.ray.tfar = std::numeric_limits<float>::infinity();
  rayhit.ray.mask  = -1;
  rayhit.ray.flags = 0;
  rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  // with rtcIntersect1 we intersect the rayhit structure with the scene
  //rtcIntersect1(my_scene, &context, &rayhit);
    rtcIntersect1(my_scene, &rayhit);
  // there are multiple variants of rtcIntersect. This one intersects a single ray with the scene

  printf("\n%f, %f, %f: ", ox, oy, oz);
  if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    /* Note how geomID and primID identify the geometry we just hit.
     * We could use them here to interpolate geometry information, compute shading, etc.
     * Since there is only a single triangle in this scene, we will
     * get geomID=0 / primID=0 for all hits.
     * There is also instID, used for instancing. See
     * the instancing tutorials for more information */
    printf("Found intersection on geometry %d, primitive %d at tfar=%f\n",
           rayhit.hit.geomID, rayhit.hit.primID, rayhit.ray.tfar);

    printf("Hit position UV %4.4f, %4.4f\n", rayhit.hit.u, rayhit.hit.v);
    printf("Hit normal %4.4f, %4.4f, %4.4f\n", rayhit.hit.Ng_x, rayhit.hit.Ng_y,
           rayhit.hit.Ng_z);
      
      
    /* Set up the next origin we use from the intersection test result: */
    float nextX, nextY, nextZ;
    nextX = nextY = nextZ = 0.f;

    // Compute the intersection point in cartesian 3D coordinates given
    // 1) Primary ray origin
    // 2) Embree computed intersection distance (tfar)
    // 3) Direction of primary ray
    nextX = rayhit.ray.org_x + rayhit.ray.tfar * rayhit.ray.dir_x;
    nextY = rayhit.ray.org_y + rayhit.ray.tfar * rayhit.ray.dir_y;
    nextZ = rayhit.ray.org_z + rayhit.ray.tfar * rayhit.ray.dir_z;

    // Compute a vector/ray representing the intersection ray cast. omega-i (w-sub-i) is used in many shading literatures.
    // The head of the ray is at the cast origin. The tail is at the surface intersection. This ray can be used with shading w.r.t. surface normals found in the rayhit.hit data structure.
    float wi1x = rayhit.ray.org_x - nextX;
    float wi1y = rayhit.ray.org_y - nextY;
    float wi1z = rayhit.ray.org_z - nextZ;

    /* Determine side of triangle of the intersection */
    /* Use the dot product of the normal to the intersection ray cast. */
    /* We use the analytical capability of the cosine by way of the dot product */
    float sign = (wi1x * rayhit.hit.Ng_x + wi1y * rayhit.hit.Ng_y + wi1z * rayhit.hit.Ng_z) < 0.f ? -1.f : 1.f;

    // Create our next origin point
    // If the next origin is immediately on the surface, subsequent intersection or occlusion ray casts may intersect their starting surface. An origin intersection is undesired behavior.
    // In practice, the subsequent cast must move some small bit (epsilon) away from the surface. Note that in other Embree tutortials, epsilon is created based on floating point resolution.
    float epsilon = 0.001f;
    nextX = nextX + sign * epsilon * rayhit.hit.Ng_x;
    nextY = nextY + sign * epsilon * rayhit.hit.Ng_y;
    nextZ = nextZ + sign * epsilon * rayhit.hit.Ng_z; 

    // Perform 2 shadow ray casts
    // The direction is hard coded but represents some computed direction based on a light location.
    // In a real application the direction vector would be determined from the light location and the surface point
    // The first is in shadow. Any material would not interact with the tested light.
    printf("x %f, y %f, z %f\n", nextX, nextY, nextZ);
    printf("Shadow cast FROM just above the surface intersection. DIRECTION through the surface to the light. Should be a hit. (Surface point is in-shadow):\n");
    rayhit.ray.org_x = nextX;
    rayhit.ray.org_y = nextY;
    rayhit.ray.org_z = nextZ;
    rayhit.ray.dir_x = 0.f;
    rayhit.ray.dir_y = 0.f;
    rayhit.ray.dir_z = 1.0f;
    rayhit.ray.tnear = 0;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcOccluded1(my_scene, &rayhit.ray);
    if (rayhit.ray.tfar < 0.f)
        printf("#surface is in shadow\n");
    else
        printf("#surface is in light\n");

    // The second is in the light.
    // Some material could iteract with the light. A material may have a Bidirectional Ray Distribution Function that is affected by the light's intensity, distance, and angle.
    printf("x %f, y %f, z %f\n", nextX, nextY, nextZ);
    printf("Shadow cast FROM just above the surface intersection. DIRECTION away from the surface should be a miss (Surface point in-light):\n");
    rayhit.ray.org_x = nextX;
    rayhit.ray.org_y = nextY;
    rayhit.ray.org_z = nextZ;
    rayhit.ray.dir_x = 0.f;
    rayhit.ray.dir_y = 0.f;
    rayhit.ray.dir_z = -1.0f;
    rayhit.ray.tnear = 0;
    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcOccluded1(my_scene, &rayhit.ray);
    if (rayhit.ray.tfar < 0.f)
        printf("#surface is in shadow\n");
    else
        printf("#surface is in light\n");

  } else
    printf("Did not find any intersection.\n");
}

// **************************************************
// Step 4 - main function
int main() {
  /* Initialization. All of this may fail, but we will be notified by
   * our errorFunction. */
  RTCDevice my_device = fnct_initializeDevice();
  RTCScene  my_scene  = fnct_geometry_inside_scene(my_device);

  // we cast the first ray
  fnct_castRay(my_scene, 0, 0, -1, 0, 0, 1);

  // we cast the second ray
  fnct_castRay(my_scene, 1, 1, -1, 0, 0, 1);
  // Note that the direction of both rays is the same (0, 0, 1)

  // you should always make sure to release resources allocated
  rtcReleaseScene(my_scene);
  rtcReleaseDevice(my_device);
  
  printf("success...\n");
  return 0;
}
