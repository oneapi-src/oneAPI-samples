#ifdef _MSC_VER
#ifndef NOMINMAX
/* use intended min and max instead of the MSVS macros */
#define NOMINMAX
#endif
#endif

/* Added for GPU */
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include "MathBindings.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

using Vec3fa = MathBindings::Vec3fa;
using LinearSpace3 = MathBindings::LinearSpace3fa;
using AffineSpace3fa = MathBindings::AffineSpace3fa;
using MathBindings::clamp;
using MathBindings::cross;
using MathBindings::deg2rad;
using MathBindings::dot;
using MathBindings::normalize;
using std::max;
using std::min;

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8


typedef AffineSpace3fa Camera;

/* The allocator helps us use USM pointers for the vector container */
/* Acheived: Dynamic element creation (push_back), USM pointer access in the
 * device kernel, Device Reads memory thus no transfer back occurs upon kernel
 * completion */
/* Dynamic RO designation is set up when we create an allocator object later */
/* This allocator object is useful for std::vector and serves as self contained
 * provisioning of functionality similar to the
 * alignedSYCLUSMMallocDeviceReadOnly(const sycl::queue& queue, size_t count,
 * size_t align) function defined globally */
typedef sycl::usm_allocator<unsigned int, sycl::usm::alloc::shared>
    USMDRO_UI_ALLOC;

const sycl::specialization_id<RTCFeatureFlags> feature_mask;
const RTCFeatureFlags required_features = RTC_FEATURE_FLAG_TRIANGLE;

/*
 * This function allocated USM memory that is only readable by the
 * device. Using this mode many small allocations are possible by the
 * application.
 */

template <typename T>
T* alignedSYCLUSMMallocDeviceReadOnly(const sycl::queue& queue, size_t count,
                                      size_t align) {
  if (count == 0) return nullptr;

  assert((align & (align - 1)) == 0);
  T* ptr = (T*)sycl::aligned_alloc_shared(
      align, count * sizeof(T), queue,
      sycl::ext::oneapi::property::usm::device_read_only());
  if (count != 0 && ptr == nullptr) throw std::bad_alloc();

  return ptr;
}

/*
 * This function allocates USM memory that can be accessible back on the host
 * (ex: output image buffer)
 */
template <typename T>
T* alignedSYCLUSMMalloc(const sycl::queue& queue, size_t count, size_t align) {
  if (count == 0) return nullptr;

  assert((align & (align - 1)) == 0);

  void* ptr = nullptr;
  ptr = sycl::aligned_alloc_shared(align, count * sizeof(T), queue);

  if (count != 0 && ptr == nullptr) throw std::bad_alloc();

  return static_cast<T*>(ptr);
}

void alignedSYCLFree(const sycl::queue& queue, void* ptr) {
  if (ptr) sycl::free(ptr, queue);
}

inline sycl::nd_range<2> make_nd_range(unsigned int size0, unsigned int size1) {
  const sycl::range<2> wg_size = sycl::range<2>(4, 4);

  /* align iteration space to work group size */
  size0 = ((size0 + wg_size[0] - 1) / wg_size[0]) * wg_size[0];
  size1 = ((size1 + wg_size[1] - 1) / wg_size[1]) * wg_size[1];

  return sycl::nd_range(sycl::range(size0, size1), wg_size);
}

void error_handler(void* userPtr, const RTCError code, const char* str) {
  if (code == RTC_ERROR_NONE) return;

  printf("fail: Embree Error: ");
  switch (code) {
    case RTC_ERROR_UNKNOWN:
      printf("RTC_ERROR_UNKNOWN");
      break;
    case RTC_ERROR_INVALID_ARGUMENT:
      printf("RTC_ERROR_INVALID_ARGUMENT");
      break;
    case RTC_ERROR_INVALID_OPERATION:
      printf("RTC_ERROR_INVALID_OPERATION");
      break;
    case RTC_ERROR_OUT_OF_MEMORY:
      printf("RTC_ERROR_OUT_OF_MEMORY");
      break;
    case RTC_ERROR_UNSUPPORTED_CPU:
      printf("RTC_ERROR_UNSUPPORTED_CPU");
      break;
    case RTC_ERROR_CANCELLED:
      printf("RTC_ERROR_CANCELLED");
      break;
    default:
      printf("invalid error code");
      break;
  }
  if (str) {
    printf(" (");
    while (*str) putchar(*str++);
    printf(")\n");
  }
  exit(1);
}

/* from tutorial_device.h */
/* vertex and triangle layout */
struct Vertex {
  float x, y, z, r;
};
struct Triangle {
  int v0, v1, v2;
};

Camera positionCamera(Vec3fa from, Vec3fa to, Vec3fa up, float fov,
                          size_t width, size_t height) {
  /* There are many ways to set up a camera projection. This one is consolidated
   * from the camera code in the Embree/tutorial/common/tutorial/camera.h object
   */
  Camera camMatrix;
  Vec3fa Z = normalize(Vec3fa(to - from));
  Vec3fa U = normalize(cross(up, Z));
  Vec3fa V = normalize(cross(Z, U));
  camMatrix.l.vx = U;
  camMatrix.l.vy = V;
  camMatrix.l.vz = Z;
  camMatrix.p = from;

  /* negate for a right handed camera*/
  camMatrix.l.vx = -camMatrix.l.vx;

  const float fovScale = 1.0f / tanf(deg2rad(0.5f * fov));

  camMatrix.l.vz = -0.5f * width * camMatrix.l.vx +
                   0.5f * height * camMatrix.l.vy +
                   0.5f * height * fovScale * camMatrix.l.vz;
  camMatrix.l.vy = -camMatrix.l.vy;

  return camMatrix;
}

/* adds a cube to the scene */
unsigned int addCube(const sycl::queue& queue, const RTCDevice& emb_device,
                     const RTCScene& scene, Vec3fa** pp_cube_face_colors,
                     Vec3fa** pp_cube_vertex_colors) {
  /*
   * Create a triangle mesh geometry for the cube, and initialize 12 triangles.
   * You can look up geometry types in the API documentation to
   * find out which type expects which buffers.
   *
   * We create buffers directly on the device, but you can also use
   * shared buffers. For shared buffers, special care must be taken
   * to ensure proper alignment and padding. This is described in
   * more detail in the API documentation.
   */
  RTCGeometry mesh = rtcNewGeometry(emb_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  *pp_cube_face_colors =
      alignedSYCLUSMMallocDeviceReadOnly<Vec3fa>(queue, 12, 16);
  /* For GPU, we change the vertex color buffer allocation from aligned malloc
   * to a SYCL allocator */
  *pp_cube_vertex_colors =
      alignedSYCLUSMMallocDeviceReadOnly<Vec3fa>(queue, 8, 16);

  /* set vertices and vertex colors */
  Vertex* vertices = alignedSYCLUSMMallocDeviceReadOnly<Vertex>(queue, 8, 16);

  /* For GPU, change rtcSetNewGeometryBuffer to rtcSetSharedGeometryBuffer to
   * work with the SYCL allocator */
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                             vertices, 0, sizeof(Vertex), 8);

  (*pp_cube_vertex_colors)[0] = Vec3fa(0, 0, 0);
  vertices[0].x = -1;
  vertices[0].y = -1;
  vertices[0].z = -1;
  (*pp_cube_vertex_colors)[1] = Vec3fa(0, 0, 1);
  vertices[1].x = -1;
  vertices[1].y = -1;
  vertices[1].z = +1;
  (*pp_cube_vertex_colors)[2] = Vec3fa(0, 1, 0);
  vertices[2].x = -1;
  vertices[2].y = +1;
  vertices[2].z = -1;
  (*pp_cube_vertex_colors)[3] = Vec3fa(0, 1, 1);
  vertices[3].x = -1;
  vertices[3].y = +1;
  vertices[3].z = +1;
  (*pp_cube_vertex_colors)[4] = Vec3fa(1, 0, 0);
  vertices[4].x = +1;
  vertices[4].y = -1;
  vertices[4].z = -1;
  (*pp_cube_vertex_colors)[5] = Vec3fa(1, 0, 1);
  vertices[5].x = +1;
  vertices[5].y = -1;
  vertices[5].z = +1;
  (*pp_cube_vertex_colors)[6] = Vec3fa(1, 1, 0);
  vertices[6].x = +1;
  vertices[6].y = +1;
  vertices[6].z = -1;
  (*pp_cube_vertex_colors)[7] = Vec3fa(1, 1, 1);
  vertices[7].x = +1;
  vertices[7].y = +1;
  vertices[7].z = +1;

  /* set triangles and face colors */
  int tri = 0;
  Triangle* triangles =
      alignedSYCLUSMMallocDeviceReadOnly<Triangle>(queue, 12, 16);

  /* For GPU, change rtcSetNewGeometryBuffer to rtcSetSharedGeometryBuffer to
   * work with the SYCL allocator */
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                             triangles, 0, sizeof(Triangle), 12);

  // left side
  (*pp_cube_face_colors)[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 1;
  triangles[tri].v2 = 2;
  tri++;
  (*pp_cube_face_colors)[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 2;
  tri++;

  // right side
  (*pp_cube_face_colors)[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 4;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 5;
  tri++;
  (*pp_cube_face_colors)[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 5;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 7;
  tri++;

  // bottom side
  (*pp_cube_face_colors)[tri] = Vec3fa(0.5f);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 1;
  tri++;
  (*pp_cube_face_colors)[tri] = Vec3fa(0.5f);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 5;
  tri++;

  // top side
  (*pp_cube_face_colors)[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 6;
  tri++;
  (*pp_cube_face_colors)[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 7;
  triangles[tri].v2 = 6;
  tri++;

  // front side
  (*pp_cube_face_colors)[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 2;
  triangles[tri].v2 = 4;
  tri++;
  (*pp_cube_face_colors)[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 4;
  tri++;

  // back side
  (*pp_cube_face_colors)[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 3;
  tri++;
  (*pp_cube_face_colors)[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 7;
  tri++;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, *pp_cube_vertex_colors, 0,
                             sizeof(Vec3fa), 8);
  /*
   * You must commit geometry objects when you are done setting them up,
   * or you will not get any intersections.
   */
  rtcCommitGeometry(mesh);

  /*
   * In rtcAttachGeometry(...), the scene takes ownership of the geom
   * by increasing its reference count. This means that we don't have
   * to hold on to the geom handle, and may release it. The geom object
   * will be released automatically when the scene is destroyed.
   *
   * rtcAttachGeometry() returns a geometry ID. We could use this to
   * identify intersected objects later on.
   */
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);
  return geomID;
}

/* adds a ground plane to the scene */
unsigned int addGroundPlane(const sycl::queue& queue,
                            const RTCDevice& emb_device,
                            const RTCScene& emb_scene,
                            Vec3fa** pp_ground_face_colors,
                            Vec3fa** pp_ground_vertex_colors) {
  /* create a triangulated plane with 2 triangles and 4 vertices */
  RTCGeometry mesh = rtcNewGeometry(emb_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  *pp_ground_face_colors =
      alignedSYCLUSMMallocDeviceReadOnly<Vec3fa>(queue, 2, 16);
  /* For GPU, we change the vertex color buffer allocation from aligned malloc
   * to a SYCL allocator */
  *pp_ground_vertex_colors =
      alignedSYCLUSMMallocDeviceReadOnly<Vec3fa>(queue, 4, 16);

  /* set vertices and vertex colors */
  Vertex* vertices = alignedSYCLUSMMallocDeviceReadOnly<Vertex>(queue, 4, 16);

  /* For GPU, change rtcSetNewGeometryBuffer to rtcSetSharedGeometryBuffer to
   * work with the SYCL allocator */
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                             vertices, 0, sizeof(Vertex), 4);

  (*pp_ground_vertex_colors)[0] = Vec3fa(1, 0, 0);
  vertices[0].x = -10;
  vertices[0].y = -2;
  vertices[0].z = -10;
  (*pp_ground_vertex_colors)[1] = Vec3fa(1, 0, 1);
  vertices[1].x = -10;
  vertices[1].y = -2;
  vertices[1].z = +10;
  (*pp_ground_vertex_colors)[2] = Vec3fa(1, 1, 0);
  vertices[2].x = +10;
  vertices[2].y = -2;
  vertices[2].z = -10;
  (*pp_ground_vertex_colors)[3] = Vec3fa(1, 1, 1);
  vertices[3].x = +10;
  vertices[3].y = -2;
  vertices[3].z = +10;

  /* set triangles */
  Triangle* triangles =
      alignedSYCLUSMMallocDeviceReadOnly<Triangle>(queue, 2, 16);

  /* For GPU, change rtcSetNewGeometryBuffer to rtcSetSharedGeometryBuffer to
   * work with the SYCL allocator */
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                             triangles, 0, sizeof(Triangle), 2);

  (*pp_ground_face_colors)[0] = Vec3fa(1, 0, 0);
  triangles[0].v0 = 0;
  triangles[0].v1 = 1;
  triangles[0].v2 = 2;
  (*pp_ground_face_colors)[1] = Vec3fa(1, 0, 0);
  triangles[1].v0 = 1;
  triangles[1].v1 = 3;
  triangles[1].v2 = 2;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, *pp_ground_vertex_colors, 0,
                             sizeof(Vec3fa), 4);
  /*
   * You must commit geometry objects when you are done setting them up,
   * or you will not get any intersections.
   */
  rtcCommitGeometry(mesh);
  /*
   * In rtcAttachGeometry(...), the scene takes ownership of the geom
   * by increasing its reference count. This means that we don't have
   * to hold on to the geom handle, and may release it. The geom object
   * will be released automatically when the scene is destroyed.
   *
   * rtcAttachGeometry() returns a geometry ID. We could use this to
   * identify intersected objects later on.
   */
  unsigned int geomID = rtcAttachGeometry(emb_scene, mesh);
  rtcReleaseGeometry(mesh);
  return geomID;
}

/* task that renders a single screen pixel */
void renderPixelStandard(
    int x, int y, unsigned char* pixels, const unsigned int width,
    const unsigned int height, const unsigned int channels, const float time,
    Camera* p_camera, const RTCFeatureFlags features, const RTCScene& scene,
    Vec3fa* p_cube_face_colors, Vec3fa* p_ground_face_colors,
    unsigned int* p_usm_geomIDs) {
  RTCIntersectArguments iargs;
  rtcInitIntersectArguments(&iargs);

  iargs.feature_mask = features;

  const Vec3fa dir =
      normalize(x * p_camera->l.vx + y * p_camera->l.vy + p_camera->l.vz);
  const Vec3fa org = Vec3fa(p_camera->p.x, p_camera->p.y, p_camera->p.z);

  /* initialize ray */
  RTCRayHit rhPrimary;
  rhPrimary.ray.dir_x = dir.x;
  rhPrimary.ray.dir_y = dir.y;
  rhPrimary.ray.dir_z = dir.z;
  rhPrimary.ray.org_x = org.x;
  rhPrimary.ray.org_y = org.y;
  rhPrimary.ray.org_z = org.z;
  rhPrimary.ray.tnear = 0.0f;
  rhPrimary.ray.time = time;
  rhPrimary.ray.tfar = std::numeric_limits<float>::infinity();
  rhPrimary.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rhPrimary.hit.primID = RTC_INVALID_GEOMETRY_ID;
  rhPrimary.ray.mask = -1;

  /* intersect ray with scene */
  rtcIntersect1(scene, &rhPrimary, &iargs);

  /* shade pixels */
  Vec3fa color = Vec3fa(0.0f);
  if (rhPrimary.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    Vec3fa diffuse;

    if (rhPrimary.hit.geomID == p_usm_geomIDs[0])
      diffuse = p_cube_face_colors[rhPrimary.hit.primID];
    else if (rhPrimary.hit.geomID == p_usm_geomIDs[1])
      diffuse = p_ground_face_colors[rhPrimary.hit.primID];

    color = color + diffuse * 0.5f;
    Vec3fa lightDir = normalize(Vec3fa(-1, -1, -1));

    /* initialize shadow ray */
    RTCRay rShadow;
    Vec3fa sOrg = org + rhPrimary.ray.tfar * dir;
    rShadow.dir_x = -lightDir.x;
    rShadow.dir_y = -lightDir.y;
    rShadow.dir_z = -lightDir.z;
    rShadow.org_x = sOrg.x;
    rShadow.org_y = sOrg.y;
    rShadow.org_z = sOrg.z;
    rShadow.tnear = 0.001f;
    rShadow.time = 0.0f;
    rShadow.tfar = std::numeric_limits<float>::infinity();
    rShadow.mask = -1;

    RTCOccludedArguments oargs;
    rtcInitOccludedArguments(&oargs);
    oargs.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;
    oargs.feature_mask = RTC_FEATURE_FLAG_TRIANGLE;
    /* trace shadow ray */
    rtcOccluded1(scene, &rShadow, &oargs);

    /* add light contribution */
    if (rShadow.tfar >= 0.0f) {
      Vec3fa Ng =
          Vec3fa(rhPrimary.hit.Ng_x, rhPrimary.hit.Ng_y, rhPrimary.hit.Ng_z);
      color =
          color + diffuse * clamp(-dot(lightDir, normalize(Ng)), 0.0f, 1.0f);
    }
  }

  /* write color to framebuffer */
  unsigned char r = (unsigned char)(255.0f * clamp(color.x, 0.0f, 1.0f));
  unsigned char g = (unsigned char)(255.0f * clamp(color.y, 0.0f, 1.0f));
  unsigned char b = (unsigned char)(255.0f * clamp(color.z, 0.0f, 1.0f));
  pixels[y * width * channels + x * channels] = r;
  pixels[y * width * channels + x * channels + 1] = g;
  pixels[y * width * channels + x * channels + 2] = b;
}

/* called by the C++ code to render */
void renderFrameStandard(sycl::queue* queue, unsigned char* pixels,
                         const unsigned int width, const unsigned int height,
                         const unsigned int channels, const float time,
                         Camera* p_camera, const RTCScene& scene,
                         Vec3fa* p_cube_face_colors,
                         Vec3fa* p_ground_face_colors,
                         unsigned int* p_usm_geomIDs) {
  sycl::event event = queue->submit([=](sycl::handler& cgh) {
      cgh.set_specialization_constant<feature_mask>(required_features);
    const sycl::nd_range<2> nd_range = make_nd_range(width, height);
    cgh.parallel_for(
        nd_range, [=](sycl::nd_item<2> item, sycl::kernel_handler kh) {
          const unsigned int x = item.get_global_id(0);
          if (x >= width) return;
          const unsigned int y = item.get_global_id(1);
          if (y >= height) return;
          const RTCFeatureFlags features =
              kh.get_specialization_constant<feature_mask>();
          renderPixelStandard(x, y, pixels, width, height, channels, time,
                              p_camera, features, scene, p_cube_face_colors,
                              p_ground_face_colors, p_usm_geomIDs);
        });
  });
  event.wait_and_throw();
}

/* called by the C++ code for cleanup */
void device_cleanup(const sycl::queue& queue, RTCScene* p_scene,
                    Vec3fa** pp_cube_face_colors,
                    Vec3fa** pp_cube_vertex_colors,
                    Vec3fa** pp_ground_face_colors,
                    Vec3fa** pp_ground_vertex_colors) {
  rtcReleaseScene(*p_scene);
  *p_scene = nullptr;
  if (*pp_cube_face_colors) alignedSYCLFree(queue, *pp_cube_face_colors);
  *pp_cube_face_colors = nullptr;
  if (*pp_cube_vertex_colors) alignedSYCLFree(queue, *pp_cube_vertex_colors);
  *pp_cube_vertex_colors = nullptr;
  if (*pp_ground_face_colors) alignedSYCLFree(queue, *pp_ground_face_colors);
  *pp_ground_face_colors = nullptr;
  if (*pp_ground_vertex_colors)
    alignedSYCLFree(queue, *pp_ground_vertex_colors);
  *pp_ground_vertex_colors = nullptr;
}

/*
 * Create a scene, which is a collection of geometry objects. Scenes are
 * what the intersect / occluded functions work on. You can think of a
 * scene as an acceleration structure, e.g. a bounding-volume hierarchy.
 *
 * Scenes, like devices, are reference-counted.
 */
RTCScene initializeScene(
    const sycl::queue& queue, const RTCDevice& device,
    Vec3fa** pp_cube_face_colors, Vec3fa** pp_cube_vertex_colors,
    Vec3fa** pp_ground_face_colors, Vec3fa** pp_ground_vertex_colors,
    std::vector<unsigned int, USMDRO_UI_ALLOC>* p_geomIDs) {
  RTCScene scene = rtcNewScene(device);

  /* add cube */
  p_geomIDs->push_back(addCube(queue, device, scene, pp_cube_face_colors,
                               pp_cube_vertex_colors));

  /* add ground plane */
  p_geomIDs->push_back(addGroundPlane(
      queue, device, scene, pp_ground_face_colors, pp_ground_vertex_colors));

  /*
   * Like geometry objects, scenes must be committed. This lets
   * Embree know that it may start building an acceleration structure.
   */
  rtcCommitScene(scene);

  return scene;
}

RTCDevice initializeDevice(sycl::context& sycl_context,
                           sycl::device& sycl_device) {
  RTCDevice device = rtcNewSYCLDevice(sycl_context, "");

  if (!device)
    printf("fail: error %d: cannot create device\n", rtcGetDeviceError(NULL));

  rtcSetDeviceErrorFunction(device, error_handler, NULL);
  return device;
}

void enablePersistentJITCache() {
#if defined(_WIN32)
  _putenv_s("SYCL_CACHE_PERSISTENT", "1");
  _putenv_s("SYCL_CACHE_DIR", "cache");
#else
  setenv("SYCL_CACHE_PERSISTENT", "1", 1);
  setenv("SYCL_CACHE_DIR", "cache", 1);
#endif
}

int main() {
  enablePersistentJITCache();

  Vec3fa* p_cube_face_colors = nullptr;
  Vec3fa* p_cube_vertex_colors = nullptr;
  Vec3fa* p_ground_face_colors = nullptr;
  Vec3fa* p_ground_vertex_colors = nullptr;

  /* The allocator helps us use USM pointers for the vector container */
  /* Acheived:
   *   1. Dynamic element creation (push_back) into USM memory
   *   2. USM pointer access in the device kernel
   *   3. Device knows this is "read only" memory, thus there is no transfer back upon kernel
   *    completion.*/
  std::vector<unsigned int, USMDRO_UI_ALLOC>* p_geomIDs = nullptr;

  RTCDevice emb_device = nullptr;
  RTCScene emb_scene = nullptr;

  unsigned char* p_pixels;

  Camera* p_camera = nullptr;

  /* This will select the first GPU supported by Embree */
  sycl::device sycl_device(rtcSYCLDeviceSelector);
  sycl::queue sycl_queue(sycl_device);
  sycl::context sycl_context(sycl_device);

  /* Creation of the USM allocator object uses arguments of the queue, and the needed
   * OneAPI usm property */
  USMDRO_UI_ALLOC qalloc(sycl_queue,
                         sycl::ext::oneapi::property::usm::device_read_only());
  /* The std::vector object is created using the allocator */
  p_geomIDs = new std::vector<unsigned int, USMDRO_UI_ALLOC>(qalloc);

  emb_device = initializeDevice(sycl_context, sycl_device);
  emb_scene = initializeScene(sycl_queue, emb_device, &p_cube_face_colors,
                              &p_cube_vertex_colors, &p_ground_face_colors,
                              &p_ground_vertex_colors, p_geomIDs);

  /* std::vector says memory is linear, thus an address to element 0 is not a
   * usm address */
  /* The USM address is used in the kernel */
  unsigned int* p_usm_geomIDs = &((*p_geomIDs)[0]);

  /* create an image buffer initialize it with all zeroes */
  const unsigned int width = 320;
  const unsigned int height = 200;
  const unsigned int channels = 3;
  p_pixels = alignedSYCLUSMMalloc<unsigned char>(sycl_queue,
                                                 width * height * channels, 64);
  std::memset(p_pixels, 0, sizeof(unsigned char) * width * height * channels);
  p_camera = alignedSYCLUSMMallocDeviceReadOnly<Camera>(sycl_queue, 1, 64);

  *p_camera = positionCamera(Vec3fa(1.5f, 1.5f, -1.5f), Vec3fa(0, 0, 0),
                             Vec3fa(0, 1, 0), 90.0f, width, height);

  renderFrameStandard(&sycl_queue, p_pixels, width, height, channels, 0.0,
                      p_camera, emb_scene, p_cube_face_colors,
                      p_ground_face_colors, p_usm_geomIDs);
  stbi_write_png("rkRayTracerGPU.png", width, height, channels, p_pixels,
                 width * channels);

  if (p_pixels) {
    alignedSYCLFree(sycl_queue, p_pixels);
    p_pixels = nullptr;
  }
  if (p_geomIDs) {
    delete p_geomIDs;
    p_geomIDs = nullptr;
  }
  if (p_camera) {
    alignedSYCLFree(sycl_queue, p_camera);
    p_camera = nullptr;
  }

  device_cleanup(sycl_queue, &emb_scene, &p_cube_face_colors,
                 &p_cube_vertex_colors, &p_ground_face_colors,
                 &p_ground_vertex_colors);
  rtcReleaseDevice(emb_device);
  printf("success\n");
  return 0;
}
