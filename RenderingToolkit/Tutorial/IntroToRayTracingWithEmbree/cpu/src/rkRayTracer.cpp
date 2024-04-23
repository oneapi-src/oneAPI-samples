#ifdef _MSC_VER
#ifndef NOMINMAX
/* use the c++ library min and max instead of the MSVS macros */
/* rkcommon will define this macro upon CMake import */
#define NOMINMAX
#endif
#endif

#include <embree4/rtcore.h>
#include <rkcommon/math/vec.h>
#include <rkcommon/memory/malloc.h>
#include <tbb/parallel_for.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

using Vec3fa = rkcommon::math::vec_t<float, 3, 1>;
using rkcommon::math::cross;
using rkcommon::math::deg2rad;
using rkcommon::math::normalize;
using std::max;
using std::min;

#ifdef _WIN32
#define alignedMalloc(a, b) _aligned_malloc(a, b)
#define alignedFree(a) _aligned_free(a)
#else
#include <mm_malloc.h>
#define alignedMalloc(a, b) _mm_malloc(a, b)
#define alignedFree(a) _mm_free(a)
#endif

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8

struct LinearSpace3 {
  Vec3fa vx, vy, vz;
};

typedef struct Affine3fa {
  LinearSpace3 l;
  Vec3fa p;
} ISPCCamera;

Vec3fa* g_cube_face_colors = nullptr;
Vec3fa* g_cube_vertex_colors = nullptr;
Vec3fa* g_ground_face_colors = nullptr;
Vec3fa* g_ground_vertex_colors = nullptr;

std::vector<unsigned int> geomIDs;

RTCDevice g_device = nullptr;
RTCScene g_scene = nullptr;

unsigned char* g_pixels;

ISPCCamera g_camera;

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

ISPCCamera positionCamera(Vec3fa from, Vec3fa to, Vec3fa up, float fov,
                          size_t width, size_t height) {
  /* There are many ways to set up a camera projection. This one is consolidated
   * from the camera code in the Embree/tutorial/common/tutorial/camera.h object
   */
  ISPCCamera camMatrix;
  Vec3fa Z =
      rkcommon::math::normalize(rkcommon::math::vec_t<float, 3, 1>(to - from));
  Vec3fa U = rkcommon::math::normalize(
      rkcommon::math::cross(rkcommon::math::vec_t<float, 3, 1>(up),
                            rkcommon::math::vec_t<float, 3, 1>(Z)));
  Vec3fa V = rkcommon::math::normalize(
      rkcommon::math::cross(rkcommon::math::vec_t<float, 3, 1>(Z),
                            rkcommon::math::vec_t<float, 3, 1>(U)));
  camMatrix.l.vx = U;
  camMatrix.l.vy = V;
  camMatrix.l.vz = Z;
  camMatrix.p = from;

  /* negate for a right handed camera*/
  camMatrix.l.vx = -camMatrix.l.vx;

  const float fovScale = 1.0f / tanf(rkcommon::math::deg2rad(0.5f * fov));

  camMatrix.l.vz = -0.5f * width * camMatrix.l.vx +
                   0.5f * height * camMatrix.l.vy +
                   0.5f * height * fovScale * camMatrix.l.vz;
  camMatrix.l.vy = -camMatrix.l.vy;

  return camMatrix;
}

/* adds a cube to the scene */
unsigned int addCube(RTCScene _scene) {
  /* create a triangulated cube with 12 triangles and 8 vertices */
  RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  g_cube_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 12, 16);
  g_cube_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 8, 16);

  /* set vertices and vertex colors */
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 8);
  g_cube_vertex_colors[0] = Vec3fa(0, 0, 0);
  vertices[0].x = -1;
  vertices[0].y = -1;
  vertices[0].z = -1;
  g_cube_vertex_colors[1] = Vec3fa(0, 0, 1);
  vertices[1].x = -1;
  vertices[1].y = -1;
  vertices[1].z = +1;
  g_cube_vertex_colors[2] = Vec3fa(0, 1, 0);
  vertices[2].x = -1;
  vertices[2].y = +1;
  vertices[2].z = -1;
  g_cube_vertex_colors[3] = Vec3fa(0, 1, 1);
  vertices[3].x = -1;
  vertices[3].y = +1;
  vertices[3].z = +1;
  g_cube_vertex_colors[4] = Vec3fa(1, 0, 0);
  vertices[4].x = +1;
  vertices[4].y = -1;
  vertices[4].z = -1;
  g_cube_vertex_colors[5] = Vec3fa(1, 0, 1);
  vertices[5].x = +1;
  vertices[5].y = -1;
  vertices[5].z = +1;
  g_cube_vertex_colors[6] = Vec3fa(1, 1, 0);
  vertices[6].x = +1;
  vertices[6].y = +1;
  vertices[6].z = -1;
  g_cube_vertex_colors[7] = Vec3fa(1, 1, 1);
  vertices[7].x = +1;
  vertices[7].y = +1;
  vertices[7].z = +1;

  /* set triangles and face colors */
  int tri = 0;
  Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 12);

  // left side
  g_cube_face_colors[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 1;
  triangles[tri].v2 = 2;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 2;
  tri++;

  // right side
  g_cube_face_colors[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 4;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 5;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 5;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 7;
  tri++;

  // bottom side
  g_cube_face_colors[tri] = Vec3fa(0.5f);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 1;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(0.5f);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 5;
  tri++;

  // top side
  g_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 6;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 7;
  triangles[tri].v2 = 6;
  tri++;

  // front side
  g_cube_face_colors[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 2;
  triangles[tri].v2 = 4;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 4;
  tri++;

  // back side
  g_cube_face_colors[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 3;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 7;
  tri++;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, g_cube_vertex_colors, 0,
                             sizeof(Vec3fa), 8);

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(_scene, mesh);
  rtcReleaseGeometry(mesh);
  return geomID;
}

/* adds a ground plane to the scene */
unsigned int addGroundPlane(RTCScene _scene) {
  /* create a triangulated plane with 2 triangles and 4 vertices */
  RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  g_ground_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 2, 16);
  g_ground_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 4, 16);

  /* set vertices */
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 4);
  g_ground_vertex_colors[0] = Vec3fa(1, 0, 0);
  vertices[0].x = -10;
  vertices[0].y = -2;
  vertices[0].z = -10;
  g_ground_vertex_colors[1] = Vec3fa(1, 0, 1);
  vertices[1].x = -10;
  vertices[1].y = -2;
  vertices[1].z = +10;
  g_ground_vertex_colors[2] = Vec3fa(1, 1, 0);
  vertices[2].x = +10;
  vertices[2].y = -2;
  vertices[2].z = -10;
  g_ground_vertex_colors[3] = Vec3fa(1, 1, 1);
  vertices[3].x = +10;
  vertices[3].y = -2;
  vertices[3].z = +10;

  /* set triangles */
  Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 2);

  g_ground_face_colors[0] = Vec3fa(1, 0, 0);
  triangles[0].v0 = 0;
  triangles[0].v1 = 1;
  triangles[0].v2 = 2;
  g_ground_face_colors[1] = Vec3fa(1, 0, 0);
  triangles[1].v0 = 1;
  triangles[1].v1 = 3;
  triangles[1].v2 = 2;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, g_ground_vertex_colors, 0,
                             sizeof(Vec3fa), 4);

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(_scene, mesh);
  rtcReleaseGeometry(mesh);
  return geomID;
}

/* task that renders a single screen pixel */
void renderPixelStandard(int x, int y, unsigned char* pixels,
                         const unsigned int width, const unsigned int height,
                         const unsigned int channels, const float time,
                         const ISPCCamera& camera) {

  /* RTCIntersectArguments is new for Embree 4 */
  RTCIntersectArguments iargs;
  rtcInitIntersectArguments(&iargs);
  iargs.feature_mask = RTC_FEATURE_FLAG_TRIANGLE;
  iargs.flags = RTC_RAY_QUERY_FLAG_COHERENT;

  const Vec3fa dir = rkcommon::math::normalize(x * camera.l.vx +
                                               y * camera.l.vy + camera.l.vz);
  const Vec3fa org = Vec3fa(camera.p.x, camera.p.y, camera.p.z);

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
  rtcIntersect1(g_scene, &rhPrimary, &iargs);

  /* shade pixels */
  Vec3fa color = Vec3fa(0.0f);
  if (rhPrimary.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    Vec3fa diffuse;

    if (rhPrimary.hit.geomID == geomIDs[0])
      diffuse = g_cube_face_colors[rhPrimary.hit.primID];
    else if (rhPrimary.hit.geomID == geomIDs[1])
      diffuse = g_ground_face_colors[rhPrimary.hit.primID];

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

    /* New for Embree 4 */
    RTCOccludedArguments oargs;
    rtcInitOccludedArguments(&oargs);
    oargs.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;
    oargs.feature_mask = RTC_FEATURE_FLAG_TRIANGLE;

    /* trace shadow ray */
    rtcOccluded1(g_scene, &rShadow, &oargs);

    /* add light contribution */
    if (rShadow.tfar >= 0.0f) {
      Vec3fa Ng =
          Vec3fa(rhPrimary.hit.Ng_x, rhPrimary.hit.Ng_y, rhPrimary.hit.Ng_z);
      color =
          color + diffuse * rkcommon::math::clamp(
                                -rkcommon::math::dot(lightDir, normalize(Ng)),
                                0.0f, 1.0f);
    }
  }

  /* write color to framebuffer */
  unsigned char r =
      (unsigned char)(255.0f * rkcommon::math::clamp(color.x, 0.0f, 1.0f));
  unsigned char g =
      (unsigned char)(255.0f * rkcommon::math::clamp(color.y, 0.0f, 1.0f));
  unsigned char b =
      (unsigned char)(255.0f * rkcommon::math::clamp(color.z, 0.0f, 1.0f));
  pixels[y * width * channels + x * channels] = r;
  pixels[y * width * channels + x * channels + 1] = g;
  pixels[y * width * channels + x * channels + 2] = b;
}

/* task that renders a single screen tile */
void renderTileTask(int taskIndex, int threadIndex, unsigned char* pixels,
                    const unsigned int width, const unsigned int height,
                    const unsigned int channels, const float time,
                    const ISPCCamera& camera, const int numTilesX,
                    const int numTilesY) {
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0 + TILE_SIZE_X, width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0 + TILE_SIZE_Y, height);

  for (unsigned int y = y0; y < y1; y++)
    for (unsigned int x = x0; x < x1; x++) {
      renderPixelStandard(x, y, pixels, width, height, channels, time, camera);
    }
}

/* called by the C++ code to render */
void renderFrameStandard(unsigned char* pixels, const unsigned int width,
                         const unsigned int height, const unsigned int channels,
                         const float time, const ISPCCamera& camera) {
  const int numTilesX = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
  const int numTilesY = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
  tbb::task_group_context tgContext;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, numTilesX * numTilesY, 1),
      [&](const tbb::blocked_range<size_t>& r) {
        const int threadIndex = tbb::this_task_arena::current_thread_index();
        for (size_t i = r.begin(); i < r.end(); i++)
          renderTileTask((int)i, threadIndex, pixels, width, height, channels,
                         time, camera, numTilesX, numTilesY);
      },
      tgContext);
  if (tgContext.is_group_execution_cancelled())
    throw std::runtime_error("fail: oneTBB task cancelled");
}

/* called by the C++ code for cleanup */
void device_cleanup() {
  rtcReleaseScene(g_scene);
  g_scene = nullptr;
  if (g_cube_face_colors) alignedFree(g_cube_face_colors);
  g_cube_face_colors = nullptr;
  if (g_cube_vertex_colors) alignedFree(g_cube_vertex_colors);
  g_cube_vertex_colors = nullptr;
  if (g_ground_face_colors) alignedFree(g_ground_face_colors);
  g_ground_face_colors = nullptr;
  if (g_ground_vertex_colors) alignedFree(g_ground_vertex_colors);
  g_ground_vertex_colors = nullptr;
}

void device_init(char* cfg) {
  /* create scene */
  g_scene = nullptr;
  g_scene = rtcNewScene(g_device);

  /* add cube */
  geomIDs.push_back(addCube(g_scene));

  /* add ground plane */
  geomIDs.push_back(addGroundPlane(g_scene));

  /* commit changes to scene */
  rtcCommitScene(g_scene);
}

int main() {
  /* create device */
  g_device = rtcNewDevice(nullptr);
  error_handler(nullptr, rtcGetDeviceError(g_device),
                "fail: Embree Error Unable to create embree device");

  /* set error handler */
  rtcSetDeviceErrorFunction(g_device, error_handler, nullptr);

  device_init(nullptr);

  /* create an image buffer initialize it with all zeroes */
  const unsigned int width = 320;
  const unsigned int height = 200;
  const unsigned int channels = 3;
  g_pixels = (unsigned char*)new unsigned char[width * height * channels];
  std::memset(g_pixels, 0, sizeof(unsigned char) * width * height * channels);

  g_camera = positionCamera(Vec3fa(1.5f, 1.5f, -1.5f), Vec3fa(0, 0, 0),
                            Vec3fa(0, 1, 0), 90.0f, width, height);

  renderFrameStandard(g_pixels, width, height, channels, 0.0, g_camera);
  stbi_write_png("rkRayTracer.png", width, height, channels,
                 g_pixels, width * channels);

  delete[] g_pixels;
  g_pixels = nullptr;
  device_cleanup();
  rtcReleaseDevice(g_device);
  printf("success\n");
  return 0;
}
