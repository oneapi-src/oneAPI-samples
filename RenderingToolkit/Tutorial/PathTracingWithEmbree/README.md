# Path Tracing with Intel&reg; Embree and the Intel&reg; oneAPI Rendering Toolkit

## Purpose

By walking through this pathtracer tutorial, you will understand how to implement a basic pathtracer with Intel Embree. You will have a better understanding of:
- A pathtracer renderer such as the one implemented within Intel OSPRay.
- Visual quality considerations when implementing applications with Embree.
- typical scene controls of interest in the Intel OSPRay Studio showcase reference application or other professional rendering solutions.
- full feature tutorial codes on the Intel Embree github repository.

Expect less than 10 minutes to compile and run the application on contemporary hardware. 

Expect at least an hour for following the algorithm, self-directed edits, then to rebuild, rerun, and understand the application.

Example output images from this `rkPathTracer` tutorial program:

We present a series of two walk through paths.
1. A _key features_ description of the logical additions to this application to make it a monte carlo path tracer.
2. A _source code_ walk through describing how the features are implemented with Embree API calls.

This sample source code is a consolidated refactor of the `pathtracer` source hosted as part of the Intel&reg; Embree tutorials in the [Embree
repository](https://github.com/embree/embree) on GitHub.

## Prerequisites

| Minimum Requirements | Description                                                                                                                                                                                                                     |
|:-------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OS                   | Linux* Ubuntu* 18.04 <br>CentOS* 8 (or compatible) <br>Windows* 10 <br>macOS* 10.15+                                                                                                                                            |
| Hardware             | Intel 64 Penryn or higher with SSE4.1 extensions; ARM64 with NEON extensions <br>(Optimization requirement: Intel&reg; Embree is further optimized for Intel 64 Skylake or higher with AVX512 extensions)                       |
| Compiler Toolchain   | Windows* OS: MSVS 2019 or MSVS 2022 with Windows* SDK and CMake* <br>Other platforms: C++11 compiler and CMake*                                                                                                                 |
| Libraries            | Install Intel&reg; oneAPI Rendering Toolkit (Render Kit) to get Intel&reg; Embree and Intel&reg; oneAPI Threading Building Blocks (oneTBB) <br>Install Intel&reg; oneAPI Base Toolkit for the `dev-utilities` default component |
| Tools                | .png capable image viewer                                                                                                                                                                                                       |

## Build and Run

### Windows*

Open an x64 Native Tools Command Prompt for VS 2019 (or 2022).

Set toolkit environment variables. Ex:

```
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Build and run the application:

```
mkdir build
cd build
cmake -G"Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
cd Release
.\rkPathTracer.exe
```

**Note**: Visual Studio 2022 users should use the `-G"Visual Studio 17 2022"`
generator flag.

Open the resulting .png image files with an image viewer.

Examine the source as desired with the IDE:

```
devenv rkPathTracer.sln
```

### Linux* or macOS*

Open a new terminal

Set toolkit environment variables. Ex:

```
source /opt/intel/oneapi/setvars.sh
```

Build and run the application:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
./rkPathTracer
```

Open the resulting .png image files with an image viewer.

## Ray Tracer Introduction Recap

In the Introduction to Ray Tracing with Intel Embree sample, we walked through a program that found basic albedo (surface color) and basic lighting for a cube and a plane.

The output image was generated via rays cast from each pixel of our camera to determine color at the corresponding scene geometry intersection. These initial rays from the camera are the primary rays in our scene.
Then, if and only if the intersected surface was not occluded from the light position, the albedo would be increased to full strength. The difference in albedo intensity gave the impression of a shadow on the plane underneath the cube. 
appearance of depth with a shadow and an idea of distance for the cube hovering above our plane. 
However, this was only the first step to computing light transport used to simulate generation of a photoreal image. So far, in the `triangle_geometry` sample application, we have only simulated direct illumination.

## What is path tracing?

Path tracing implements a simulation of global illumination.

Our primary rays alone do not capture light illuminating a surface from other surfaces visible to the intersection. So, in this tutorial, we construct a multiple segment ray path. The path continues from a primary ray intersection point to sample light from all other visible surfaces and lights in the scene.
By integrating the light visible from the hemisphere above all intersections for the paths intersection, (global illumination)[https://en.wikipedia.org/wiki/Path_tracing] is simulated.

The path itself need not stop after one additional path 'segment' (a segment being a reflection or refraction). Multiple bounces can capture light from multiple surfaces. In a limited compute resource environment, we either set a constant path length or create logical heuristics limiting the path length.
Surface properties affect the light reflected. Here we introduce material properties to alter the light reflected off surfaces. Such a function is commonly referred to as a bidirectional ray distribution function (BRDF).

## Key Features Added

This program adds application level features to the `triangle_geometry` sample demonstrate a reference monte carlo path tracer program.

New features added are discussed at a high level. The features are then described with in-source implementation details. It may take a few rounds of looking at the key feature description and the source-accompanied description to understand the application.


### Feature: Added Scenes

Global illumination is difficult to demonstrate without surfaces affecting and reflecting light to visible surfaces within the scene.

To demonstrate, we have three sets of hardcoded scene data.

1. We implement Cornell Box scene geometry, as defined within Intel OSPRay for the Intel OSPRay `ospExamples` application. The Cornell Box is a familiar dataset that can show capability of computing global illumination and shadows. We have a code stub to add an extra sphere to the scene.

2. We implement the `triangle_geometry` scene. The intent with this scene is to observe the same scene in a global illumination environment.

3. We have a 'pool' scene. This scene is heavily centered on observing the behavior of a fresnel surface.

Each geometry specified allows the user to try different materials and colors. The audience is encouraged to modify and rebuild the scenes to examine and understand behavior!


### Feature: Materials
We three basic materials implemented in the sample.

1. Lambertian(https://en.wikipedia.org/wiki/Lambertian_reflectance)
   This is a soft diffuse material that scatters light proportional to the cosine of the angle of incidence in all directions. Another description often used for a Lambertian surface is: ideal _matte_ surface.

2. Mirror
   We introduce a mirror as a material surface. In our sample we give the mirror an albedo that can be used to limit reflected energy similar to imperfect real life mirrors. Limiting reflectance more closely matches the behavior of real mirrors.

3. Fresnel
   We introduce fresnel materials. These materials have reflective and refractive light transport behaviors.

The audience is ecouraged to modify material parameters to examine behavior!
   
### Feature: Light

We have 3 basic lights implemented.

1. Directional Light

This is similar to the `triangle_geometry` sample light.

2. Point Light

This light emits in all directions. It can also have a radius (making it a sphere) and corresponding geometry. It can leave soft shadows in the scene.

3. Spot light

This is a disc light that emits in a specified direction. It also has a radius and corresponding geometry. It can leave soft shadows in the scene.

### Feature: Global Illumination

We describe our core algorithm at a high (informal) level:

1. We take a color sample at each pixel of our image. The sample is to be computed from an n-segment ray emanating from a pixel. The pixel is mapped to the capture plane of a virtual camera. The ray is cast into the scene.
2. At a ray-to-scene intersection, we determine the light coming from the material at the intersection. If it is a light source we add that lights radiance based on the light type and ray direction before terminating the path.
3. Otherwise, we perform a shadow ray test from the intersection to all light emitters in the scene. We aggregate the visible radiance proportional to the type of light and distance to the surface and the material.
4. Next, we generate a bounce ray pointing to a location on the unit hemisphere above an intersection. The bounce ray direction is a function of the material. This ray represents the direction of the next segment of our *path* being traced.
5. In the case of a Lambertian surface we generate a ray segment randomly as a representative reflection of all incident light at the point. Lowhttps://stackoverflow.com/questions/22575662/filename-too-long-in-git-for-windowser ray angles (approaching tangent) will contribute very global light through the new ray segment. Higher angles will contribute more.
6. In the case of a mirror, the reflection is not random. It is computed as a mirror ray. Our mirror material rays will not attenuate based on direction. They will only attenuate based upon the inherent reflectivity of the mirror.
7. Lastly, Fresnel materials will reflect or refract rays based on a material constant (eta) and the angle of incidence. Different materials will have different constant.
8. Light attenuation (color) is evaluated based on the material and corresponding outgoing and incoming light directions from the surface. Note: The directions are referred to in source as Omega out and Omega in, *wo* and *wi*.
9. The probability distribution function is computed for the path segment. This allows us to integrate multiple samples per pixel together in a montecarlo fashion. It also appropriately attenuates radiance sampled from a given direction.
10. The light weight for subsequent path segments is computed. The weight is attenuated by material evaluated attenuation (e.g. color) and the pdf.
11. The next path segment is computed. If the path is at its maximum length the Luminance sample for the pixel is returned.

Multiple samples can be taken for each pixel. They can be used to generate an aggregate of samples per pixel. In this implementation, each sample will be taken at a random offset within the bounds of the pixel. This provides an antialiasing effect for the image.

### Feature: Accumulation buffer

- Next, The accumulation buffer stores total luminance sampled underneath all pixels over all accumulations. To write to an image file output,  we divide each pixel by number of accumulations to get our per channel color averages.
- An accumulation buffer is useful in an interactive rendering context to intermittently update a windowed frame buffer. The application can continue with a stationary scene and camera to accumulate and thus converge the image.
- In this tutorial program, we use the buffer to write the first sample of all pixels in our image before accumulating many more rendered frames (accumulations).


### Feature: Convergence 

- If we had taken an infinite number of samples with infinite path length, all paths under all sampled pixels can logically account for all light in the scene. Such a result will be a converged image.
- Practically, we will set the number of samples per pixel, set an accumulation limit, and maximum path length to the discretion of the application.
- We set these to 1, 500, and 8 in the application respectively. Change these values to see the affect noise from a limited number of samples or paths segments has on an image.


Examples:
1spp , 1 accumulation, 1 total sample
1spp, 500 accumulations, 500 total samples

### Embree functions

The following Embree API functions are used in this application:

Geometry:
- rtcNewGeometry(...)
- rtcSetNewGeometryBuffer(...)
- rtcCommitGeometry(...)
- rtcAttachGeometry(...)
- rtcReleaseGeometry(...)
- rtcSetGeometryVertexAttributeCount(...)
- rtcSetSharedGeometryBuffer(...)

Ray Queries:
- rtcIntersect1(..)
- rtcOccluded1(..)
- rtcInitIntersectContext(...)

Device:
- rtcGetDeviceError(..)
- rtcNewDevice(...)
- rtcSetDeviceErrorFunction(...)

Scene:
- rtcNewScene(...)
- rtcReleaseScene(...)

See the API pdf (manual)[https://raw.githubusercontent.com/embree/embree/master/readme.pdf] for more information (~2MB).

## Details of Improvements (Source)

It is highly advised to walk through the source in an IDE environment that allows for quick lookup of symbol and function definitions!

### rkPathTracer.cpp
In `rkPathTracer.cpp` we have `main(..)`. The application will create a Renderer to render a selected scene. The renderer first generates and writes an image with one accumulation. The render then renders all accumulations.
```
  std::unique_ptr<Renderer> r;

  // SceneSelector sceneSelector = SceneSelector::SHOW_POOL;
  SceneSelector sceneSelector = SceneSelector::SHOW_CORNELL_BOX;
  // SceneSelector sceneSelector = SceneSelector::SHOW_CUBE_AND_PLANE;
  r = std::make_unique<Renderer>(width, height, channels, spp, accu_limit,
                                 max_path_length, sceneSelector);
```
Accumulation:
```
  r->render_accumulation();
  
  ...

  /* Render all remaining accumulations (in addition to the first) */
  for (unsigned long long i = 1; i < accu_limit; i++) {
    ...

    r->render_accumulation();

    ...
  }
```

We observe the image size, number of channels, accumulation limit, number of samples per pixel and maximum number of segments to a path are all set here.

The audience is encouraged to alter these values, then review single and multiple accumulation images to understand the output.

```
  /* create an image buffer initialize it with all zeroes */
  const unsigned int width = 512;
  const unsigned int height = 512;
  const unsigned int channels = 3;
  /* Control the total number of accumulations, the total number of samples per
   * pixel per accumulation, and the maximum path length of any given traced
   * path.*/
  const unsigned long long accu_limit = 500;
  const unsigned int spp = 1;
  const unsigned int max_path_length = 8;
```
The values above are resident in higher level APIs like Intel OSPRay for controlling scenes. Look for them in the Intel OSPRay Studio showcase reference application GUI.

### Renderer.h

The renderer initializes a pixel buffer and the accumulation buffer.

```
m_pixels = (unsigned char*)new unsigned char[m_width * m_height * m_channels];
  std::memset(m_pixels, 0,
              sizeof(unsigned char) * m_width * m_height * m_channels);
```

```
  m_accu.resize(m_width * m_height);
  for (auto i = 0; i < m_width * m_height; i++)
    m_accu[i] = std::make_shared<Vec3ff>(0.0f);
```


The renderer then initializes the Embree device with `rtcNewDevice(..)`. A device error handling function is assigned for the Embree API to report any runtime errors.

```
void Renderer::init_device(const char* cfg) {
  /* create device */
  m_device = rtcNewDevice(nullptr);
  handle_error(nullptr, rtcGetDeviceError(m_device),
               "fail: Embree Error Unable to create embree device");

  /* set error handler */
  rtcSetDeviceErrorFunction(m_device, handle_error, nullptr);
}
```

The Renderer instantiates a scene management object `SceneGraph`.

```
void Renderer::init_scene(char* cfg, unsigned int width, unsigned int height) {
  m_sg = std::make_shared<SceneGraph>(m_device, m_sceneSelector, m_width,
                                      m_height);
}
```
The Renderer then creates a Path Tracer object:

```
  m_pt = std::make_shared<PathTracer>(max_path_length, m_width, m_height,
                                      m_sg->getNumLights());
```

### SceneGraph.h

The `SceneGraph` initializes the Embree scene with `rtcNewScene(..)`. The Scene Graph then will initialize Embree geometries based on the selected scene. std::map objects are used for lookup tables to find Embree geometry and primitive information when we intersect objects when ray tracing.

```
void SceneGraph::init_embree_scene(const RTCDevice device,
                                   SceneSelector SELECT_SCENE,
                                   const unsigned int width,
                                   const unsigned int height) {
  m_sceneSelector = SELECT_SCENE;
  /* create scene */
  m_scene = nullptr;
  m_scene = rtcNewScene(device);

  switch (m_sceneSelector) {
    case SceneSelector::SHOW_CUBE_AND_PLANE:
      /* add cube, add ground plane, and light */

      geometries.push_back(std::make_unique<CubeAndPlane>(
          m_scene, device, m_mapGeomToPrim, m_mapGeomToLightIdx, m_lights,
          m_camera, width, height));

...

      break;

```

We use a list of geometries (from `Geometry.h`) to keep all of our scenes automatically managed. Definition:

```
  // We'll use this 'geometries' container to automatically clean up the data
  // arrays created that are used to create embree geometries //
  std::vector<std::unique_ptr<Geometry>> geometries;
```

### Geometry.h

The `Geometry` object only serves to allow an extensible mechanism for geometry creation and destruction.

Use the `geometries` std::vector object to add, remove, or edit geometries and create your own scenes in this application.


### CornellBox.h

The Cornell box is one such derived `Geometry` object. When it is created we create Embree geometry with `rtcNewGeometry(..)` in `CornellBox::add_geometry(..)`. 

Creating the box geometry means setting a buffer of defined vertex positions for the geometry with `rtcSetNewGeometryBuffer(..)` and giving it `RTC_BUFFER_TYPE_VERTEX`.
```
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
      m_cornellBoxVertices.size());
```

Next, we setup an index buffer for the Geometry defining Quads.

```
  /* set quads */
  Quad* quads = (Quad*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0,
                                               RTC_FORMAT_UINT4, sizeof(Quad),
                                               m_cornellBoxIndices.size());
```
Lastly, we bind vertex colors to a vertex attribute with `rtcSetGeometryVertexAttributeCount(..)` and `rtcSetSharedGeometryBuffer(..)`. Vertex attributes are not used directly in this sample. They are a place holder for expanding the application to consider vertex colors with the API.
Notice that such buffers need to be aligned per the API specificiation. `alignedMalloc(..)` is used.

```
  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, *m_cornell_vertex_colors, 0,
                             sizeof(Vec3fa), m_cornellBoxVertices.size());
```

The geometry `mesh` is committed. Then it is attached to our scene. The mesh is then released.

```
  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);
```

We set up a look up map for each face's albedo (color) and assigned material. This is used later when the pathtracer intersects objects.
```
  MatAndPrimColorTable mpTable;
  mpTable.materialTable = m_cornellBoxMats;
  mpTable.primColorTable = m_cornell_face_colors;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));
```

The other `Geometry` derivative objects are highly similar. Albeit, the may use different primitive types (e.g. Quad vs Triangle)

- Note that the `Sphere` object uses an Embree defined Sphere primitive as opposed to defining many vertices manually to approximate a sphere.
- The Cornell scene also configures a light and camera associated with it. The camera view is oriented based on field of view, look at direction, up direction, width and height in pixels.

```
 float fov = 30.0f;
  float fovrad = fov * M_PI / 180.0f;
  float half_fovrad = fovrad * 0.5f;
  camera = positionCamera(Vec3fa(0.0, 0.0, -1.0f - 1.f / tanf(half_fovrad)),
                          Vec3fa(0, 0, 0), Vec3fa(0, 1, 0), fov, width, height);
```

We define a spot light with given direction, position, radius, powerm and opening angle properties. It is added to the list of lights in our scene. We create a geometry for it.

```
  /* Here we have a light as a disc geometry */
  Vec3fa spotPos(0.f, 0.95f, 0.0f);
  Vec3fa spotDir(0.f, -1.f, 0.f);
  Vec3fa spotPow = 5.f * Vec3fa(0.78f, 0.551f, 0.183f);
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
```

Adding the light geometry is similar to adding our other geometries. See Light.h for the implementation.

Each of the geometries has the ability to assign a hard coded material. Example with Cornell Box:
```
const std::vector<enum class MaterialType>
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
        ...
```

Use the enum below to try different materials for the different scenes. It is defined in `Materials.h` described next.
```
/* Added for pathtracer */
enum class MaterialType {
  MATERIAL_MATTE,
  MATERIAL_MIRROR,
  MATERIAL_GLASS,
  MATERIAL_WATER,
  MATERIAL_EMITTER
};
```

### Materials.h

`Materials.h` defines functions `sample` `eval` and `pdf` for each Material. They are for computing:
- a direction sample
- a light quantity (for color) given the sample
- a probability distribution function (PDF) value given the sample

These functions are used later in the path tracer loop.

Each `Material_` prefixed function runs a material specific code path. In this example, we use one of Lambertian (Matte), Mirror, or Dielectric (Fresnel) materials depending on the material assigned to an intersected surface.

_Sample_

Example: A matte sample given a 2D random variable:
```
Vec3fa Lambertian_sample(const Vec3fa& wo, const DifferentialGeometry& dg,
                         const Vec2f& randomMatSample) {
  return cosineSampleHemisphere(randomMatSample.x, randomMatSample.y, dg.Ns);
}
...
inline Vec3fa cosineSampleHemisphere(const float u, const float v,
                                     const Vec3fa& N) {
  /* Determine cartesian coordinate for new Vec3fa */
  const float phi = float(2.0f * M_PI) * u;
  const float cosTheta = sqrt(v);
  const float sinTheta = sqrt(1.0f - v);
  const float sinPhi = sinf(phi);
  const float cosPhi = cosf(phi);

  Vec3fa localDir = Vec3fa(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
  /* Gives the new Vec3fa transformed about the input Vec3fa */

  return frame(N) * localDir;
}

```

_Eval_

Example: appearance evaluation given a direction from a matte material:
```
inline Vec3fa Lambertian_eval(const Vec3fa& albedo, const Vec3fa& wo,
                              const DifferentialGeometry& dg,
                              const Vec3fa& wi_v) {
  /* The diffuse material. Reflectance (albedo) times the cosign fall off of the
   * vector about the normal. */
  return albedo * (1.f / (float)(float(M_PI))) * clamp(dot(wi_v, dg.Ns));
}
```

_PDF_:

Example: Probability Distribution Function for a matte material:
```
float Lambertian_pdf(const DifferentialGeometry& dg, const Vec3fa& wi1) {
  return dot(wi1, dg.Ns) / float(M_PI);
}
```


Some compute between sample, eval, and pdf functions for a material may be redundant. Obtaining direction, eval, and pdf values is separated for study purposes. In the Embree 3 repository tutorials, a `Sample` data structure is used to group pdf information with a direction.

Note that the path tracer application on the Embree repository contains implementations of several more material models.

### Lights.h

Similarly we use sample and eval functions for each of the three lights implemented.

With the sample function, a random direction is sampled from the intersection point to the light surface. Sometimes with lights that have geometries, we will see in the path tracing loop that a random direction to the surface will be obscure. This provides a soft shadow effect.

```
Light_SampleRes SpotLight::sample(const DifferentialGeometry& dg,
                                  const Vec2f& s) {
  Light_SampleRes res;

  // extant light vector from the hit point
  res.dir = m_position - dg.P;

  if (m_radius > 0.f)
    res.dir = m_coordFrame * uniformSampleDisk(m_radius, s) + res.dir;

  const float dist2 = dot(res.dir, res.dir);
  const float invdist = rsqrt(dist2);

  // normalized light vector
  res.dir = res.dir * invdist;
  res.dist = dist2 * invdist;

  // cosine of the negated light direction and light vector.
  const float cosAngle = -dot(m_coordFrame.vz, res.dir);
  const float angularAttenuation =
      clamp((cosAngle - m_cosAngleMax) * m_cosAngleScale);

  if (m_radius > 0.f)
    res.pdf = m_diskPdf * dist2 * abs(cosAngle);
  else
    res.pdf = inf;  // we always take this res

  // convert from power to radiance by attenuating by distance^2; attenuate by
  // angle
  res.weight = m_power * ((invdist * invdist) * angularAttenuation);

  return res;
}
```

With the eval function, a light value is taken given a particular direction toward the light. `SpotLight` Example:
```
Light_EvalRes SpotLight::eval(const Vec3fa& org, const Vec3fa& dir) {
  Light_EvalRes res;
  res.value = Vec3fa(0.f);
  res.dist = inf;
  res.pdf = 0.f;

  if (m_radius > 0.f) {
    // intersect disk
    const float cosAngle = -dot(dir, m_coordFrame.vz);
    if (cosAngle > m_cosAngleMax) {  // inside illuminated cone?
      const Vec3fa vp = org - m_position;
      const float dp = dot(vp, m_coordFrame.vz);
      if (dp > 0.f) {  // in front of light?
        const float t = dp * rcp(cosAngle);
        const Vec3fa vd = vp + t * dir;
        if (dot(vd, vd) < (m_radius * m_radius)) {  // inside disk?
          const float angularAttenuation =
              min((cosAngle - m_cosAngleMax) * m_cosAngleScale, 1.f);
          const float pdf = m_diskPdf * cosAngle;
          res.value =
              m_power * (angularAttenuation * pdf);  // *sqr(t)/sqr(t) cancels
          res.dist = t;
          res.pdf = pdf * (t * t);
        }
      }
    }
  }

  return res;
}
```

### Renderer.h (Accumulation)

At this point, all objects and parameters for Embree have been supplied and configured. We are now able to query and extract results from the Embree API. In `Renderer.h`, we revisit `render_accumulation(..)`.

The compute work for our image is split into image based 2D tiles. Intel&reg; oneTBB will then schedule tasks to get executed on hardware thread elements. Scheduling is based on dynamic detection of system multithreading topology.

Each `tbb::parallel_for(..)` task will get its own set of tiles to compute on. The number of tiles is derived from the oneTBB runtime. Each tile gets its own random number generator, see the definitition of a `RandomEngine` data structure.

```
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, numTilesX * numTilesY, 1),
      [&](const tbb::blocked_range<size_t>& r) {
        const int threadIndex = tbb::this_task_arena::current_thread_index();

        RandomEngine reng;
        std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

        for (size_t i = r.begin(); i < r.end(); i++) {
          render_tile_task((int)i, threadIndex, numTilesX, numTilesY, reng,
                           distrib);
        }
      },
      tgContext);
```

Performance monitors that show multithreaded residency will show usage of all hardware threads leading to a major increase to performance over a single threaded implementation.

In the `Renderer::render_tile_task(..)` function we render each ray color sample from each pixel of the image.
```
void Renderer::render_tile_task(
    int taskIndex, int threadIndex, const int numTilesX, const int numTilesY,
    RandomEngine& reng, std::uniform_real_distribution<float>& distrib) {
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0 + TILE_SIZE_X, m_width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0 + TILE_SIZE_Y, m_height);

  for (unsigned int y = y0; y < y1; y++)
    for (unsigned int x = x0; x < x1; x++) {
      Vec3fa color = render_pixel_samples(x, y, reng, distrib);
...
```

The result color is added to the accumulation buffer.

Each result color channel of the accumulation buffer is transformed to 8bit unsigned characters, and stored in the output pixel buffer.
```
      /* write color from accumulation buffer to framebuffer */
      unsigned char r =
          (unsigned char)(255.0f * clamp(accu_color.x * f, 0.0f, 1.0f));
      unsigned char g =
          (unsigned char)(255.0f * clamp(accu_color.y * f, 0.0f, 1.0f));
      unsigned char b =
          (unsigned char)(255.0f * clamp(accu_color.z * f, 0.0f, 1.0f));
      m_pixels[y * m_width * m_channels + x * m_channels] = r;
      m_pixels[y * m_width * m_channels + x * m_channels + 1] = g;
      m_pixels[y * m_width * m_channels + x * m_channels + 2] = b;
```

An accumulation buffer is useful in an interactive application where the frame buffer is blitted to screen. For example, Intel OSPRay Studio will render accumulation passes and blit each update to screen. The result is a stationary camera will allow accumulation updates to converge the result of the image. The `rkPathTracer` sample application uses the accumulation buffer for study.


### PathTracer.h (Path Loop)

The path tracing loop for each sample under each pixel for each accumulation is in `PathTracer.h`.

In `PathTracer::render_path(..)` we:

Find a direction and origin for the specified path.

```
  Vec3fa dir = sg->get_direction_from_pixel(x, y);
  Vec3fa org = sg->get_camera_origin();
```
Initialize an Embree `RTCRayHit` data structure for storage of collision information. The RTCRayHit structure stores our ray information but also our normal, barycentric coordinates for extended features like textures, as well as geometry identification information.

```
  /* initialize ray */
  RTCRayHit rayhit;
  init_RayHit(rayhit, org, dir, 0.0f, std::numeric_limits<float>::infinity(),
              m_time);

```

Initialize our aggregate path luminance and the luminance weight. The luminance weight will be attenuated as our path encounters intersections.

```

  Vec3fa L = Vec3fa(0.0f);
  Vec3fa Lw = Vec3fa(1.0f);
```

Initialize a fresnel constant for the ray origin. This is used to calculate indexes of refraction across different mediums. We use `1.f` for a vacuum.

```
  Medium medium, nextMedium;
  medium.eta = nextMedium.eta = 1.f;
```

Define a `DifferentialGeometry` data structure for storing hit information.

```
  DifferentialGeometry dg;
```
Lastly, we tell Embree to be free to optimize for coherent rays, given that first rays on the path are primary rays.

```
  sg->set_intersect_context_coherent();
```

Next, we iterate for every segment of the path. If it so happens that our remaining weight along the path is sufficiently low, we terminate the path.
```
    /* terminate if contribution too low */
    if (max(Lw.x, max(Lw.y, Lw.z)) < 0.01f) break;
```
Perform an intersection test for the path segment into the scene. If nothing is encountered we terminate the path. Otherwise, we flip the direction of the ray as we look to perform material computation with a ray leaving the material (ultimately, into our camera)
```
    if (!sg->intersect_path_and_scene(org, dir, rayhit, dg)) break;

    const Vec3fa wo = -dir;
```

Ask our SceneGraph what material the ray intersected. So, we access the lookup table constructed earlier for primitive materials.
```
    materialType =
        sg->m_mapGeomToPrim[rayhit.hit.geomID].materialTable[rayhit.hit.primID];
```

If we encounter a geometry that is a light source, we evaluate the light coming to our origin point and add that intensity to our aggregate luminance for the path. The path is terminated if it hits an emitter.
```
    if (materialType == MaterialType::MATERIAL_EMITTER) {
      std::shared_ptr<Light> light =
          sg->get_light_from_geomID(rayhit.hit.geomID);
      Light_EvalRes le = light->eval(org, dir);
      L = L + Lw * le.value;
      ...
      break;
```
If we do not encounter an emitter we look up the albedo of the material:

```
      albedo = sg->m_mapGeomToPrim[rayhit.hit.geomID]
                   .primColorTable[rayhit.hit.primID];
```
We initialize an attenuation scaling value for our material. As well as a data structure for our incoming ray (the next segment), and a random variable for sampling a direction from our material.
```
    Vec3fa c = Vec3fa(1.0f);

    Vec3fa wi1;
    Vec2f randomMatSample(distrib(reng), distrib(reng));
```


We tell Embree our rays will become incoherent (no longer primary rays)
```
sg->set_intersect_context_incoherent();
```

If the material does not simply pass light through, we perform a shadow ray test to each light. We perform a random sample on each light, perform an occlusion test to the sample, then add direction and material weighted luminance to our running total for the path.
```
    if (Material_direct_illumination(materialType)) {
      /* Cast shadow ray(s) from the hit point */
      sg->cast_shadow_rays(dg, albedo, materialType, Lw, wo, medium, m_time, L,
                           reng, distrib);
    }

    ...
    //From SceneGraph.h cast_shadow_rays
        Vec2f randomLightSample(distrib(reng), distrib(reng));
    Light_SampleRes ls = light->sample(dg, randomLightSample);

    /* If the sample probability density evaluation is 0 then no need to
     * consider this shadow ray */
    if (ls.pdf <= 0.0f) continue;

    RTCRayHit shadow;
    init_RayHit(shadow, dg.P, ls.dir, dg.eps, ls.dist, time);
    rtcOccluded1(m_scene, &m_context, &shadow.ray);
    if (shadow.ray.tfar >= 0.0f) {
      L = L + Lw * ls.weight *
                  Material_eval(albedo, materialType, Lw, wo, dg, ls.dir,
                                medium, randomLightSample);
    }

```

Next we figure our attenuation and the details for the next segment of the path.

We sample the material to find a direction based off the material properties and random value (BRDF).
```
    wi1 = Material_sample(materialType, Lw, wo, dg, medium, nextMedium,
                          randomMatSample);
```

We attenuate our 'color' evaluation for the material based on the direction sample.
```
    c = c * Material_eval(albedo, materialType, Lw, wo, dg, wi1, medium);
```


We find the monte carlo estimator proportion (probability distribution function) value for the direction sample.
```
    float nextPDF = Material_pdf(materialType, Lw, wo, dg, medium, wi1);
```

We scale our luminance weighting for subsequent segments by the current materials attenuation and the directional PDF.
```
    if (nextPDF <= 1E-4f) break;
    Lw = Lw * c / nextPDF;
```

We set the medium state for the next segment.
```
  medium = nextMedium;
```

We move slightly up from the intersection point along the unit normal. This location is used as our next origin. `dg.eps` and epsilon value is used to avoid artifacts.
```
    float sign = dot(wi1, dg.Ng) < 0.0f ? -1.0f : 1.0f;
    dg.P = dg.P + sign * dg.eps * dg.Ng;
    org = dg.P;
```

We set our next direction to be the incoming light ray to the surface. Then we initialize the `RTCRayHit` data structure for computation over the next segment in the pathtracer loop.
```
    dir = normalize(wi1);
    init_RayHit(rayhit, org, dir, dg.eps, inf, m_time);
```

Finally, when the `PathTracer::render_path(..)` function has completed. The total luminance `L` for the path is returned.
```
      return L;
```

## Performance and Quality

### Intel&reg; VTune&trade; Amplifier

Intel VTune Amplifier is a great tool when attempting to optimize your application. The hotspots mode from VTune helps us find bottlenecks in our application.

Random Number Generation

`rtcIntersect1` and `rtcOccluded1` constitute a significant portion of the compute. Looking at the other functions we see `fdlog(..)`.
`fdlog(..)` is called by the application's random number generation routines. This suggests that for this scene, and this renderer's monte carlo scheme, random number generation is also significant proportion of compute effort.
The full Embree tutorials use a hand written fast LCG based random number generator. For explanation and simplicity purposes, we use C++11 random number generators in this tutorial. The quality (recurrence, distribution, speed) of random numbers will affect the quality of your results.

For example, moving to `mt19937` based random number generators without any extra hashing gives reasonable results in single accumulation and multi-accumulation images. However, it  was observed to perform slower than a `minstd_rand` generator. This performance impact was observed in the VTune hotspots function profile. However, a quality trade off can be readily observed if the `minstd_rand` generator is not hashed as in our tutorial. Visual artifacts may be apparent even given a high sample count.
We observe random number concerns due to their impact on performance. Random number needs can be different amongst different applications.

### Fidelity And Convergence

Convergence in an image is highly dependent on scene configuration. Keeping a light at an sharp angle with respect to visible surfaces from the light can result in a noisier image. In instances where convergence is taking longer, a developer may consider a denoise pass provided from a library like Intel Open Image Denoise. Many vendors of professional rendering solutions use Intel Open Image Denoise as a final frame pass to enhance image quality.
Cornell Box at 500 samples
Cornell Box at 1000 samples
Cornell Box at 2000 samples
Intel Open Image Denoise oidnDenoise 

# Next Steps:

## Ray tracers in the wild

### Intel&reg; OSPRay: off-the-shelf renderers
The Intel OSPRay API defines renderer objects. If you are looking for an off the shelf renderer for your environment, consider any of these [renderers]( https://www.ospray.org/documentation.html#renderers)
- Pathtracer `pathtracer`:
Supports volumetric visualization, global illumination, materials, textures. Significant superset of features over this tutorial program.
- SciVis `scivis`:
Supports volumetric visualization, ambient occlusion, lights.
-AmbientOcclusion `ao`:
Supports volumetric visualization, ambient occlusion, no lights.

### Scene Graphs:
Typical rendering applications employ a scenegraph for managing complex scene data from cameras, instancing, objects, to materials, textures, and animations. This tutorial hardcodes a few basic geometric objects and materials. The full Intel Embree tutorial programs contain a basic reference [scene graph]( https://github.com/embree/embree/blob/v3.13.4/tutorials/common/tutorial/tutorial.cpp). Similarly, the OSPRay ospExamples viewer employs a reference scene graph. Lastly, Intel OSPRay Studio uses it’s own reference scene graph.
Each implementation can be a good reference point in building out your own scene graph in a production application. When moving beyond sandbox programs, software scalability challenges make a scene graph a practical necessity.


### Lights

Our lights are limited. This keeps tutorial code simple. Consider adding different lights. See the full Intel Embree tutorials demonstrate a few different [light types to try](https://github.com/embree/embree/tree/v3.13.4/tutorials/common/lights)

### Materials omissions

Bi-directional Ray Distribution Function (BRDF) parameterization

The full Embree tutorial application passes a structure representing a Bi-directional Ray Distribution Function (BRDF) representing reflectance model parameters. Our application hard codes reflection models in the interest of simplicity. However, the full tutorial application includes a BRDF parser for use with geometric models stored on disk in .obj/.mtl format.
For example: when reviewing Embree repository codes, notice that parameters associated with the [Phong](https://en.wikipedia.org/wiki/Phong_reflection_model) reflection model are considered.

`pathtracer_device.cpp` from the Embree tutorials features additional hand coded material types.

### Texturing

We do not cover texturing in this sample. Review the Embree repository source to see a demonstration of applying textures.

### Transparency

Transparent materials are omitted in this tutorial. The full Intel Embree path tracer demonstrates [transparency]( https://github.com/embree/embree/blob/v3.13.4/tutorials/pathtracer/pathtracer_device.cpp#L1630) in materials. 

## More Information
You can find more information at the [ Intel oneAPI Rendering Toolkit portal ](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).

# License

Code samples are licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.
