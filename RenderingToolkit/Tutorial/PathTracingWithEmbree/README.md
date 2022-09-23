# Path Tracing with Intel&reg; Embree

## Purpose

By walking through this pathtracer tutorial, you will understand how to implement a basic pathtracer with Intel Embree. You will have a better understanding of:
- A pathtracer renderer such as the one implemented within Intel OSPRay.
- Visual quality considerations when implementing applications with Embree.
- typical scene controls of interest in the Intel OSPRay Studio showcase reference application or other professional rendering solutions.
- full feature tutorial codes on the Intel Embree github repository.

Expect less than 10 minutes to compile and run the application on contemporary hardware. 

Expect at least an hour for following the algorithm, self-directed edits, then to rebuild, rerun, and understand the application.

Some example output images from this `rkPathTracer` tutorial program:

This sample source code is a consolidated refactor of the `pathtracer` source hosted as part of the Intel&reg; Embree tutorials in the [Embree
repository](https://github.com/embree/embree) on GitHub.

## Prerequisites

| Minimum Requirements | Description                                                                                                                                                                                                                     |
|:-------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OS                   | Linux* Ubuntu* 18.04 <br>CentOS* 8 (or compatible) <br>Windows* 10 <br>macOS* 10.15+                                                                                                                                            |
| Hardware             | Intel 64 Penryn or higher with SSE4.1 extensions; ARM64 with NEON extensions <br>(Optimization requirement: Intel&reg; Embree is further optimized for Intel 64 Skylake or higher with AVX512 extensions)                       |
| Compiler Toolchain   | Windows* OS: MSVS 2019 or MSVS 2022 with Windows* SDK and CMake* <br>Other platforms: C++11 compiler and CMake*                                                                                                                 |
| Libraries            | Install Intel&reg; oneAPI Rendering Toolkit (Render Kit), including Intel&reg; Embree and Intel® oneAPI Threading Building Blocks (oneTBB) <br>Install Intel&reg; oneAPI Base Toolkit for the `dev-utilities` default component |
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

Path tracing implments a simulation of global illumination.

Our primary rays alone do not capture light illuminating a surface from other surfaces visible to the intersection. So, in this tutorial, we construct a multiple segment ray path. The path continues from a primary ray intersection point to sample light from all other visible surfaces and lights in the scene.
By integrating the light visible from the hemisphere above all intersections for the paths intersection, (global illumination)[https://en.wikipedia.org/wiki/Path_tracing] is simulated.

The path itself need not stop after one additional path 'segment' (a segment being a reflection or refraction). Multiple bounces can capture light from multiple surfaces. In a limited compute resource environment, we either set a constant path length or create logical heuristics limiting the path length.
Surface properties affect the light reflected. Here we introduce material properties to alter the light reflected off surfaces. Such a function is commonly referred to as a bidirectional ray distribution function (BRDF).

## Key Features Added

New features added to the ray tracer `triangle_geometry` sample are discussed at a high level. The features are then highlighted with in-source implementation details. It may take a few rounds of looking at the high level description and then the source-accompanied description to understand the application.



### Added Scenes

Global illumination is difficult to demonstrate without surfaces affecting and reflecting light to visible surfaces within the scene.

To demonstrate, we have three sets of hardcoded scene data.

1. We implement Cornell Box scene geometry, as defined within Intel OSPRay for the Intel OSPRay `ospExamples` application. The Cornell Box is a familiar dataset that can show capability of computing global illumination and shadows. We have a code stub to add an extra sphere to the scene.

2. We implement the `trangle_geometry` scene. The intent with this scene is to observe the same scene in a global illumination environment.

3. We have a 'pool' scene. This scene is heavily centered on observing the behavior of a fresnel surface.

The audience is ecouraged to modify and rebuild the scenes to examine and understand behavior!


### Materials
We three basic materials implemented in the sample.

1) Lambertian(https://en.wikipedia.org/wiki/Lambertian_reflectance)
   This is a soft diffuse material that scatters light proportional to the cosine of the angle of incidence in all directions. Another word often used to describe a Lambertian surface is a *matte*� surface.

2) Mirror
   We introduce a mirror as a material surface. In our sample we give the mirror an albedo that can be used to limit reflected energy similar to imperfect real life mirrors. Limiting reflectance more closely matches the behavior of real mirrors.

3) Fresnel
   We introduce fresnel materials. These materials have reflective and refractive light transport behaviors.

The audience is ecouraged to modify material parameters to examine behavior!
   
### Light

We have 3 basic lights implemented.

1. Directional Light

This is similar to the `triangle_geometry` sample light.

2. Point Light

This light emits in all directions. It can also have a radius (making it a sphere) and corresponding geometry. It can leave soft shadows in the scene.

3. Spot light

This is a disc light that emits in a specified direction. It also has a radius and corresponding geometry. It can leave soft shadows in the scene.

### Global Illumination

We describe our core algorithm at a high (imformal) level:

1. We take a color sample at each pixel of our image. The sample is to be computed from an n-segment ray eminating from a pixel. The pixel is mapped to the capture plane of a virtual camera. The ray is cast into the scene.
2. At a ray-to-scene intersection, we determine the light coming from the material at the intersection. If it is a light source we add that lights radiance based on the light type and ray direction before terminating the path.
2. Otherwise, we perform a shadow ray test from the intersection to all light emitters in the scene. We aggregate the visible radiance proportional to the type of light and distance to the surface and the material.
2. Next, we generate a bounce ray pointing to a location on the unit hemisphere above an intersection. The bounce ray direction is a function of the material. This ray represents the direction of the next segment of our *path* being traced.
3. In the case of a Lambertian surface we generate a ray segment randomly as a representative reflection of all incident light at the point. Lowhttps://stackoverflow.com/questions/22575662/filename-too-long-in-git-for-windowser ray angles (approaching tangent) will contribute very global light through the new ray segment. Higher angles will contribute more.
4. In the case of a mirror, the reflection is not random. It is computed as a mirror ray. Our mirror material rays will not attenuate based on direction. They will only attenuate based upon the inherent reflectivity of the mirror.
5. Lastly, Fresnel materials will reflect or refract rays based on a material constant (eta) and the angle of incidence. Different materials will have different constant. 
6. Light attenuation (color) is evaluated based on the material and corresponding outgoing and incoming light directions from the surface. Note: The directions are referred to in source as Omega out and Omega in, *wo* and *wi*.
7. The probability distribution function is computed for the path segment. This allows us to integrate multiple samples per pixel together in a montecarlo fashion. It also appropriately attenuates radiance sampled from a given direction.
8. The light weight for subsequent path segments is computed. The weight is attenuated by material evaluated attentuation (e.g. color) and the pdf.
9. The next path segment is computed. If the path is at its maximum length the Luminance sample for the pixel is returned.

Mulitple samples can be taken for each pixel. They can be used to generate an aggregate of samples per pixel. In this implementation, each sample will be taken at a random offset within the bounds of the pixel. This provides an antialiasing effect for the image.

### Accumulation buffer

Next, The accumulation buffer stores total luminance sampled underneath all pixels over all accumulations. To write to an image file output,  we divide each pixel by number of accumulations to get our per channel color averages.
An accumulation buffer is useful in an interactive rendering context to intermittently update a windowed frame buffer. The application can continue with a stationary scene and camera to accumulate and thus converge the image.
In this tutorial program, we use it to write the first sample of all pixels in our image before accumulating many more rendered frames. This is useful for visual debug when developing an application.
The Luminance samples are stored added to an accumulation buffer. The accumulation buffer is divided by number of accumulations to give red green and blue pixel triplets. The application writes an image to disk.
 
If we had taken an infinite number of samples with infinite path length, all paths under all sampled pixels can logically account for all light in the scene. Such a result will be a converged image. Practically, we will set the number of samples per pixel, set an accumulation limit, and maximum path length to the discretion of the application. We set these to 1, 500, and 8 in the application respectively. Change these values to see the affect noise from a limited number of samples or paths segments has on an image.


Examples:
1spp , 1 accumulation
1spp, 500 accumulations

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

### Scene Selection
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

### Renderer

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

### Scene

The `SceneGraph` initializes the Embree scene with `rtcNewScene(..)`. The Scene Graph then will intialize Embree geometries based on the selected scene. std::map objects are used for lookup tables to find Embree geometry and primitive information when we intersect objects when ray tracing.

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


### Geometry

The `Geometry` object does not do anything special except allow an easy/extensible mechanism for geometry creation and destruction. Use the geometries std::vector object to add remove edit geometries and create your own scenes in this application.


### Cornell

The Cornell box is one such derived Geometry object. When it is created we create Embree geometry with `rtcNewGeometry` in `add_geometry(..)`. 

Creating the box geometry means setting a buffer of defined vertex positions for the geometry with `rtcSetNewGeometryBuffer(..)` and giving it `RTC_BUFFER_TYPE_VERTEX`.
```
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
      m_cornellBoxVertices.size());
```

Next, we setup an index buffer for for the Geometry defining Quads.

```
  /* set quads */
  Quad* quads = (Quad*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0,
                                               RTC_FORMAT_UINT4, sizeof(Quad),
                                               m_cornellBoxIndices.size());
```
Lastly, we bind vertex colors to a vertex attribute with `rtcSetGeometryVertexAttributeCount(..)` and `rtcSetSharedGeometryBuffer(..)`. Vertex attributes are not used directly in this sample. They are a place holder for expanding the application to consider vertex colors with the API.
Notice that such buffers need to be aligned per the API specficiation. `alignedMalloc(..)` is used.

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

The other `Geometry` derivative objects are highly similar, albeit the may use different primitive types. Note that the `Sphere` object uses an Embree defined Sphere primative as opposed to defining vertices manually for a sphere.

The Cornell scene also configures a light and camera associated with it. The camera view is oriented based on field of view, look at direction, up direction, width and height in pixels.

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






### Materials

### Lights

### Accumulation Buffer

### Accumulate

### Path Loop

## Performance and Quality

### VTune

Intel VTune Amplifier is a great tool when attempting to optimize your application. The hotspots chart below from VTune helps us find bottlenecks in our application. The unnamed functions are rtcIntersect1 and rtcOccluded1. The chart shows that together, these `Embree` functions, are the plurality of the compute, which for this introductory application is desirable.

```
Top Hotspots
Function             Module                 CPU Time  % of CPU Time(%)
-------------------  ---------------------  --------  ----------------
func@0x1814b9a20     embree3.dll            130.604s             23.6%
fdlog                ucrtbase.dll           108.756s             19.6%
func@0x1814cce90     embree3.dll             84.626s             15.3%
renderPixelFunction  rkPathTracer.exe        30.289s              5.5%
cosf                 ucrtbase.dll            13.829s              2.5%
[Others]             N/A                    185.692s             33.5%

Top Tasks
Task Type   Task Time  Task Count  Average Task Time
----------  ---------  ----------  -----------------
tbb_custom   625.022s       1,461             0.428s
```

### rtcIntersect1/rtcOccluded1

In this example `RTC_INTERSECT_CONTEXT_FLAG_COHERENT` is set prior to tracing primary rays with `rtcIntersect`. Using this bitflag shows [enhanced performance]( https://github.com/embree/embree/blob/v3.13.4/doc/src/api.md#fast-coherent-rays) when using ray packets. It is in place in our tutorial program because our primary rays are coherent.

Other rtcIntersect<n>/rtcOccluded<n> calls can schedule 

### Random Number Generation

rtcIntersect and rtcOccluded together are a plurality of compute. Looking at the other functions we see `fdlog(..)`.
fdlog is called by the application's random number generation routines. This suggests that for this scene, and this renderer's monte carlo scheme, random number generation is a significant proportion of the code.
The full Embree tutorials use a hand written fast LCG based random number generator. For explanation and simplicity purposes, we use C++11 random number generators in this tutorial. The quality (recurrence, distribution, speed) of random numbers will affect the quality of your results.  For example, moving to `mt19937` based random number generators without any extra hashing gives reasonable results in single accumulation and multi-accumulation images. However, it  was observed to perform slower than a `minstd_rand` generator. This performance impact was observed in the VTune hotspots function profile. However, a quality trade off can be readily observed if the `minstd_rand` generator is not hashed as in our tutorial. Visual artifacts may be apparent even given a high sample count.
We observe random number concerns due to their impact on performance. Random number needs can be different amongst different applications.

### Fidelity And Convergence

Convergence in an image is highly dependent on scene configuration. Keeping a point light at an sharp angle with respect to visible surfaces from the light can result in a noisier image. In instances where convergence is taking longer a developer may consider a denoise pass provided by functionality from a library like Intel Open Image Denoise. Many vendors of professional rendering solutions use Intel Open Image Denoise as a default final frame pass to enhance image quality.
Ceiling point Light 500 samples
Ceiling point Light 1000 samples
Ceiling point Light 2000 samples
Intel Open Image Denoise oidnDenoise 

# Next Steps:

## Ray tracers in the wild

### Intel OSPRay off-the-shelf renderers
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
    Lights
    Our lights have no physical dimensions. This keeps tutorial code simple. However, interesting lights and images are often generated with lights that have dimensions. Consider augmenting this application’s directional light an point light to give them a physical size. The full Intel Embree tutorials demonstrate a few different [light types to try](https://github.com/embree/embree/tree/v3.13.4/tutorials/common/lights)

### Materials omissions

Bidirectional Ray Distribution Function (BRDF) parameterization

The full Embree tutorial application passes a structure representing a BRDF representing reflectance model parameters. Our application hard codes reflection models in the interest of simplicity. However, the full tutorial application includes a BRDF parser for use with geometric models stored on disk in .obj/.mtl format.
When reviewing other codes, notice that parameters associated with the [Phong](https://en.wikipedia.org/wiki/Phong_reflection_model) reflection model are considered.

### Texturing

We do not cover texturing in this sample. Review the Intel Embree repository source to see a demonstration of applying terxtures.

### Transparency

Transparent materials are omitted in this tutorial. The full Intel Embree path tracer demonstrates [transparency]( https://github.com/embree/embree/blob/v3.13.4/tutorials/pathtracer/pathtracer_device.cpp#L1630) in materials. 

## More Information
You can find more information at the [ Intel oneAPI Rendering Toolkit portal ](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).

## License

Code samples are licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.
