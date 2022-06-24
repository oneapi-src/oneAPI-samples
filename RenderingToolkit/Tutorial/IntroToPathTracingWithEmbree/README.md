WIP
# Path Tracing with Intel&reg; Embree

## Purpose

By walking through this pathtracer tutorial, you will understand how to implement a basic pathtracer with Intel Embree. Also, you will have a better understanding of:
•	visual quality considerations when implementing applications with Embree.
•	the pathtracer renderer as implemented within Intel OSPRay. 
•	typical scene controls of interest in the Intel OSPRay Studio showcase reference application or other professional rendering solutions.

Onboarding with the pathtracer can help deeper study of the full feature tutorial codes on the Intel Embree github repository.
Expect less than 10 minutes to compile and run the application on contemporary hardware. Expect at least an hour for following the algorithm, self-directed edits, then to rebuild, rerun, and understand the application.
An output image from this `pathtracer_oneapi` tutorial program:
This sample source code is a consolidated refactor of the `pathtracer` source hosted as part of the Intel&reg; Embree tutorials in the [Embree
repository](https://github.com/embree/embree) on GitHub.

## Prerequisites

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br>CentOS* 8 (or compatible) <br>Windows* 10 <br>macOS* 10.15+
| Hardware                          | Intel 64 Penryn or higher with SSE4.1 extensions; ARM64 with NEON extensions <br>(Optimization requirement: Intel&reg; Embree is further optimized for Intel 64 Skylake or higher with AVX512 extensions)
| Compiler Toolchain                | Windows* OS: MSVS 2019 or MSVS 2022 with Windows* SDK and CMake* <br>Other platforms: C++11 compiler and CMake*
| Libraries                         | Install Intel&reg; oneAPI Rendering Toolkit (Render Kit), including Intel&reg; Embree and Intel® oneAPI Threading Building Blocks (oneTBB) <br>Install Intel&reg; oneAPI Base Toolkit for the `dev-utilities` default component
| Tools                             | .png capable image viewer

## Build and Run

### Windows*

```
mkdir build
cd build
cmake -G"Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
cd Release
.\pathtracer_oneapi.exe
```
**Note**: Visual Studio 2022 users should use the `-G"Visual Studio 17 2022"`
generator flag.

Open the resulting .png image files with an image viewer.

### Linux* or macOS*
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
./pathtracer_oneapi
```
Open the resulting .png image files with an image viewer.

## Ray Tracer Introduction Recap

In the Introduction to Ray Tracing with Intel Embree sample, we walked through a program that found basic albedo (surface color) and basic lighting for a cube and a plane. The output image was generated via rays cast from each pixel of our camera to determine color at the corresponding intersection. These initial rays from the camera are primary rays in our scene. Then, if and only if the intersected surface was not occluded, the albedo would be increased to full strength. This gave the appearance of depth with a shadow and an idea of distance for the cube hovering above our plane. 
However, this was only the first step to computing light transport used to simulate generation of a photoreal image. So far, we have only simulated direct illumination.

## What is path tracing?
Path tracing implments a simulation of global illumination.
Our primary rays alone do not capture light illuminating a surface from other surfaces visible to the intersection. So, in this tutorial, we construct a path that continues from a primary ray intersection point to sample light reflecting from all other visible surfaces and lights in the scene. By integrating the light visible from the hemisphere above the intersection, (global illumination)[https://en.wikipedia.org/wiki/Path_tracing] is simulated.
The path itself need not stop after one additional bounce (reflection or refraction). Multiple bounces can capture light from multiple surfaces. In a compute environment, we either set a constant or create logical heuristics limiting the path length. For our paths to have more interesting surfaces, we update the geometric scene data in this tutorial over the previous.
Implied in our description of light reflecting from a surface is that surface properties affect the light reflected. Here we introduce material properties to alter the light reflected off surfaces.

## Key Features Added
New features added to the ray tracer `triangle_geometry` sample are discussed at a high level. The features are then highlighted with in-source implementation details. It may take a few rounds of looking at the high level description and then the source-accompanied description to understand the application.
  

### Added Scene
Global illumination is difficult to demonstrate without surfaces affecting and reflecting light to visible surfaces within the scene. We implement Cornell Box scene geometry , as defined within Intel OSPRay for the Intel OSPRay `ospExamples` application. The Cornell Box is a familiar dataset that can show capability of computing global illumination and shadows.
Materials
We have two basic materials implemented in the sample.
1)	Lambertian(https://en.wikipedia.org/wiki/Lambertian_reflectance)
This is a soft diffuse material that scatters light proportional to the cosign of the angle of incidence in all directions. Another word often used to describe a Lambertian surface is a “matte” surface.
2)	Mirror
We introduce a near-perfect mirror as a material surface. This mirror only reflects 80% of incident light. Limiting reflectance more closely matches the behavior of real mirrors.
With whatever material chosen.
### Light

A point light has been added to this sample. Versus an infinitely distanced directional light, the point light demonstrates energy dissipation with distance from the light source. It also demonstrates varying amounts of light proportional to the angle of the point light to a surface.

### Global Illumination

We describe the algorithm at a high level:
1. At a ray-to-scene intersection we determine the light attenuation due to the material at the intersection.
2. To do this, we generate a bounce ray pointing to a location on the unit hemisphere above an intersection. This ray represents the next leg of our ‘path’ being traced.
3. In the case of a Lambertian surface we generate a ray randomly as a representative reflection of all incident light at the point. Lower ray angles (approaching tangent) will contribute very little global light through the new ray. Higher angles will contribute more.
4. In the case of a mirror, the reflection is not random and computed as a mirror ray. Mirror material rays will not attenuate based on direction. They will only attenuate based upon the inherent reflectivity of the mirror.
5. In either case, a weight for that representative reflection is computed based on the probability distribution function of that ray based on the material.
6. Next, we iterate through all lights in the scene. We add the non-occluded incoming light proportional to: attenuation from previous bounces, the weight of the light based on the type of light and distance to the surface,  material attenuation.
7. The next leg of the ‘path’ has an attenuation computed proportional to material attenuation and the probability distribution function scalar for the ‘bounce’ ray leaving the surface.
8. A ray cast occurs using the surface intersect in the direction of the ‘bounce’ ray.
9. When the maximum path length is achieved a luminance (color) value is returned. In this sample application, path length is hardcoded to 8.
10. This luminance is added to the luminance computed to previous paths sampled for the same camera pixel. An average is taken by dividing the luminance by the number of total samples
 If we take an infinite number of samples with infinite path length, all paths under all sampled pixels will logically account for all light in the scene. The result will be a converged image. 
11. In this tutorial, by default, we hardcode 1 sample per pixel. 1 sample per pixel alone will yield a noisy image. This could be changed to use more samples, but in the interest of study, by default we use accumulations of sampled image frames into an accumulation buffer.

### Accumulation buffer

The accumulation buffer stores total luminance sampled underneath all pixels over all accumulations. To write to an image file output,  we divide by number of accumulations to get our per channel color averages. An accumulation buffer is useful in an interactive rendering context to intermittently update a windowed frame buffer. The application can continue with a stationary scene and camera to combine many accumulations to converge the output. Here, we use it to check a first sampling of our image before accumulating many more rendered frames. This is useful for visual debug when developing an application.
1spp , 1 accumulation
1spp, 500 accumulations


## Details of Improvements (Source)
### Scene Selection
### Geometry
### Materials
### Lights
### Accumulation Buffer
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
renderPixelFunction  pathtracer_oneapi.exe   30.289s              5.5%
cosf                 ucrtbase.dll            13.829s              2.5%
[Others]             N/A                    185.692s             33.5%

Top Tasks
Task Type   Task Time  Task Count  Average Task Time
----------  ---------  ----------  -----------------
tbb_custom   625.022s       1,461             0.428s
```
### rtcIntersect1/rtcOccluded1
In this example `RTC_INTERSECT_CONTEXT_FLAG_COHERENT` is set prior to tracing primary rays with `rtcIntersect`. Using this bitflag shows [enhanced performance]( https://github.com/embree/embree/blob/v3.13.4/doc/src/api.md#fast-coherent-rays) when using ray packets. It is in place in our tutorial program because our primary rays are coherent.
Random Number Generation
rtcIntersect and rtcOccluded together are a plurality of compute. Looking at the other functions we see `fdlog(…)`.
fdlog is called by the application’s random number generation routines. This suggests that for this scene, and this renderer’s monte carlo scheme, random number generation is a significant proportion of the code.
The full Embree tutorials use a hand written fast LCG based random number generator. For explanation and simplicity purposes, we use C++11 random number generators in this tutorial. The quality (recurrence, distribution, speed) of random numbers will affect the quality of your results.  For example, moving to `mt19937` based random number generators without any extra hashing gives reasonable results in single accumulation and multi-accumulation images. However, it  was observed to perform slower than a `minstd_rand` generator. This performance impact was observed in the VTune hotspots function profile. However, a quality trade off can be readily observed if the `minstd_rand` generator is not hashed as in our tutorial. Visual artifacts may be apparent even given a high sample count.
We observe random number concerns due to their impact on performance. Random number needs can be different amongst different applications.

### Fidelity And Convergence
Convergence in an image is highly dependent on scene configuration. Keeping a point light at an sharp angle with respect to visible surfaces from the light can result in a noisier image. In instances where convergence is taking longer a developer may consider a denoise pass provided by functionality from a library like Intel Open Image Denoise. My vendors of professional rendering solutions use Intel Open Image Denoise as a default final frame pass to enhance image quality.
Ceiling point Light 500spp
Ceiling point Light 1000spp
Ceiling point Light 2000spp





# Next Steps:
## Ray tracers in the wild
Intel OSPRay off-the-shelf renderers
The Intel OSPRay API defines renderer objects. If you are looking for an off the shelf renderer for your environment, consider any of these [renderers]( https://www.ospray.org/documentation.html#renderers)
Pathtracer `pathtracer`:
Supports volumetric visualization, global illumination, materials, textures. Significant superset of features over this tutorial program.
SciVis `scivis`:
Supports volumetric visualization, ambient occlusion, lights.
AmbientOcclusion `ao`:
Supports volumetric visualization, ambient occlusion, no lights.
Scene Graphs:
Typical rendering applications employ a scenegraph for managing complex scene data from cameras, instancing, objects, to materials, textures, and animations. This tutorial hardcodes a few basic geometric objects and materials. The full Intel Embree tutorial programs contain a basic reference [scene graph]( https://github.com/embree/embree/blob/v3.13.4/tutorials/common/tutorial/tutorial.cpp). Similarly, the OSPRay ospExamples viewer employs a reference scene graph. Lastly, Intel OSPRay Studio uses it’s own reference scene graph.
Each implementation can be a good reference point in building out your own scene graph in a production application. When moving beyond sandbox programs, software scalability challenges make a scene graph a practical necessity.
	Lights
	Our lights have no physical dimensions. This keeps tutorial code simple. However, interesting lights and images are often generated with lights that have dimensions. Consider augmenting this application’s directional light an point light to give them a physical size. The full Intel Embree tutorials demonstrate a few different [light types to try](https://github.com/embree/embree/tree/v3.13.4/tutorials/common/lights)
	 
## Materials omissions
Bidirectional Ray Distribution Function (BRDF) parameterization
The full application passes a structure representing a BRDF representing reflectance model parameters. Our application hard codes two reflection models  in the interest of simplicity. However, the full tutorial application includes a BRDF parser for use with geometric models stored on disk in .obj/.mtl format.
When reviewing other codes, notice that parameters associated with the [Phong](https://en.wikipedia.org/wiki/Phong_reflection_model) reflection model are considered.

## Texturing
## Transparency

Transparent materials are omitted in this tutorial. The full Intel Embree path tracer shows [transparency]( https://github.com/embree/embree/blob/v3.13.4/tutorials/pathtracer/pathtracer_device.cpp#L1630) in materials. Keep in mind that if you have materials that have transparency or refract. You can add this 

You can find more information at the [ Intel oneAPI Rendering Toolkit portal ](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).

## License

Code samples are licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.





