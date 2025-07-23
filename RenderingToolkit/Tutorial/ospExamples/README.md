ospExamples
-----------

Apart from tutorials, `OSPRay` comes with a C++ app called
[`ospExamples`](https://github.com/oneapi-src/oneAPI-samples/tree/development/RenderingToolkit/Tutorial/ospExamples)
which is an elaborate easy-to-use tutorial, with a single interface to
try various `OSPRay` features. It is aimed at providing users with
multiple simple scenes composed of basic geometry types, lights, volumes
etc. to get started with OSPRay quickly.

`ospExamples` app runs a `GLFWOSPRayWindow` instance that manages
instances of the camera, framebuffer, renderer and other OSPRay objects
necessary to render an interactive scene. The scene is rendered on a
`GLFW` window with an `imgui` GUI controls panel for the user to
manipulate the scene at runtime.

The application is refactored from https://github.com/ospray/ospray/tree/master/apps such that it can be compiled and built upon independently of OSPRay engine.


<figure>
<img src="https://ospray.github.io/images/ospExamples.png"
style="width:90.0%"
alt="ospExamples application with default boxes scene." />
<figcaption aria-hidden="true"><code>ospExamples</code> application with
default <code>boxes</code> scene.</figcaption>
</figure>



### Scenes

Different scenes can be selected from the `scenes` dropdown and each
scene corresponds to an instance of a special `detail::Builder` struct.
Example builders are located in
[`apps/common/ospray_testing/builders/`](https://github.com/ospray/ospray/tree/master/apps/common/ospray_testing/builders).
These builders provide a usage guide for the OSPRay scene hierarchy and
OSPRay API in the form of `cpp` wrappers. They instantiate and manage
objects for the specific scene like `cpp::Geometry`, `cpp::Volume`,
`cpp::Light` etc.

The `detail::Builder` base struct is mostly responsible for setting up
OSPRay `world` and objects common in all scenes (for example lighting
and ground plane), which can be conveniently overridden in the derived
builders.

Given below are different scenes listed with their string identifiers:

boxes  
A simple scene with `box` geometry type.

cornell_box  
A scene depicting a classic cornell box with `quad` mesh geometry type
for rendering two cubes and a `quad` light type.

curves  
A simple scene with `curve` geometry type and options to change
`curveBasis`. For details on different basis’ please check documentation
of [curves](#curves).

gravity_spheres_volume  
A scene with `structuredRegular` type of [volume](#volumes).

gravity_spheres_isosurface  
A scene depicting iso-surface rendering of `gravity_spheres_volume`
using geometry type `isosurface`.

perlin_noise_volumes  
An example scene with `structuredRegular` volume type depicting perlin
noise.

random_spheres  
A simple scene depicting `sphere` geometry type.

streamlines  
A scene showcasing streamlines geometry derived from `curve` geometry
type.

subdivision_cube  
A scene with a cube of `subdivision` geometry type to showcase
subdivision surfaces.

unstructured_volume  
A simple scene with a volume of `unstructured` volume type.

### Renderer

This app comes with four [renderer](#renderers) options: `scivis`,
`pathtracer`, `ao` and `debug`. The app provides some common rendering
controls like `pixelSamples` and other more specific to the renderer
type like `aoSamples` for the `scivis` and `ao` renderer or
`maxPathLength` for the `pathtracer`.

The sun-sky lighting can be used in a sample scene by enabling the
`renderSunSky` option of the `pathtracer` renderer. It allows the user
to change `turbidity` and `sunDirection`.

<figure>
<img src="https://ospray.github.io/images/renderSunSky.png"
style="width:90.0%"
alt="Rendering an evening sky with the renderSunSky option." />
<figcaption aria-hidden="true">Rendering an evening sky with the
<code>renderSunSky</code> option.</figcaption>
</figure>

## Build and Run

### Additional Notes
oneAPI Rendering Toolkit 2023.1 version's cmake file contains an errata. The errata will produce an error while building the example. Please apply the following workaround described in the following page. 2023.1.1 version will address the issue.

https://community.intel.com/t5/Intel-oneAPI-Rendering-Toolkit/2023-1-troubleshoot-errata-CMake-Error/m-p/1476040#M98

### Windows

1. Run a new **x64 Native Tools Command Prompt for MSVS 2019**.

```
call <path-to-oneapi-folder>\setvars.bat
cd <path-to-oneAPI-samples>\RenderingToolkit\Tutorial\ospExamples
mkdir build
cd build
cmake ..
cmake --build .
cd Debug
ospExamples.exe
```



### Linux

1. Start a new Terminal session.
```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/Tutorial/ospExamples
mkdir build
cd build
cmake ..
cmake --build .
./ospExamples
```


### macOS

1. Start a new Terminal session.

```
source <path-to-oneapi-folder>/setvars.sh
cd <path-to-oneAPI-samples>/RenderingToolkit/Tutorial/ospExamples
mkdir build
cd build
cmake ..
cmake --build .
./ospExamples
```
