# Introduction to Ray Tracing with Intel Embree

## Purpose
This sample demonstrates building and running a basic geometric ray tracing application with Embree. A code walkthrough is provided to understand the ray tracer, as well as the usage of the Embree API. Use this walkthrough prior to further self directed deep dive exploration of Embree tutorial programs.

This source is a consolidated refactor of the tutorial_geometry source hosted as part of the Embree tutorials in the Embree repository. 

![tutorial_geometry_oneapi program output](example_images/triangle_geometry_oneapi.png)

## Prerequisites

| Minimum Requirements              | Description
|:---                               |:---
| OS                                | Linux* OS: Ubuntu* 18.04, CentOS* 8 (or compatible); Windows* 10; MacOS* 10.15+
| Hardware                          | Intel 64 Penryn or higher with SSE4.1 extensions; ARM64 with NEON extensions
| Compiler Toolchain                | Windows* OS: MSVS 2019 or MSVS 2022 with Windows* SDK and CMake*; Other platforms: C++11 compiler and CMake*
| Libraries                         | - Install Intel oneAPI Rendering Toolkit including Embree, and oneTBB -Install Intel oneAPI Base Toolkit for the 'dev-utilites' default component
| Tools                             | .png capable image viewer 

| Optimized Requirements            | Description
| :---                              | :---
| Hardware                          | Intel 64 Skylake or newer with AVX512 extentions, ARM64 with NEON extensions

| Objective                         | Description
|:---                               |:---
| What you will learn               | A basic understanding of constructing a ray tracer with the Embree API.
| Time to complete                  | 5 minutes to compile, run, and review output image; 20+ minutes to study



## License

Code samples are licensed under the Apache 2.0 license. See [LICENSE.txt](LICENSE.txt) for details.

## Build and Run

### Windows* OS

```
mkdir build
cd build
cmake -G"Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
cd Release
.\triangle_geometry_oneapi.exe
```
	>Note: Visual Studio 2022 users should use the -G"Visual Studio 17 2022" generator flag.

Open the resulting file: `triangle_geometry_oneapi.png` with an image viewer.

### Linux* OS or Mac* OS
```
mkdir build
cd build
cmake ..
cmake --build .
./triangle_geometry_oneapi
```

Open the resulting file: `triangle_geometry_oneapi.png` with an image viewer.

## Objective

### Scene Description

1. The code renders an image of two geometric objects from a perspective camera. The first ofile:///C:/example_images/triangle_geometry_oneapi.pngbject is a cube. The cube consists of 8 vertices, 12 triangles, and 6 faces. Triangles are our primitive geometric object.
2. The cube is of size 2 x 2 x 2 units and is centered about the origin in world space. Each cube face (group of two triangles) is given a new color.
3. The second object is our ground plane. The ground plane consists of 4 vertices, 2 triangles, and 1 face. The ground plane is red.
4. The ground plane is 20 x 20 units in world space. The center of the ground plane is 0, -2, 0 in cartesian (x,y,z) coordinate world space. The plane is coplanar with y=-2;
5. A light is defined in the direction of -1, -1, -1.

### Camera Transform and Ray Casts

1. The virtual camera is positioned at 1.5, 1.5, -1.5 in cartesian (x,y,z) world space. The camera is looking at the origin 0, 0, 0. The camera is thus facing the center of the cube.
2. The world 'up' direction is aligned with the positive y axis (0, 1, 0).
3. The camera is defined with an angular field of vision (given in degrees) as well as a width and height in pixels. We use the image aspect ratio, the field of vision, camera position, and camera look at point to transform each ray cast based on which pixel is being computed. 
5. All image pixel color values are initialized to black. When a ray is cast and an object hit occurs, the color associated with the intersected triangle is quieried. The queried color is added to the image pixel at half magnitude/intensity.
6. Next, a second hit query occurs from the hit location to determine if the hit is not in shadow with respect to our light. If the hit is not in shadow, our image pixel intensity is doubled. Leaving shadows at half intensity gives the impression of the cube casting a shadow onto our ground plane.
7. The computed image is written to disk for review.


## Code Walkthrough using Embree API

SECTION TODO


First, beginning in the main function, an Embree device is created using the `rtcNewDevice` function. 
The object created by `rtcNewDevice` provides a factory for all other Embree objects. These objects, such as geometry, can only be used with their associated device. In this sample, no configuration parameters are used.

```
    /* create device */
    g_device = rtcNewDevice(nullptr);
```
	> Note: Configuration parameters provide threading control, isa selection, frquency control, and internal memory control. Explorers are encouraged to use the no-parameter default until they begin benchmarking their completed solution.


`rtcSetDeviceErrorFunction` allows the embree runtime to assign our developer defined error function as a call back if and when an error occurs. If a call to the Embree API generates and error, the error handler will be executed.

Next, we initialize objects.

	>Note: This functionality is defined in the `device_init` and `TutorialData_Constructor` functions in the full suite of Embree tutorials

The `device_init` user function sets up objects associated with our Embree `RTCDevice` (struct: RTCDevice). In it We initialize pointers for our Embree RTC as well as vertex and face color attributes.



Lastly, we release the device with the `rtcReleaseDevice` function.


### Elective Exercise
The code as written renders face colors. Consider using the Embree API to render triangle primitives that have vertex colors interpolated across each face. What changes can be made to this source to achieve this effect?

## Next Steps

### Get the full tutorial codes

- Acquire Embree standlone code from the Embree repository.
- Alternately, acquire Embree from the oneAPI Rendering Toolkit rkutil Superbuild cmake script.
	- The script is available after toolkit install in the oneAPI-root-directory/rkutil/latest/superbuild folder. It is also available on Github.
	- The superbuild script gives a complete, and most importantly build time configurable sandbox rendering toolkit libraries
		- Instructions: Windows, Linux, MacOS


### Preview of full feature codes

The triangle_geometry_oneapi sample avoids some implementation features in the interest of being a basic introduction. However, the full details of ray tracing features available with Embree are best observed in the Embree tutorial codes. These codes reside within the Embree repository. 

Our oneAPI Rendering Toolkit triangle_geometry sample forgoes use of the Embree common library found in the full codes. The Embree common library implements math, tasking, and system access routines that are common to rendering applications. The Embree common library lives in the Embree repository but is not the same as the Embree API. The Embree common library defines objects and functions in an insular fashion. In turn, Embree uses the Embree common library so Embree can be flexible enough to provide standalone ray tracing capability when required.

The full version of triangle_geometry uses Embree common in a manner that is more scalable and maintainable than our introductory application here on the oneAPI samples repository. It is recommended to use the Embree common library objects and routines in production projects where applicable. Popular IDEs like Microsoft Visual Studio 2019, can allow for easy review of types and function definitions used in full versions of tutorial applications, and thus are recommended during source code exploration.

### Differences with Full Tutorial Geometry

**rkcommon vs Embree's common library**

For our abridged oneAPI tutorial, we use rkcommon. rkcommon is provided as an internal support library for the oneAPI Rendering Toolkit. We use one replacement object and three replacement function calls that are defined in rkcommon. 
- Vec3fa
	- Our Vec3fa vector container comes from rkcommon::math::vec_t<float, 3, 1>. This is a container for three floats with some basic overloads. The Full Embree tutorials use embree::Vec3fa from Embree common, where Vector operations are slightly different. It uses hardware instruction capability available on Intel hardware.
- normalize
	- This function normalizes a vector.
- cross
	- This function finds the cross product between two vectors.
- deg2rad
	- This function converts angles given in degrees to radians.

**parallel execution**

Full Embree tutorials encapsulate `parallel_for` from oneTBB as `embree::parallel_for`. As needed, find `embree::parallel_for` in embree/common/algorithms. Full Embree tutorials also encapsulate current_thread_index underneath the TaskScheduler object found in embree/common/tasking/taskschedulertbb.h

oneAPI triangle_geometry samples:
```
    tbb::task_group_context tgContext;
    tbb::parallel_for(tbb::blocked_range<size_t>(0,numTilesX * numTilesY, 1), [&](const tbb::blocked_range<size_t>& r) {
        const int threadIndex = tbb::this_task_arena::current_thread_index();
        for (size_t i = r.begin(); i < r.end(); i++) 
            renderTileTask((int)i, threadIndex, pixels, width, height, channels, time, camera, numTilesX, numTilesY);
            
        }, tgContext);
    if (tgContext.is_group_execution_cancelled())
        throw std::runtime_error("task cancelled");
```
Full Embree tutorial:
```
  parallel_for(size_t(0),size_t(numTilesX*numTilesY),[&](const range<size_t>& range) {
    const int threadIndex = (int)TaskScheduler::threadIndex();
    for (size_t i=range.begin(); i<range.end(); i++)
      renderTileTask((int)i,threadIndex,pixels,width,height,time,camera,numTilesX,numTilesY);
  }); 
```

Users are encouraged to review and use the Embree common objects when moving from sample to proof of concept and production applications.

**Embree tutorial objects**

The full Embree tutorial programs launch applications using a Tutorial class object. This object is reused across all tutorials and implements many convenience features. Depending on the scope of the specific tutorial application, features include but are not limited to:
- Interactive windowing
	- full triangle_geometry by default starts in interactive mode
- A reference scenegraph facility with load routines for different scene types
- Command parsing for scene control
	- Example: Setting camera perameters via command line
- Animated cameras
- Motion blur
- Metadata shading
	- We only render with a pure albedo color in our oneAPI example. But the full tutorial can visulize various metadata. Examples:
		- Visualize compute cost of a ray.
		- Visualize differential geometry at a ray intersection
- Performance metrics for rays
- More

The oneAPI triangle_geometry sample program also manually defines transform objects that are offered in the Embree tutorial library. Notice our camera containers which hold data for transformations: `AffineSpace3fa`, `LinearSpace3fa`.
```
struct LinearSpace3 {
    Vec3fa vx, vy, vz;

};
typedef struct Affine3fa {
    LinearSpace3 l;
    Vec3fa p;
} ISPCCamera;
```

Within Embree common these containers service a dedicated camera object with convenient functions for camera transforations. Here are snippets from the full tutorials that constitute the perspective camera projection.

embree/common/math/affinespace.h:
```
      /*! return matrix for looking at given point, only in 3D */
      static __forceinline AffineSpaceT lookat(const VectorT& eye, const VectorT& point, const VectorT& up) {
        VectorT Z = normalize(point-eye);
        VectorT U = normalize(cross(up,Z));
        VectorT V = normalize(cross(Z,U));
        return AffineSpaceT(L(U,V,Z),eye);
      }
```
embree/tutorials/common/tutorial/camera.h
```
    AffineSpace3fa camera2world ()
    {
      AffineSpace3fa local2world = AffineSpace3fa::lookat(from, to, up);
      if (!(local2world == local2world))
        throw std::runtime_error("invalid camera specified");
      
      if (handedness == RIGHT_HANDED)
        local2world.l.vx = -local2world.l.vx;
      
      return local2world;
    }
    ...

    ISPCCamera getISPCCamera (size_t width, size_t height)
    {
      const float fovScale = 1.0f/tanf(deg2rad(0.5f*fov));
      const AffineSpace3fa local2world = camera2world();
      Vec3fa vx = local2world.l.vx;
      Vec3fa vy = -local2world.l.vy;
      Vec3fa vz = -0.5f*width*local2world.l.vx + 0.5f*height*local2world.l.vy + 0.5f*height*fovScale*local2world.l.vz;
      Vec3fa p =  local2world.p;
      return ISPCCamera(AffineSpace3fa(vx,vy,vz,p));
    }
```
These are defined in camera.h of in embree/tutorials/common/tutorial.

## Conclusion

At this point, we have covered a baseline for how to raytrace a scene with the Embree API and we have introduced the full Embree tutorial programs for futher evaluation. Still, many users may find they wish to use a rendering API at a higher layer, perhaps at an _engine_ layer. Such developers should consider examining the Intel OSPRay API and library, which implements rendering facitilities on top of Embree.

Reviewing triangle_geometry.cpp and tutorial_geomatry_device.cpp as defined in embree/tutorials is a helpful next step
You can find more information at the [ Intel oneAPI Rendering Toolkit portal ](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).




