# Introduction to Ray Tracing with Intel&reg; Embree

## Purpose
This tutorial sample demonstrates building and running a basic geometric ray tracing
applications with Intel&reg; Embree, the high-performance implementation of the Intel&reg; Embree API.

- Use this code and accompanying walkthrough to understand basic ray tracing of an image with the Intel&reg; Embree API. This sample prepares an API
explorer for further self-directed exploration of Intel&reg; Embree repository
tutorial programs. 

- You can expect to need less than 5 minutes to compile, run,
and review the ouput imagery. However, you might need 20 minutes (or more) to
understand the walkthrough and code depending on rendering algorithm and math
familiarity.

- This sample source code is derived from the `triangle_geometry`/`triangle_geometry_sycl`
source hosted as part of the Intel&reg; Embree tutorials in the [Embree
repository](https://github.com/embree/embree) on GitHub.

__Output Image:__

![rkRayTracer program
output](./cpu/example_images/rkRayTracer.png)

## Device Targets

1. [CPU](./cpu)
- Use the CPU `rkRayTracer` tutorial sample for an Introductory walkthrough targeting an Intel64 (x86-64) host Intel&reg; CPU.
- Start with this tutorial.

2. [GPU](./gpu)
- Use the GPU `rkRayTracerGPU` tutorial sample for a walkthrough of targeting Intel&reg; GPUs compatible with Xe-HPG architecture or higher (ex: Intel&reg; ARC graphics).

## Conclusion

After walking through Embree ray tracing capabilities, many users may desire to use a
rendering API at a higher layer, perhaps at an _engine_ layer. Such developers
should consider examining the Intel&reg; OSPRay API and library, which implements
rendering facilities on top of Embree.

You can find more information by visiting [Intel&reg; oneAPI Rendering
Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/rendering-toolkit.html).

## License

Code samples are licensed under the Apache 2.0 license. See
[LICENSE.txt](LICENSE.txt) for details.
