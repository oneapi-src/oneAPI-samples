// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/* This is a small example tutorial how to use OSPRay in an application using
   GLM instead of rkcommon for math types.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#define NOMINMAX
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include <cstring>

#include <iostream>
#include <vector>

#include "ospray/ospray_cpp.h"
#define OSPRAY_GLM_DEFINITIONS
#include "ospray/ospray_cpp/ext/glm.h"
#include "rkcommon/utility/SaveImage.h"

int main(int argc, const char **argv)
{
  // image size
  glm::ivec2 imgSize;
  imgSize.x = 1024; // width
  imgSize.y = 768; // height

  // camera
  glm::vec3 cam_pos{0.f, 0.f, 0.f};
  glm::vec3 cam_up{0.f, 1.f, 0.f};
  glm::vec3 cam_view{0.1f, 0.f, 1.f};

  // triangle mesh data
  std::vector<glm::vec3> vertex = {glm::vec3(-1.0f, -1.0f, 3.0f),
      glm::vec3(-1.0f, 1.0f, 3.0f),
      glm::vec3(1.0f, -1.0f, 3.0f),
      glm::vec3(0.1f, 0.1f, 0.3f)};

  std::vector<glm::vec4> color = {glm::vec4(0.9f, 0.5f, 0.5f, 1.0f),
      glm::vec4(0.8f, 0.8f, 0.8f, 1.0f),
      glm::vec4(0.8f, 0.8f, 0.8f, 1.0f),
      glm::vec4(0.5f, 0.9f, 0.5f, 1.0f)};

  std::vector<glm::uvec3> index = {glm::uvec3(0, 1, 2), glm::uvec3(1, 2, 3)};

  // initialize OSPRay; OSPRay parses (and removes) its commandline parameters,
  // e.g. "--osp:debug"
  OSPError init_error = ospInit(&argc, argv);
  if (init_error != OSP_NO_ERROR)
    return init_error;

  // use scoped lifetimes of wrappers to release everything before ospShutdown()
  {
    // create and setup camera
    ospray::cpp::Camera camera("perspective");
    camera.setParam("aspect", imgSize.x / (float)imgSize.y);
    camera.setParam("position", cam_pos);
    camera.setParam("direction", cam_view);
    camera.setParam("up", cam_up);
    camera.commit(); // commit each object to indicate modifications are done

    // create and setup model and mesh
    ospray::cpp::Geometry mesh("mesh");
    mesh.setParam("vertex.position", ospray::cpp::CopiedData(vertex));
    mesh.setParam("vertex.color", ospray::cpp::CopiedData(color));
    mesh.setParam("index", ospray::cpp::CopiedData(index));
    mesh.commit();

    // put the mesh into a model
    ospray::cpp::GeometricModel model(mesh);
    model.commit();

    // put the model into a group (collection of models)
    ospray::cpp::Group group;
    group.setParam("geometry", ospray::cpp::CopiedData(model));
    group.commit();

    // put the group into an instance (give the group a world transform)
    ospray::cpp::Instance instance(group);
    instance.commit();

    // put the instance in the world
    ospray::cpp::World world;
    world.setParam("instance", ospray::cpp::CopiedData(instance));

    // create and setup light for Ambient Occlusion
    ospray::cpp::Light light("ambient");
    light.commit();

    world.setParam("light", ospray::cpp::CopiedData(light));
    world.commit();

    // create renderer, choose Scientific Visualization renderer
    ospray::cpp::Renderer renderer("scivis");

    // complete setup of renderer
    renderer.setParam("aoSamples", 1);
    renderer.setParam("backgroundColor", 1.0f); // white, transparent
    renderer.commit();

    // create and setup framebuffer
    ospray::cpp::FrameBuffer framebuffer(
        imgSize.x, imgSize.y, OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);
    framebuffer.clear();

    // render one frame
    framebuffer.renderFrame(renderer, camera, world);

    // access framebuffer and write its content as PPM file
    uint32_t *fb = (uint32_t *)framebuffer.map(OSP_FB_COLOR);
    rkcommon::utility::writePPM("firstFrameCpp.ppm", imgSize.x, imgSize.y, fb);
    framebuffer.unmap(fb);

    // render 10 more frames, which are accumulated to result in a better
    // converged image
    for (int frames = 0; frames < 10; frames++)
      framebuffer.renderFrame(renderer, camera, world);

    fb = (uint32_t *)framebuffer.map(OSP_FB_COLOR);
    rkcommon::utility::writePPM(
        "accumulatedFrameCpp.ppm", imgSize.x, imgSize.y, fb);
    framebuffer.unmap(fb);

    ospray::cpp::PickResult res =
        framebuffer.pick(renderer, camera, world, 0.5f, 0.5f);

    if (res.hasHit) {
      std::cout << "Picked geometry [inst: " << res.instance.handle()
                << ", model: " << res.model.handle() << ", prim: " << res.primID
                << "]" << std::endl;
    }
  }

  ospShutdown();

  return 0;
}
