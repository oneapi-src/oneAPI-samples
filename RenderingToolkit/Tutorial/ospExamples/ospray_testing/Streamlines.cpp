// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"
#include "ospray_testing.h"
#include "rkcommon/utility/random.h"
// stl
#include <random>
using namespace rkcommon::math;

namespace ospray {
namespace testing {

struct Streamlines : public detail::Builder
{
  Streamlines() = default;
  ~Streamlines() override = default;

  void commit() override;

  cpp::Group buildGroup() const override;
};

// Inlined definitions ////////////////////////////////////////////////////

void Streamlines::commit()
{
  Builder::commit();

  addPlane = false;
}

cpp::Group Streamlines::buildGroup() const
{
  cpp::Geometry slGeom("curve");

  std::vector<vec4f> points;
  std::vector<unsigned int> indices;
  std::vector<vec4f> colors;

  std::mt19937 rng(randomSeed);
  utility::uniform_real_distribution<float> radDist(0.5f, 1.5f);
  utility::uniform_real_distribution<float> stepDist(0.001f, 0.1f);
  utility::uniform_real_distribution<float> sDist(0, 360);
  utility::uniform_real_distribution<float> dDist(360, 720);
  utility::uniform_real_distribution<float> freqDist(0.5f, 1.5f);

  // create multiple lines
  int numLines = 100;
  for (int l = 0; l < numLines; l++) {
    int dStart = sDist(rng);
    int dEnd = dDist(rng);
    float radius = radDist(rng);
    float h = 0;
    float hStep = stepDist(rng);
    float f = freqDist(rng);

    float r = (720 - dEnd) / 360.f;
    vec4f c(r, 1 - r, 1 - r / 2, 1.f);

    // spiral up with changing radius of curvature
    for (int d = dStart; d < dStart + dEnd; d += 10, h += hStep) {
      vec3f p, q;
      float startRadius, endRadius;

      p.x = radius * std::sin(d * M_PI / 180.f);
      p.y = h - 2;
      p.z = radius * std::cos(d * M_PI / 180.f);
      startRadius = 0.015f * std::sin(f * d * M_PI / 180) + 0.02f;

      q.x = (radius - 0.05f) * std::sin((d + 10) * M_PI / 180.f);
      q.y = h + hStep - 2;
      q.z = (radius - 0.05f) * std::cos((d + 10) * M_PI / 180.f);
      endRadius = 0.015f * std::sin(f * (d + 10) * M_PI / 180) + 0.02f;
      if (d == dStart) {
        const vec3f rim = lerp(1.f + endRadius / length(q - p), q, p);
        const vec3f cap = lerp(1.f + startRadius / length(rim - p), p, rim);
        points.push_back(vec4f(cap, 0.f));
        points.push_back(vec4f(rim, 0.f));
        points.push_back(vec4f(p, startRadius));
        points.push_back(vec4f(q, endRadius));
        indices.push_back(points.size() - 4);
        colors.push_back(c);
        colors.push_back(c);
      } else if (d + 10 < dStart + dEnd && d + 20 > dStart + dEnd) {
        const vec3f rim = lerp(1.f + startRadius / length(p - q), p, q);
        const vec3f cap = lerp(1.f + endRadius / length(rim - q), q, rim);
        points.push_back(vec4f(p, startRadius));
        points.push_back(vec4f(q, endRadius));
        points.push_back(vec4f(rim, 0.f));
        points.push_back(vec4f(cap, 0.f));
        indices.push_back(points.size() - 7);
        indices.push_back(points.size() - 6);
        indices.push_back(points.size() - 5);
        indices.push_back(points.size() - 4);
        colors.push_back(c);
        colors.push_back(c);
      } else if ((d != dStart && d != dStart + 10) && d + 20 < dStart + dEnd) {
        points.push_back(vec4f(p, startRadius));
        indices.push_back(points.size() - 4);
      }
      colors.push_back(c);
      radius -= 0.05f;
    }
  }

  slGeom.setParam("vertex.position_radius", cpp::CopiedData(points));
  slGeom.setParam("index", cpp::CopiedData(indices));
  slGeom.setParam("vertex.color", cpp::CopiedData(colors));
  slGeom.setParam("type", OSP_ROUND);
  slGeom.setParam("basis", OSP_CATMULL_ROM);

  slGeom.commit();

  cpp::GeometricModel model(slGeom);

  if (rendererType == "pathtracer" || rendererType == "scivis"
      || rendererType == "ao") {
    cpp::Material slMat(rendererType, "obj");
    slMat.commit();
    model.setParam("material", slMat);
  }

  model.commit();

  cpp::Group group;

  group.setParam("geometry", cpp::CopiedData(model));
  group.commit();

  return group;
}

OSP_REGISTER_TESTING_BUILDER(Streamlines, streamlines);

} // namespace testing
} // namespace ospray
