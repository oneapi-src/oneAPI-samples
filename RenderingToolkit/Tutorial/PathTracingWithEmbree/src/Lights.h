#pragma once

#ifndef FILE_LIGHTSSEEN
#define FILE_LIGHTSSEEN
/* Added for pathtracer */
#include "Materials.h"
#include "definitions.h"

// for very small cones treat as singular light, because float precision is not
// good enough
#define COS_ANGLE_MAX 0.99999988f

enum class LightType { INFINITE_DIRECTIONAL_LIGHT, POINT_LIGHT, SPOT_LIGHT };

/* Added for pathtracer */
struct Light_SampleRes {
  Vec3fa weight;  //!< radiance that arrives at the given point divided by pdf
  Vec3fa dir;     //!< direction towards the light source
  float dist;     //!< largest valid t_far value for a shadow ray
  float pdf;      //!< probability density that this sample was taken
};

struct Light_EvalRes {
  Vec3fa value;  //!< radiance that arrives at the given point (not weighted by
                 //!< pdf)
  float dist;
  float
      pdf;  //!< probability density that the direction would have been sampled
};

class Light {
 public:
  Light(){};

  virtual Light_SampleRes sample(const DifferentialGeometry& dg,
                                 const Vec2f& randomLightSample) = 0;
  virtual Light_EvalRes eval(const Vec3fa& org, const Vec3fa&);

  LightType m_type;
};

Light_EvalRes Light::eval(const Vec3fa& dg, const Vec3fa& dir) {
  Light_EvalRes res;
  res.value = Vec3fa(0.f);
  res.dist = inf;
  res.pdf = 0.f;
  return res;
}

class PointLight : public Light {
 public:
  PointLight(Vec3fa pos, Vec3fa pow, float r)
      : m_position(pos), m_power(pow), m_radius(r) {
    m_type = LightType::POINT_LIGHT;
  };
  ~PointLight(){};

  Light_SampleRes sample(const DifferentialGeometry& dg, const Vec2f& s);

  Light_EvalRes eval(const Vec3fa& org, const Vec3fa& dir);
  unsigned int add_geometry(
      RTCScene scene, RTCDevice device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);

  inline void clean_geometry();

  Vec3fa m_position;
  Vec3fa m_power;
  float m_radius;
};

Light_SampleRes PointLight::sample(const DifferentialGeometry& dg,
                                   const Vec2f& s) {
  Light_SampleRes res;

  // extant light vector from the hit point
  const Vec3fa dir = m_position - dg.P;
  const float dist2 = dot(dir, dir);
  const float invdist = rsqrt(dist2);

  // normalized light vector
  res.dir = dir * invdist;
  res.dist = dist2 * invdist;

  res.pdf = inf;  // per default we always take this res

  // convert from power to radiance by attenuating by distance^2
  res.weight = m_power * (invdist * invdist);
  const float sinTheta = m_radius * invdist;

  if ((m_radius > 0.f) && (sinTheta > 0.005f)) {
    // res surface of sphere as seen by hit point -> cone of directions
    // for very small cones treat as point light, because float precision is not
    // good enough
    if (sinTheta < 1.f) {
      const float cosTheta = sqrt(1.f - sinTheta * sinTheta);
      const Vec3fa localDir = uniformSampleCone(cosTheta, s);
      res.dir = frame(res.dir) * localDir;
      res.pdf = uniformSampleConePDF(cosTheta);
      const float c = localDir.z;
      res.dist =
          c * res.dist - sqrt((m_radius * m_radius) - (1.f - c * c) * dist2);
      // TODO scale radiance by actual distance
    } else {  // inside sphere
      const Vec3fa localDir = cosineSampleHemisphere(s);
      res.dir = frame(dg.Ns) * localDir;
      res.pdf = cosineSampleHemispherePDF(localDir);
      // TODO:
      res.weight = m_power * rcp(m_radius * m_radius);
      res.dist = m_radius;
    }
  }

  return res;
}

Light_EvalRes PointLight::eval(const Vec3fa& org, const Vec3fa& dir) {
  Light_EvalRes res;
  res.value = Vec3fa(0.f);
  res.dist = inf;
  res.pdf = 0.f;

  if (m_radius > 0.f) {
    const Vec3fa A = m_position - org;
    const float a = dot(dir, dir);
    const float b = 2.f * dot(dir, A);
    const float centerDist2 = dot(A, A);
    const float c = centerDist2 - (m_radius * m_radius);
    const float radical = (b * b) - 4.f * a * c;

    if (radical > 0.f) {
      const float t_near = (b - sqrt(radical)) / (2.f * a);
      const float t_far = (b + sqrt(radical)) / (2.f * a);

      if (t_far > 0.0f) {
        // TODO: handle interior case
        res.dist = t_near;
        const float sinTheta2 = (m_radius * m_radius) * rcp(centerDist2);
        const float cosTheta = sqrt(1.f - sinTheta2);
        res.pdf = uniformSampleConePDF(cosTheta);
        const float invdist = rcp(t_near);
        res.value = m_power * res.pdf * (invdist * invdist);
      }
    }
  }

  return res;
}

unsigned int PointLight::add_geometry(
    RTCScene scene, RTCDevice device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(Vertex), 1);

  // Sphere primitive defined as singular Vec4 point for embree
  Vertex p = {m_position.x, m_position.y, m_position.z, m_radius};
  vertices[0] = p;

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = {MaterialType::MATERIAL_EMITTER};
  // We don't want to store 'albedo' colors for the point light, we will use
  // sample/eval functions and members of the light object
  mpTable.primColorTable = nullptr;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));

  return geomID;
}

/* Only a place holder. Nothing is here because we do not attach any attributes
 * for the sphere.  */
inline void PointLight::clean_geometry() {}

class DirectionalLight : public Light {
 public:
  DirectionalLight(const Vec3fa& _direction, const Vec3fa& _radiance,
                   float _cosAngle);
  ~DirectionalLight(){};

  Light_SampleRes sample(const DifferentialGeometry& dg, const Vec2f& s);

  Light_EvalRes eval(const Vec3fa& org, const Vec3fa& dir);
  LinearSpace3fa m_coordFrame;  //!< coordinate frame, with vz == direction
                                //!< *towards* the light source
  Vec3fa m_radiance;            //!< RGB color and intensity of light
  float m_cosAngle;  //!< Angular limit of the cone light in an easier to use
                     //!< form: cosine of the half angle in radians
  float m_pdf;       //!< Probability to sample a direction to the light
};

//! Set the parameters of an ispc-side DirectionalLight object
DirectionalLight::DirectionalLight(const Vec3fa& _direction,
                                   const Vec3fa& _radiance, float _cosAngle) {
  m_coordFrame = frame(_direction);
  m_radiance = _radiance;
  m_cosAngle = _cosAngle;
  m_pdf = _cosAngle < COS_ANGLE_MAX ? uniformSampleConePDF(_cosAngle) : inf;
  m_type = LightType::INFINITE_DIRECTIONAL_LIGHT;
}

Light_SampleRes DirectionalLight::sample(const DifferentialGeometry& dg,
                                         const Vec2f& s) {
  Light_SampleRes res;

  res.dir = m_coordFrame.vz;
  res.dist = inf;
  res.pdf = m_pdf;

  if (m_cosAngle < COS_ANGLE_MAX)
    res.dir = m_coordFrame * uniformSampleCone(m_cosAngle, s);

  res.weight = m_radiance;  // *pdf/pdf cancel

  return res;
}

Light_EvalRes DirectionalLight::eval(const Vec3fa&, const Vec3fa& dir) {
  Light_EvalRes res;
  res.dist = inf;

  if (m_cosAngle < COS_ANGLE_MAX && dot(m_coordFrame.vz, dir) > m_cosAngle) {
    res.value = m_radiance * m_pdf;
    res.pdf = m_pdf;
  } else {
    res.value = Vec3fa(0.f);
    res.pdf = 0.f;
  }

  return res;
}

class SpotLight : public Light {
 public:
  SpotLight(const Vec3fa& _position, const Vec3fa& _direction,
            const Vec3fa& _power, float _cosAngleMax, float _cosAngleScale,
            float _radius)
      : m_position(_position),
        m_direction(_direction),
        m_power(_power),
        m_cosAngleMax(_cosAngleMax),
        m_cosAngleScale(_cosAngleScale),
        m_radius(_radius)

  {
    m_coordFrame = frame(_direction);
    m_diskPdf = uniformSampleDiskPDF(_radius);
  };
  ~SpotLight(){};
  Light_SampleRes sample(const DifferentialGeometry& dg, const Vec2f& s);

  Light_EvalRes eval(const Vec3fa& org, const Vec3fa& dir);
  unsigned int add_geometry(
      RTCScene scene, RTCDevice device,
      std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim);

  Vec3fa m_position;            //!< Position of the SpotLight
  LinearSpace3fa m_coordFrame;  //!< coordinate frame, with vz == direction that
                                //!< the SpotLight is emitting
  Vec3fa m_power;               //!< RGB color and intensity of the SpotLight
  float m_cosAngleMax;  //!< Angular limit of the spot in an easier to use form:
                        //!< cosine of the half angle in radians
  float m_cosAngleScale;  //!< 1/(cos(border of the penumbra area) -
                          //!< cosAngleMax); positive
  float m_radius;         //!< defines the size of the (extended) SpotLight
  float m_diskPdf;        //!< pdf of disk with radius
  Vec3fa m_direction;     // store the disk direction
};

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

unsigned int SpotLight::add_geometry(
    RTCScene scene, RTCDevice device,
    std::map<unsigned int, MatAndPrimColorTable>& mapGeomToPrim) {
  RTCGeometry mesh =
      rtcNewGeometry(device, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(Vertex), 1);

  // Sphere primitive defined as singular Vec4 point for embree
  Vertex p = {m_position.x, m_position.y, m_position.z, m_radius};
  vertices[0] = p;

  Normal* normals = (Normal*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_NORMAL, 0, RTC_FORMAT_FLOAT3, sizeof(Normal), 1);
  Normal n = {m_direction.x, m_direction.y, m_direction.z};
  normals[0] = n;

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);

  MatAndPrimColorTable mpTable;
  mpTable.materialTable = {MaterialType::MATERIAL_EMITTER};
  // We don't want to store albedo colors for the point light, we will use
  // sample/eval functions and members of the light object
  mpTable.primColorTable = nullptr;
  mapGeomToPrim.insert(std::make_pair(geomID, mpTable));

  return geomID;
}

#endif /* FILE_LIGHTSSEEN */
