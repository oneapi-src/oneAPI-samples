#pragma once

#ifndef FILE_LIGHTSSEEN
#define FILE_LIGHTSSEEN
/* Added for pathtracer */
#include "definitions.h"

// for very small cones treat as singular light, because float precision is not good enough
#define COS_ANGLE_MAX 0.99999988f

enum class LightType {
    INFINITE_DIRECTIONAL_LIGHT,
    POINT_LIGHT,
    SPOT_LIGHT
};


/* Added for pathtracer */
struct Light_SampleRes {
    Vec3fa weight;  //!< radiance that arrives at the given point divided by pdf
    Vec3fa dir;     //!< direction towards the light source
    float dist;     //!< largest valid t_far value for a shadow ray
    float pdf;      //!< probability density that this sample was taken
};

struct Light_EvalRes
{
    Vec3fa value;     //!< radiance that arrives at the given point (not weighted by pdf)
    float dist;
    float pdf;       //!< probability density that the direction would have been sampled
};

class Light {
public:
    Light() {};

    virtual Light_SampleRes sample(const DifferentialGeometry& dg, const Vec2f& randomLightSample) = 0;
    virtual Light_EvalRes eval(const Vec3fa& org,
        const Vec3fa&);

    LightType type;
};

Light_EvalRes Light::eval(const Vec3fa& dg,
    const Vec3fa& dir)
{
    Light_EvalRes res;
    res.value = Vec3fa(0.f);
    res.dist = inf;
    res.pdf = 0.f;
    return res;
}

class PointLight : public Light  {
public:
    PointLight(Vec3fa pos, Vec3fa pow, float r) : m_position(pos), power(pow), m_radius(r) { this->type = LightType::POINT_LIGHT; };
    ~PointLight() {};

    Light_SampleRes sample(const DifferentialGeometry& dg, const Vec2f& s);
    //Light_EvalRes eval(
    //    const DifferentialGeometry& dg,
    //    const Vec3fa& dir);
Light_EvalRes eval(
        const Vec3fa& org,
        const Vec3fa& dir);
    unsigned int addGeometry(RTCScene scene, RTCDevice device);

    inline void cleanGeometry();

    Vec3fa m_position;
    Vec3fa power;
    float m_radius;

};


Light_SampleRes PointLight::sample(const DifferentialGeometry& dg, const Vec2f& s) {

    Light_SampleRes res;

    // extant light vector from the hit point
    const Vec3fa dir = m_position - dg.P;
    const float dist2 = dot(dir, dir);
    const float invdist = rsqrt(dist2);

    // normalized light vector
    res.dir = dir * invdist;
    res.dist = dist2 * invdist;

    res.pdf = inf; // per default we always take this res

    // convert from power to radiance by attenuating by distance^2
    res.weight = this->power * (invdist * invdist);
    const float sinTheta = m_radius * invdist;

    if ((m_radius > 0.f) && (sinTheta > 0.005f)) {
        // res surface of sphere as seen by hit point -> cone of directions
        // for very small cones treat as point light, because float precision is not good enough
        if (sinTheta < 1.f) {
            const float cosTheta = sqrt(1.f - sinTheta * sinTheta);
            const Vec3fa localDir = uniformSampleCone(cosTheta, s);
            res.dir = frame(res.dir) * localDir;
            res.pdf = uniformSampleConePDF(cosTheta);
            const float c = localDir.z;
            res.dist = c * res.dist - sqrt((m_radius * m_radius) - (1.f - c * c) * dist2);
            // TODO scale radiance by actual distance
        }
        else { // inside sphere
            const Vec3fa localDir = cosineSampleHemisphere(s);
            res.dir = frame(dg.Ns) * localDir;
            res.pdf = cosineSampleHemispherePDF(localDir);
            // TODO:
            res.weight = this->power * rcp(m_radius*m_radius);
            res.dist = m_radius;
        }
    }

    return res;
}

//Light_EvalRes PointLight::eval(
//    const DifferentialGeometry& dg,
//    const Vec3fa& dir)
    Light_EvalRes PointLight::eval(
        const Vec3fa& org,
        const Vec3fa& dir)
{
    Light_EvalRes res;
    res.value = Vec3fa(0.f);
    res.dist = inf;
    res.pdf = 0.f;

    if (m_radius > 0.f) {
        const Vec3fa A = m_position - org;
        const float a = dot(dir, dir);
        const float b = 2.f * dot(dir, A);
        const float centerDist2 = dot(A, A);
        const float c = centerDist2 - (m_radius*m_radius);
        const float radical = (b*b) - 4.f * a * c;

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
                res.value = this->power * res.pdf * (invdist* invdist);
            }
        }
    }

    return res;
}

unsigned int PointLight::addGeometry(RTCScene scene, RTCDevice device) {

        RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
        Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
            mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(Vertex), 1);

        // Sphere primitive defined as singular Vec4 point for embree
        Vertex p = { m_position.x, m_position.y, m_position.z, m_radius };
        vertices[0] = p;

        rtcCommitGeometry(mesh);
        unsigned int geomID = rtcAttachGeometry(scene, mesh);
        rtcReleaseGeometry(mesh);

        MatAndPrimColorTable mpTable;
        mpTable.materialTable = { MaterialType::MATERIAL_EMITTER };
        //We don't want to store albedo colors for the point light, we will use sample/eval functions and members of the light object
        mpTable.primColorTable = nullptr;
        g_geomIDs.insert(std::make_pair(geomID, mpTable));

        return geomID;




}

/* Only a place holder. Nothing is here because we do not attach any attributes
 * for the sphere.  */
inline void PointLight::cleanGeometry() {}

class DirectionalLight : public Light {
public:
    DirectionalLight(const Vec3fa& _direction, const Vec3fa& _radiance, float _cosAngle);
    ~DirectionalLight() {};

    Light_SampleRes sample(const DifferentialGeometry& dg, const Vec2f& s);
    //Light_EvalRes eval(
    //    const DifferentialGeometry&,
    //    const Vec3fa& dir);
        Light_EvalRes eval(
        const Vec3fa& org,
        const Vec3fa& dir);
    LinearSpace3fa coordFrame;   //!< coordinate frame, with vz == direction *towards* the light source
    Vec3fa radiance;   //!< RGB color and intensity of light
    float cosAngle;   //!< Angular limit of the cone light in an easier to use form: cosine of the half angle in radians
    float pdf;        //!< Probability to sample a direction to the light
};

//! Set the parameters of an ispc-side DirectionalLight object
DirectionalLight::DirectionalLight(const Vec3fa& _direction, const Vec3fa& _radiance, float _cosAngle)
{
    this->coordFrame = frame(_direction);
    this->radiance = _radiance;
    this->cosAngle = _cosAngle;
    this->pdf = _cosAngle < COS_ANGLE_MAX ? uniformSampleConePDF(_cosAngle) : inf;
    this->type = LightType::INFINITE_DIRECTIONAL_LIGHT;
}


Light_SampleRes DirectionalLight::sample(const DifferentialGeometry& dg, const Vec2f& s) {
    Light_SampleRes res;

    res.dir = this->coordFrame.vz;
    res.dist = inf;
    res.pdf = this->pdf;

    if (this->cosAngle < COS_ANGLE_MAX)
        res.dir = this->coordFrame * uniformSampleCone(this->cosAngle, s);

    res.weight = this->radiance; // *pdf/pdf cancel

    return res;
}

Light_EvalRes DirectionalLight::eval(
    const Vec3fa&,
    const Vec3fa& dir)
{

    Light_EvalRes res;
    res.dist = inf;

    if (this->cosAngle < COS_ANGLE_MAX && dot(this->coordFrame.vz, dir) > this->cosAngle) {
        res.value = this->radiance * this->pdf;
        res.pdf = this->pdf;
    }
    else {
        res.value = Vec3fa(0.f);
        res.pdf = 0.f;
    }
    
    return res;
}

class SpotLight : public Light {
public:
    SpotLight(const Vec3fa& _position, const Vec3fa& _direction, const Vec3fa&
        _power, float _cosAngleMax, float _cosAngleScale, float radius) :
        position(_position),
        power(power),
        cosAngleMax(cosAngleMax),
        cosAngleScale(cosAngleScale),
        m_radius(radius) 
    {

        this->coordFrame = frame(_direction);
        this->diskPdf = uniformSampleDiskPDF(radius);
    };
    ~SpotLight() {};
    Light_SampleRes sample(
        const DifferentialGeometry& dg,
        const Vec2f& s);
    Light_EvalRes eval(
            const DifferentialGeometry& dg,
            const Vec3fa& dir);

    Vec3fa position;         //!< Position of the SpotLight
    LinearSpace3fa coordFrame;         //!< coordinate frame, with vz == direction that the SpotLight is emitting
    Vec3fa power;            //!< RGB color and intensity of the SpotLight
    float cosAngleMax;      //!< Angular limit of the spot in an easier to use form: cosine of the half angle in radians
    float cosAngleScale;    //!< 1/(cos(border of the penumbra area) - cosAngleMax); positive
    float m_radius;           //!< defines the size of the (extended) SpotLight
    float diskPdf;          //!< pdf of disk with radius

};


Light_SampleRes SpotLight::sample(
    const DifferentialGeometry& dg,
    const Vec2f& s)
{
    Light_SampleRes res;

    // extant light vector from the hit point
    res.dir = this->position - dg.P;

    if (m_radius > 0.f)
        res.dir = this->coordFrame * uniformSampleDisk(m_radius, s) + res.dir;

    const float dist2 = dot(res.dir, res.dir);
    const float invdist = rsqrt(dist2);

    // normalized light vector
    res.dir = res.dir * invdist;
    res.dist = dist2 * invdist;

    // cosine of the negated light direction and light vector.
    const float cosAngle = -dot(this->coordFrame.vz, res.dir);
    const float angularAttenuation = clamp((cosAngle - this->cosAngleMax) * this->cosAngleScale);

    if (m_radius > 0.f)
        res.pdf = this->diskPdf * dist2 * abs(cosAngle);
    else
        res.pdf = inf; // we always take this res

      // convert from power to radiance by attenuating by distance^2; attenuate by angle
    res.weight = this->power * ((invdist*invdist) * angularAttenuation);

    return res;
}

Light_EvalRes SpotLight::eval(
    const DifferentialGeometry& dg,
    const Vec3fa& dir)
{
    Light_EvalRes res;
    res.value = Vec3fa(0.f);
    res.dist = inf;
    res.pdf = 0.f;

    if (m_radius > 0.f) {
        // intersect disk
        const float cosAngle = -dot(dir, this->coordFrame.vz);
        if (cosAngle > this->cosAngleMax) { // inside illuminated cone?
            const Vec3fa vp = dg.P - this->position;
            const float dp = dot(vp, this->coordFrame.vz);
            if (dp > 0.f) { // in front of light?
                const float t = dp * rcp(cosAngle);
                const Vec3fa vd = vp + t * dir;
                if (dot(vd, vd) < (m_radius*m_radius)) { // inside disk?
                    const float angularAttenuation = min((cosAngle - this->cosAngleMax) * this->cosAngleScale, 1.f);
                    const float pdf = this->diskPdf * cosAngle;
                    res.value = this->power * (angularAttenuation * pdf); // *sqr(t)/sqr(t) cancels
                    res.dist = t;
                    res.pdf = pdf * (t*t);
                }
            }
        }
    }

    return res;
}

#endif /* FILE_LIGHTSSEEN */