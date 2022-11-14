#pragma once

#ifndef FILE_MATERIALSSEEN
#define FILE_MATERIALSSEEN

#include "definitions.h"

/* Added for pathtracer */
enum class MaterialType {
  MATERIAL_MATTE,
  MATERIAL_MIRROR,
  MATERIAL_GLASS,
  MATERIAL_WATER,
  MATERIAL_EMITTER
};

/* Added for pathtracer */
struct Medium {
  /* Just using medium eta value not transmission in this code */
  // Vec3fa transmission;
  float eta;
};

/* Added for path tracer: creating a lookup structure for intersected geometries
 */
struct MatAndPrimColorTable {
  std::vector<MaterialType> materialTable;
  Vec3fa* primColorTable;
};

inline Vec3fa refract(const Vec3fa& V, const Vec3fa& N, const float eta,
                      const float cosi, float& cost, float& refractionPDF) {
  const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
  if (k < 0.0f) {
    cost = 0.0f;
    refractionPDF = 0.f;
    return Vec3fa(0.f);
  }
  cost = sqrt(k);
  refractionPDF = eta * eta;
  return eta * (cosi * N - V) - cost * N;
}

/*! Reflects a viewing vector V at a normal N. */
inline Vec3fa reflect(const Vec3fa& V, const Vec3fa& N) {
  return 2.0f * dot(V, N) * N - V;
}

inline float fresnelDielectric(const float cosi, const float cost,
                               const float eta) {
  const float Rper = (eta * cosi - cost) * rcp(eta * cosi + cost);
  const float Rpar = (cosi - eta * cost) * rcp(cosi + eta * cost);
  return 0.5f * (Rpar * Rpar + Rper * Rper);
}

inline float fresnelDielectric(const float cosi, const float eta) {
  const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
  if (k < 0.0f) return 1.0f;
  const float cost = sqrt(k);

  return fresnelDielectric(cosi, cost, eta);
}

/* Monte Carlo ray tracing "_eval" functions:
   evaluates X for a given set of random variables (direction)
*/

inline Vec3fa Dielectric_eval(const Vec3fa& albedo, const Vec3fa& Lw,
                              const Vec3fa& wo, const DifferentialGeometry& dg,
                              const Vec3fa& wi_v, const Medium& medium,
                              const float dielEta, const float s) {
  float eta = 0.0f;
  Medium mediumOutside;
  mediumOutside.eta = 1.0f;

  Medium mediumInside;
  mediumInside.eta = dielEta;

  Medium mediumFront, mediumBack;
  if (medium.eta == mediumInside.eta) {
    eta = mediumInside.eta / mediumOutside.eta;
    mediumFront = mediumInside;
    mediumBack = mediumOutside;
  } else {
    eta = mediumOutside.eta / mediumInside.eta;
    mediumFront = mediumOutside;
    mediumBack = mediumInside;
  }

  float cosThetaO = clamp(dot(wo, dg.Ns));
  float cosThetaI;

  /* refraction computation */
  Vec3fa refractionDir;
  float refractionPDF;
  refractionDir = refract(wo, dg.Ns, eta, cosThetaO, cosThetaI, refractionPDF);

  /* reflection computation */
  Vec3fa reflectionDir;
  reflectionDir = reflect(wo, dg.Ns);
  float reflectionPDF = 1.0f;

  float R = fresnelDielectric(cosThetaO, cosThetaI, eta);
  Vec3fa cs = Vec3fa(R);
  Vec3fa ct = Vec3fa(1.0f - R);

  const Vec3fa m0 = Lw * cs / reflectionPDF;
  const Vec3fa m1 = Lw * ct / refractionPDF;

  const float C0 = reflectionPDF == 0.0f ? 0.0f : max(max(m0.x, m0.y), m0.z);
  const float C1 = refractionPDF == 0.0f ? 0.0f : max(max(m1.x, m1.y), m1.z);
  const float C = C0 + C1;

  if (C == 0.0f) {
    return Vec3fa(0.f, 0.f, 0.f);
  }

  /* Compare weights for the reflection and the refraction. Pick a direction
   * given s is a random between 0 and 1 */
  const float CP0 = C0 / C;
  if (s < CP0)
    return albedo * cs;
  else
    return albedo * ct;
}

inline Vec3fa Dielectric_eval(const Vec3fa& albedo, const Vec3fa& Lw,
                              const Vec3fa& wo, const DifferentialGeometry& dg,
                              const Vec3fa& wi_v, const Medium& medium,
                              const float dielEta) {
  float eta = 0.0f;
  Medium mediumOutside;
  mediumOutside.eta = 1.0f;

  Medium mediumInside;
  mediumInside.eta = dielEta;

  Medium mediumFront, mediumBack;
  if (medium.eta == mediumInside.eta) {
    eta = mediumInside.eta / mediumOutside.eta;
    mediumFront = mediumInside;
    mediumBack = mediumOutside;
  } else {
    eta = mediumOutside.eta / mediumInside.eta;
    mediumFront = mediumOutside;
    mediumBack = mediumInside;
  }

  float cosThetaO = clamp(dot(wo, dg.Ns));
  float cosThetaI;

  /* refraction computation */
  Vec3fa refractionDir;
  float refractionPDF;
  refractionDir = refract(wo, dg.Ns, eta, cosThetaO, cosThetaI, refractionPDF);

  /* reflection computation */
  Vec3fa reflectionDir;
  reflectionDir = reflect(wo, dg.Ns);
  float reflectionPDF = 1.0f;

  float R = fresnelDielectric(cosThetaO, cosThetaI, eta);
  Vec3fa cs = Vec3fa(R);
  Vec3fa ct = Vec3fa(1.0f - R);

  const Vec3fa m0 = Lw * cs / reflectionPDF;
  const Vec3fa m1 = Lw * ct / refractionPDF;

  const float C0 = reflectionPDF == 0.0f ? 0.0f : max(max(m0.x, m0.y), m0.z);
  const float C1 = refractionPDF == 0.0f ? 0.0f : max(max(m1.x, m1.y), m1.z);
  const float C = C0 + C1;

  if (C == 0.0f) {
    return Vec3fa(0.f, 0.f, 0.f);
  }

  /* Compare weights for the reflection and the refraction. Pick a direction
   * given s is a random between 0 and 1 */
  const float CP0 = C0 / C;
  if (dot(wi_v, dg.Ns) >= 0.f)
    return albedo * cs;
  else
    return albedo * ct;
}

inline Vec3fa Lambertian_eval(const Vec3fa& albedo, const Vec3fa& wo,
                              const DifferentialGeometry& dg,
                              const Vec3fa& wi_v) {
  /* The diffuse material. Reflectance (albedo) times the cosign fall off of the
   * vector about the normal. */
  return albedo * (1.f / (float)(float(M_PI))) * clamp(dot(wi_v, dg.Ns));
}

inline Vec3fa Mirror_eval(const Vec3fa& albedo, const Vec3fa& wo,
                          const DifferentialGeometry& dg, Vec3fa wi_v) {
  return albedo;
}

Vec3fa Material_eval(Vec3fa albedo, MaterialType materialType, const Vec3fa& Lw,
                     const Vec3fa& wo, const DifferentialGeometry& dg,
                     const Vec3fa& wi, const Medium& medium, const Vec2f& s) {
  Vec3fa c = Vec3fa(0.0f);
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return Lambertian_eval(albedo, wo, dg, wi);
      break;
    case MaterialType::MATERIAL_MIRROR:
      return Mirror_eval(albedo, wo, dg, wi);
      break;
    case MaterialType::MATERIAL_GLASS:
      return Dielectric_eval(albedo, Lw, wo, dg, wi, medium, 1.5f, s.x);
      /* Try thin dielectric!? */
      break;
    case MaterialType::MATERIAL_WATER:
      return Dielectric_eval(albedo, Lw, wo, dg, wi, medium, 1.3f, s.x);
      break;
      /* Return our debug color if something goes awry */
    default:
      break;
  }
  return c;
}

Vec3fa Material_eval(Vec3fa albedo, MaterialType materialType, const Vec3fa& Lw,
                     const Vec3fa& wo, const DifferentialGeometry& dg,
                     const Vec3fa& wi, const Medium& medium) {
  Vec3fa c = Vec3fa(0.0f);
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return Lambertian_eval(albedo, wo, dg, wi);
      break;
    case MaterialType::MATERIAL_MIRROR:
      return Mirror_eval(albedo, wo, dg, wi);
      break;
    case MaterialType::MATERIAL_GLASS:
      return Dielectric_eval(albedo, Lw, wo, dg, wi, medium, 1.5f);
      /* Try thin dielectric!? */
      break;
    case MaterialType::MATERIAL_WATER:
      return Dielectric_eval(albedo, Lw, wo, dg, wi, medium, 1.3f);
      break;
      /* Return our debug color if something goes awry */
    default:
      break;
  }
  return c;
}

/* Material Sampling Functions */

Vec3fa Dielectric_sample(const Vec3fa& Lw, const Vec3fa& wo,
                         const DifferentialGeometry& dg, const Medium& medium,
                         float dielEta, Medium& nextMedium, const float s) {
  float eta = 0.0f;
  Medium mediumOutside;
  mediumOutside.eta = 1.0f;

  Medium mediumInside;
  mediumInside.eta = dielEta;

  Medium mediumFront, mediumBack;
  if (medium.eta == mediumInside.eta) {
    eta = mediumInside.eta / mediumOutside.eta;
    mediumFront = mediumInside;
    mediumBack = mediumOutside;
  } else {
    eta = mediumOutside.eta / mediumInside.eta;
    mediumFront = mediumOutside;
    mediumBack = mediumInside;
  }

  float cosThetaO = clamp(dot(wo, dg.Ns));
  float cosThetaI;

  /* refraction computation */
  Vec3fa refractionDir;
  float refractionPDF;
  refractionDir = refract(wo, dg.Ns, eta, cosThetaO, cosThetaI, refractionPDF);

  /* reflection computation */
  Vec3fa reflectionDir;
  reflectionDir = reflect(wo, dg.Ns);
  float reflectionPDF = 1.0f;

  float R = fresnelDielectric(cosThetaO, cosThetaI, eta);
  Vec3fa cs = Vec3fa(R);
  Vec3fa ct = Vec3fa(1.0f - R);

  const Vec3fa m0 = Lw * cs / reflectionPDF;
  const Vec3fa m1 = Lw * ct / refractionPDF;

  const float C0 = reflectionPDF == 0.0f ? 0.0f : max(max(m0.x, m0.y), m0.z);
  const float C1 = refractionPDF == 0.0f ? 0.0f : max(max(m1.x, m1.y), m1.z);
  const float C = C0 + C1;

  if (C == 0.0f) {
    return Vec3fa(0, 0, 0);
  }

  /* Compare weights for the reflection and the refraction. Pick a direction
   * given s is a random between 0 and 1 */
  const float CP0 = C0 / C;
  const float CP1 = C1 / C;
  if (s < CP0) {
    nextMedium = mediumFront;
    return reflectionDir;
  } else {
    nextMedium = mediumBack;
    return refractionDir;
  }
}

Vec3fa Lambertian_sample(const Vec3fa& wo, const DifferentialGeometry& dg,
                         const Vec2f& randomMatSample) {
  return cosineSampleHemisphere(randomMatSample.x, randomMatSample.y, dg.Ns);
}

Vec3fa Mirror_sample(const Vec3fa& Lw, const Vec3fa& wo,
                     const DifferentialGeometry& dg) {
  /* Compute a reflection vector 2 * N.L * N - L */
  return 2.0f * dot(wo, dg.Ns) * dg.Ns - wo;
}

Vec3fa Material_sample(MaterialType materialType, const Vec3fa& Lw,
                       const Vec3fa& wo, const DifferentialGeometry& dg,
                       const Medium& medium, Medium& nextMedium,
                       const Vec2f& randomMatSample) {
  Vec3fa dir = Vec3fa(0.0f);
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return Lambertian_sample(wo, dg, randomMatSample);
      break;
    case MaterialType::MATERIAL_MIRROR:
      return Mirror_sample(Lw, wo, dg);
      break;
    case MaterialType::MATERIAL_GLASS:
      return Dielectric_sample(Lw, wo, dg, medium, 1.5f, nextMedium,
                               randomMatSample.x);
      break;
    case MaterialType::MATERIAL_WATER:
      return Dielectric_sample(Lw, wo, dg, medium, 1.3f, nextMedium,
                               randomMatSample.x);
      break;
    default:
      break;
  }

  return dir;
}

/* Compute PDF */

/* Important to use the same random sample as is used for the
 * Lambertian_sample(..) if looking just for the pdf */
float Lambertian_pdf(const float& s) {
  float cosTheta = sqrt(s);
  return cosTheta / float(M_PI);
}

float Lambertian_pdf(const DifferentialGeometry& dg, const Vec3fa& wi1) {
  return dot(wi1, dg.Ns) / float(M_PI);
}

float Mirror_pdf() { return 1.0f; }

float Dielectric_pdf(const Vec3fa& Lw, const Vec3fa& wo,
                     const DifferentialGeometry& dg, const Medium& medium,
                     const float dielEta, const float s) {
  float pdf = 0.f;

  float eta = 0.0f;
  Medium mediumOutside;
  mediumOutside.eta = 1.0f;

  Medium mediumInside;
  mediumInside.eta = dielEta;

  if (medium.eta == mediumInside.eta) {
    eta = mediumInside.eta / mediumOutside.eta;
  } else {
    eta = mediumOutside.eta / mediumInside.eta;
  }

  float cosThetaO = clamp(dot(wo, dg.Ns));
  float cosThetaI;

  /* refraction computation */
  float refractPDF;
  const float k = 1.0f - eta * eta * (1.0f - cosThetaO * cosThetaO);
  if (k < 0.0f) {
    cosThetaI = 0.0f;
    refractPDF = 0.0f;
  } else {
    cosThetaI = sqrt(k);
    refractPDF = eta * eta;
  }

  /* reflection computation */
  float reflectPDF;
  reflectPDF = 1.0f;

  float R = fresnelDielectric(cosThetaO, cosThetaI, eta);
  Vec3fa cs = Vec3fa(R);
  Vec3fa ct = Vec3fa(1.0f - R);

  const Vec3fa m0 = Lw * cs / reflectPDF;
  const Vec3fa m1 = Lw * ct / refractPDF;

  const float C0 = reflectPDF == 0.0f ? 0.0f : max(max(m0.x, m0.y), m0.z);
  const float C1 = refractPDF == 0.0f ? 0.0f : max(max(m1.x, m1.y), m1.z);
  const float C = C0 + C1;

  if (C == 0.0f) {
    return 0.0f;
  }
  /* Compare weights for the reflection and the refraction. Pick a pdf
   * given s.x is a random between 0 and 1 */
  const float CP0 = C0 / C;
  const float CP1 = C1 / C;
  if (s < CP0) {
    pdf = reflectPDF * CP0;
  } else {
    pdf = refractPDF * CP1;
  }

  return pdf;
}

float Dielectric_pdf(const Vec3fa& Lw, const Vec3fa& wo,
                     const DifferentialGeometry& dg, const Medium& medium,
                     const float dielEta, const Vec3fa& wi1) {
  float pdf = 0.f;

  float eta = 0.0f;
  Medium mediumOutside;
  mediumOutside.eta = 1.0f;

  Medium mediumInside;
  mediumInside.eta = dielEta;

  if (medium.eta == mediumInside.eta) {
    eta = mediumInside.eta / mediumOutside.eta;
  } else {
    eta = mediumOutside.eta / mediumInside.eta;
  }

  float cosThetaO = clamp(dot(wo, dg.Ns));
  float cosThetaI;

  /* refraction computation */
  float refractPDF;
  const float k = 1.0f - eta * eta * (1.0f - cosThetaO * cosThetaO);
  if (k < 0.0f) {
    cosThetaI = 0.0f;
    refractPDF = 0.0f;
  } else {
    cosThetaI = sqrt(k);
    refractPDF = eta * eta;
  }

  /* reflection computation */
  float reflectPDF;
  reflectPDF = 1.0f;

  float R = fresnelDielectric(cosThetaO, cosThetaI, eta);
  Vec3fa cs = Vec3fa(R);
  Vec3fa ct = Vec3fa(1.0f - R);

  const Vec3fa m0 = Lw * cs / reflectPDF;
  const Vec3fa m1 = Lw * ct / refractPDF;

  const float C0 = reflectPDF == 0.0f ? 0.0f : max(max(m0.x, m0.y), m0.z);
  const float C1 = refractPDF == 0.0f ? 0.0f : max(max(m1.x, m1.y), m1.z);
  const float C = C0 + C1;

  if (C == 0.0f) {
    return 0.0f;
  }
  /* Compare weights for the reflection and the refraction. Pick a pdf
   * given s.x is a random between 0 and 1 */
  const float CP0 = C0 / C;
  const float CP1 = C1 / C;
  if (dot(wi1, dg.Ns) >= 0.f) {
    pdf = reflectPDF * CP0;
  } else {
    pdf = refractPDF * CP1;
  }

  return pdf;
}

/* Determine Probability Density Function for Materials */

float Material_pdf(MaterialType materialType, const Vec3fa& Lw,
                   const Vec3fa& wo, const DifferentialGeometry& dg,
                   const Medium& medium, const Vec2f& randomSample) {
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return Lambertian_pdf(randomSample.y);
      break;
    case MaterialType::MATERIAL_MIRROR:
      return Mirror_pdf();
      break;
    case MaterialType::MATERIAL_GLASS:
      return Dielectric_pdf(Lw, wo, dg, medium, 1.5f, randomSample.x);
      break;
    case MaterialType::MATERIAL_WATER:
      return Dielectric_pdf(Lw, wo, dg, medium, 1.3f, randomSample.x);
      break;
    default:
      break;
  }
  return 0.f;
}

float Material_pdf(MaterialType materialType, const Vec3fa& Lw,
                   const Vec3fa& wo, const DifferentialGeometry& dg,
                   const Medium& medium, const Vec3fa& wi1) {
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return Lambertian_pdf(dg, wi1);
      break;
    case MaterialType::MATERIAL_MIRROR:
      return Mirror_pdf();
      break;
    case MaterialType::MATERIAL_GLASS:
      return Dielectric_pdf(Lw, wo, dg, medium, 1.5f, wi1);
      break;
    case MaterialType::MATERIAL_WATER:
      return Dielectric_pdf(Lw, wo, dg, medium, 1.3f, wi1);
      break;
    default:
      break;
  }
  return 0.f;
}

inline bool Material_direct_illumination(MaterialType materialType) {
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return true;
      break;
    case MaterialType::MATERIAL_MIRROR:
      return false;
      break;
    case MaterialType::MATERIAL_GLASS:
    case MaterialType::MATERIAL_WATER:
      return false;
      break;

    default:
      break;
  }
  return false;
}

#endif /* FILE_MATERIALSSEEN */
