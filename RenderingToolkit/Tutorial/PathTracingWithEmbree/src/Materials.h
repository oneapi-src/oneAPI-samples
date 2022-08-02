#pragma once


#ifndef FILE_MATERIALSSEEN
#define FILE_MATERIALSSEEN

/* Added for pathtracer */
enum class MaterialType {
    MATERIAL_MATTE,
    MATERIAL_MIRROR,
    MATERIAL_GLASS,
};

/* Added for pathtracer */
struct Medium {
    /* Just using medium eta value not transmission in this code */
    //Vec3fa transmission;
    float eta;
};

/* Added for path tracer: creating a lookup structure for intersected geometries */
struct MatAndPrimColorTable {
    std::vector<enum class MaterialType> materialTable;
    Vec3fa* primColorTable;
};

/* Added for path tracer: for holding material properties for each geometry id */
std::map< unsigned int, MatAndPrimColorTable> g_geomIDs;

/* Added for pathtracer. The frame function creates a transform from a normal. */
LinearSpace3fa frame(const Vec3fa& N) {
  const Vec3fa dx0(0, N.z, -N.y);
  const Vec3fa dx1(-N.z, 0, N.x);

  const Vec3fa dx = normalize((dot(dx0, dx0) > dot(dx1, dx1)) ? dx0 : dx1);
  const Vec3fa dy = normalize(cross(N, dx));

  return LinearSpace3fa(dx, dy, N);
}

/*! Cosine weighted hemisphere sampling. Up direction is provided as argument.
 */
inline Vec3fa cosineSampleHemisphere(const float u, const float v,
                                       const Vec3fa& N) {
  /* Determine cartesian coordinate for new Vec3fa */
  const float phi = float(2.0 * M_PI) * u;
  const float cosTheta = sqrt(v);
  const float sinTheta = sqrt(1.0f - v);
  const float sinPhi = sinf(phi);
  const float cosPhi = cosf(phi);

  Vec3fa localDir = Vec3fa(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
  /* Gives the new Vec3fa transformed about the input Vec3fa */

  return frame(N) * localDir;

}

inline Vec3fa cosinePDFHemisphere(const float s) {
    return sqrt(s) / float(M_PI);

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

inline Vec3fa Dielectric_eval(const Vec3fa& albedo, const Vec3fa& Lw, const Vec3fa& wo,
    const DifferentialGeometry& dg, Vec3fa wi_v, const Medium& medium, const float& s) {
    
    float eta = 0.0f;
    Medium mediumOutside;
    mediumOutside.eta = 1.0f;

    Medium mediumInside;
    mediumInside.eta = 1.4f;

    Medium mediumFront, mediumBack;
    if (medium.eta == mediumInside.eta) {
        eta = mediumInside.eta / mediumOutside.eta;
        mediumFront = mediumInside;
        mediumBack = mediumOutside;
    }
    else {
        eta = mediumOutside.eta / mediumInside.eta;
        mediumFront = mediumOutside;
        mediumBack = mediumInside;
    }

    float cosThetaO = clamp(dot(wo, dg.Ns));
    float cosThetaI;

    /* refraction computation */
    Vec3fa refractionDir;
    float refractionPDF;
    const float k = 1.0f - eta * eta * (1.0f - cosThetaO * cosThetaO);
    if (k < 0.0f) {
        cosThetaI = 0.0f;
        refractionDir = Vec3fa(0.f);
        refractionPDF = 0.f;
    }
    else {
        cosThetaI = sqrt(k);
        refractionDir = eta * (cosThetaO * dg.Ns - wo) - cosThetaI * dg.Ns;
        refractionPDF = eta * eta;
    }

    /* reflection computation */
    Vec3fa reflectionDir;
    reflectionDir = 2.0f * dot(wo, dg.Ns) * dg.Ns - wo;
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

inline Vec3fa Lambertian_eval(const Vec3fa& albedo, const Vec3fa& wo,
    const DifferentialGeometry& dg, const Vec3fa& wi_v) {
    /* The diffuse material. Reflectance (albedo) times the cosign fall off of the
     * vector about the normal. */
    return albedo * (1.f / (float)(float(M_PI))) * clamp(dot(wi_v, dg.Ns));
}

inline Vec3fa Mirror_eval(const Vec3fa& albedo, const Vec3fa& wo,
    const DifferentialGeometry& dg, Vec3fa wi_v) {
    return albedo;
}

Vec3fa Material_eval(Vec3fa albedo, MaterialType materialType,
    const Vec3fa& Lw, const Vec3fa& wo, const DifferentialGeometry& dg,
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
        return Dielectric_eval(albedo, Lw, wo, dg, wi, medium, s.x);
        /* Try thin dielectric!? */
        break;
        /* Return our debug color if something goes awry */
    default:
        break;
    }
    return c;
}


/* Material Sampling Functions */

Vec3fa Dielectric_sample(const Vec3fa& Lw, const Vec3fa& wo,
                          const DifferentialGeometry& dg,
                          Medium& medium, const float& s) {
  float eta = 0.0f;
  Medium mediumOutside;
  mediumOutside.eta = 1.0f;

  Medium mediumInside;
  mediumInside.eta = 1.4f;

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
  const float k = 1.0f - eta * eta * (1.0f - cosThetaO * cosThetaO);
  if (k < 0.0f) {
    cosThetaI = 0.0f;
    refractionDir = Vec3fa(0.f);
    refractionPDF = 0.f;
  }
  else {
      cosThetaI = sqrt(k);
      refractionDir = eta * (cosThetaO * dg.Ns - wo) - cosThetaI * dg.Ns;
      refractionPDF = eta * eta;
  }

  /* reflection computation */
  Vec3fa reflectionDir;
  reflectionDir = 2.0f * dot(wo, dg.Ns) * dg.Ns - wo;
  float reflectionPDF = 1.0f;

  float R = fresnelDielectric(cosThetaO, cosThetaI, eta);
  Vec3fa cs = Vec3fa(R);
  Vec3fa ct = Vec3fa(1.0f - R);

  const Vec3fa m0 = Lw * cs / reflectionPDF;
  const Vec3fa m1 = Lw * ct / refractionPDF;

  const float C0 = reflectionPDF == 0.0f ? 0.0f : max(max(m0.x, m0.y), m0.z);
  const float C1 = refractionPDF == 0.0f ? 0.0f : max(max(m1.x, m1.y), m1.z);
  const float C = C0 + C1;

  Sample3f ret;
  if (C == 0.0f) {
      return Vec3fa(0, 0, 0);
  }

  /* Compare weights for the reflection and the refraction. Pick a direction
   * given s is a random between 0 and 1 */
  const float CP0 = C0 / C;
  const float CP1 = C1 / C;
  if (s < CP0) {
      medium = mediumFront;
      return reflectionDir;
  }
  else {
      medium = mediumBack;
      return refractionDir;
  }
}

Vec3fa Lambertian_sample(
    const Vec3fa& wo, const DifferentialGeometry& dg,
    const Vec2f& randomMatSample) {

    return cosineSampleHemisphere(randomMatSample.x, randomMatSample.y, dg.Ns);

}

Vec3fa Mirror_sample(const Vec3fa& Lw, const Vec3fa& wo,
    const DifferentialGeometry& dg) {
    /* Compute a reflection vector 2 * N.L * N - L */
    return 2.0f * dot(wo, dg.Ns) * dg.Ns - wo;
}

Vec3fa Material_sample(MaterialType materialType,
                        const Vec3fa& Lw, const Vec3fa& wo,
                        const DifferentialGeometry& dg,
                        Medium& medium, const Vec2f& randomMatSample) {
  Vec3fa dir = Vec3fa(0.0f);
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return Lambertian_sample(wo, dg, randomMatSample);
      break;
    case MaterialType::MATERIAL_MIRROR:
      return Mirror_sample(Lw, wo, dg);
      break;
    case MaterialType::MATERIAL_GLASS:
      return Dielectric_sample(Lw, wo, dg, medium, randomMatSample.x);
      //        return ThinDielectric__sample(Lw, wo, dg, wi, medium,
      //        randomMatSample);
      break;
      /* Return our debug color if something goes awry */
    default:
      break;
  }

  return dir;
}


/* Compute PDF */

/* Important to use the same random sample as is used for the Lambertian_sample(..) if looking just for the pdf */
float Lambertian_pdf(const float& s) {
    float cosTheta = sqrt(s);
    return cosTheta / float(M_PI);
}
float Mirror_pdf() {
    return 1.0f;
}

float Dielectric_pdf(const Vec3fa& Lw, const Vec3fa& wo,
    const DifferentialGeometry& dg,
    const Medium& medium, const float& s) {
    float pdf = 0.f;

    float eta = 0.0f;
    Medium mediumOutside;
    mediumOutside.eta = 1.0f;

    Medium mediumInside;
    mediumInside.eta = 1.4f;

    if (medium.eta == mediumInside.eta ) {
        eta = mediumInside.eta / mediumOutside.eta;
    }
    else {
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
    }
    else {
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
    }
    else {
        pdf = refractPDF * CP1;
    }

    return pdf;
}




/* Determine Probability Density Function for Materials */

float Material_pdf(MaterialType materialType, const Vec3fa& Lw,
                      const Vec3fa& wo, const DifferentialGeometry& dg,
                      const Medium& medium, const Vec2f& randomSample){

    switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
        return Lambertian_pdf(randomSample.y);
        break;
    case MaterialType::MATERIAL_MIRROR:
        return Mirror_pdf();
        break;
    case MaterialType::MATERIAL_GLASS:
        return Dielectric_pdf(Lw, wo,
            dg, medium, randomSample.x);
        break;
        /* Return our debug color if something goes awry */
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
        return false;
        /* Try thin dielectric!? */
        break;
    default:
        break;
    }
    return false;
}
/*

// evaluates X for a given set of random variables (direction) 
X_eval()

// generates a random variable (direction) based on X (material, light source)
X_sample()

// return the PDF for a given random variable (direction) when it would have been samples based on X
X_pdf()

*/

#endif /* FILE_MATERIALSSEEN */