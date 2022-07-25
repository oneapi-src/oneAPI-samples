#pagma once
/// cosine-weighted sampling of hemisphere oriented along the +z-axis
////////////////////////////////////////////////////////////////////////////////

/* Added for pathtracer transforming a normal */
LinearSpace3fa frame(const Vec3fa& N) {
  const Vec3fa dx0(0, N.z, -N.y);
  const Vec3fa dx1(-N.z, 0, N.x);

  const Vec3fa dx = normalize((dot(dx0, dx0) > dot(dx1, dx1)) ? dx0 : dx1);
  const Vec3fa dy = normalize(cross(N, dx));

  return LinearSpace3fa(dx, dy, N);
}

/*! Cosine weighted hemisphere sampling. Up direction is provided as argument.
 */
inline Sample3f cosineSampleHemisphere(const float u, const float v,
                                       const Vec3fa& N) {
  /* Determine cartesian coordinate for new Vec3fa */
  const float phi = float(2.0 * M_PI) * u;
  const float cosTheta = sqrt(v);
  const float sinTheta = sqrt(1.0f - v);
  const float sinPhi = sinf(phi);
  const float cosPhi = cosf(phi);

  Vec3fa localDir = Vec3fa(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
  /* Gives the new Vec3fa transformed about the input Vec3fa */
  Sample3f s;
  s.v = frame(N) * localDir;

  /* Gives a smooth pdf */
  s.pdf = localDir.z / float(M_PI);
  return s;
}

Vec3fa Lambertian__eval(const Vec3fa& R, const Vec3fa& wo,
                        DifferentialGeometry dg, Vec3fa wi_v) {
  /* The diffuse material. Reflectance (albedo) times the cosign fall off of the
   * vector about the normal. */
  return R * (1.f / (float)(float(M_PI))) * clamp(dot(wi_v, dg.Ns));
}

Vec3fa Mirror__sample(const Vec3fa& R, const Vec3fa& Lw, const Vec3fa& wo,
                      const DifferentialGeometry& dg, Sample3f& wi) {
  Sample3f sam;
  sam.pdf = 1.0f;
  /* Compute a reflection vector 2 * N.L * N - L */
  sam.v = normalize(2.0f * dot(wo, dg.Ns) * dg.Ns - wo);
  wi = sam;
  return R;
}

inline Vec3fa sample_component2(const Vec3fa& c0, const Sample3f& wi0,
                                const Medium& medium0, const Vec3fa& c1,
                                const Sample3f& wi1, const Medium& medium1,
                                const Vec3fa& Lw, Sample3f& wi_o,
                                Medium& medium_o, const float s) {
  const Vec3fa m0 = Lw * c0 / wi0.pdf;
  const Vec3fa m1 = Lw * c1 / wi1.pdf;

  const float C0 = wi0.pdf == 0.0f ? 0.0f : max(max(m0.x, m0.y), m0.z);
  const float C1 = wi1.pdf == 0.0f ? 0.0f : max(max(m1.x, m1.y), m1.z);
  const float C = C0 + C1;

  Sample3f ret;
  if (C == 0.0f) {
    wi_o.v = Vec3fa(0, 0, 0);
    wi_o.pdf = 0.0f;

    return Vec3fa(0, 0, 0);
  }

  /* Compare weights for the reflection and the refraction. Pick a direction
   * given s is a random between 0 and 1 */
  const float CP0 = C0 / C;
  const float CP1 = C1 / C;
  if (s < CP0) {
    wi_o.v = wi0.v;
    wi_o.pdf = wi0.pdf * CP0;

    medium_o = medium0;
    return c0;
  } else {
    wi_o.v = wi1.v;
    wi_o.pdf = wi1.pdf * CP1;

    medium_o = medium1;
    return c1;
  }
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

Vec3fa Dielectric__sample(const Vec3fa& Lw, const Vec3fa& wo,
                          const DifferentialGeometry& dg, Sample3f& wi_o,
                          Medium& medium, const Vec2f& s) {
  float eta = 0.0f;
  Medium mediumOutside;
  mediumOutside.eta = 1.0f;
  mediumOutside.transmission = Vec3fa(1.f);

  Medium mediumInside;
  mediumInside.eta = 1.4f;
  mediumInside.transmission = Vec3fa(1.f);

  Medium mediumFront, mediumBack;
  if (medium.eta == mediumInside.eta &&
      medium.transmission == mediumInside.transmission) {
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
  Sample3f wit;

  const float k = 1.0f - eta * eta * (1.0f - cosThetaO * cosThetaO);
  if (k < 0.0f) {
    cosThetaI = 0.0f;
    wit.v = Vec3fa(0.f);
    wit.pdf = 0.0f;
  }
  cosThetaI = sqrt(k);
  wit.v = eta * (cosThetaO * dg.Ns - wo) - cosThetaI * dg.Ns;
  wit.pdf = eta * eta;

  /* reflection computation */
  Sample3f wis;
  wis.v = normalize(2.0f * dot(wo, dg.Ns) * dg.Ns - wo);
  wis.pdf = 1.0f;

  float R = fresnelDielectric(cosThetaO, cosThetaI, eta);
  Vec3fa cs = Vec3fa(R);
  Vec3fa ct = Vec3fa(1.0f - R);
  return sample_component2(cs, wis, mediumFront, ct, wit, mediumBack, Lw, wi_o,
                           medium, s.x);
}

Vec3fa ThinDielectric__sample(const Vec3fa& Lw, const Vec3fa& wo,
                              const DifferentialGeometry& dg, Sample3f& wi_o,
                              Medium& medium, const Vec2f& s) {
  /* eta for glass is between 1.4 and 1.8. */
  const float eta = 1.4f;
  const float thickness = 0.1f;
  const Vec3fa transmission = Vec3fa(1.0f);
  const Vec3fa transmissionFactor =
      Vec3fa(logf(transmission.x), logf(transmission.y), logf(transmission.z)) *
      thickness;

  float cosThetaO = clamp(dot(wo, dg.Ns));
  if (cosThetaO <= 0.0f) return Vec3fa(0.0f);
  float R = fresnelDielectric(cosThetaO, rcp(eta));
  Sample3f wit;
  wit.pdf = 1.f;
  wit.v = -wo;

  Sample3f wis;
  wis.pdf = 1.f;
  wis.v = normalize(2.0f * dot(wo, dg.Ns) * dg.Ns - wo);

  Vec3fa ct = Vec3fa(exp(transmissionFactor.x * rcp(cosThetaO)),
                     exp(transmissionFactor.y * rcp(cosThetaO)),
                     exp(transmissionFactor.z * rcp(cosThetaO))) *
              Vec3fa(1.0f - R);
  Vec3fa cs = Vec3fa(R);
  /* With the thin dialectric, we use the same medium for the space between
   * geometry. However this could be extended for Dielectics that have
   * significant mass (thick glass/water) */
  return sample_component2(cs, wis, medium, ct, wit, medium, Lw, wi_o, medium,
                           s.x);
}

Vec3fa Material__sample(Vec3fa R, enum class MaterialType materialType,
                        const Vec3fa& Lw, const Vec3fa& wo,
                        const DifferentialGeometry& dg, Sample3f& wi,
                        Medium& medium, const Vec2f& randomMatSample) {
  Vec3fa c = Vec3fa(0.0f);
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      wi = cosineSampleHemisphere(randomMatSample.x, randomMatSample.y, dg.Ns);
      return Lambertian__eval(R, wo, dg, wi.v);
      break;

    case MaterialType::MATERIAL_MIRROR:
      return Mirror__sample(R, Lw, wo, dg, wi);
      break;
    case MaterialType::MATERIAL_GLASS:
      return Dielectric__sample(Lw, wo, dg, wi, medium, randomMatSample);
      //        return ThinDielectric__sample(Lw, wo, dg, wi, medium,
      //        randomMatSample);
      break;
      /* Return our debug color if something goes awry */
    default:
      c = R;
      break;
  }

  return c;
}

Vec3fa Material__eval(Vec3fa R, enum class MaterialType materialType,
                      const Vec3fa& wo, const DifferentialGeometry& dg,
                      const Vec3fa& wi) {
  Vec3fa c = Vec3fa(0.0f);
  switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
      return Lambertian__eval(R, wo, dg, wi);
      break;
    case MaterialType::MATERIAL_MIRROR:
      return Vec3fa(0.0f);
      break;
    case MaterialType::MATERIAL_GLASS:
      return Vec3fa(0.0f);
      break;
      /* Return our debug color if something goes awry */
    default:
      c = R;
      break;
  }
  return c;
}

float Material__pdf(Vec3fa R, enum class MaterialType materialType,
                      const Vec3fa& wo, const DifferentialGeometry& dg,
                      const Vec3fa& wi){

}

/*

// evaluates X for a given set of random variables (direction) 
X_eval()

// generates a random variable (direction) based on X (material, light source)
X_sample()

// return the PDF for a given random variable (direction) when it would have been samples based on X
X_pdf()

*/

