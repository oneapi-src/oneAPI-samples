struct PathTracer{

}

struct GuidedPathTracer{
    //
    Device guiding_device;
    Field guding_field;
    SampleDataStorage guiding sample_data_storage;
}


/* Added for pathtracer */
inline void initRayHit(RTCRayHit& rayhit, const Vec3fa& org, const Vec3fa& dir,
                       float tnear, float tfar, float time) {
  rayhit.ray.dir_x = dir.x;
  rayhit.ray.dir_y = dir.y;
  rayhit.ray.dir_z = dir.z;
  rayhit.ray.org_x = org.x;
  rayhit.ray.org_y = org.y;
  rayhit.ray.org_z = org.z;
  rayhit.ray.tnear = tnear;
  rayhit.ray.time = time;
  rayhit.ray.tfar = tfar;
  rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
  rayhit.ray.mask = -1;
}

/* Added for pathtracer */
inline Vec3fa face_forward(const Vec3fa& dir, const Vec3fa& _Ng) {
  const Vec3fa Ng = _Ng;
  return dot(dir, Ng) < 0.0f ? Ng : -Ng;
}


/* task that renders a single screen pixel */
Vec3fa PathTracer_renderPixelFunction(float x, float y, const unsigned int width,
                           const unsigned int height, RandomEngine& reng,
                           std::uniform_real_distribution<float>& distrib,
                           const float time, const AffineSpace3fa& camera) {
  RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  Vec3fa dir = normalize(x * camera.l.vx + y * camera.l.vy + camera.l.vz);
  Vec3fa org = Vec3fa(camera.p.x, camera.p.y, camera.p.z);

  /* initialize ray */
  RTCRayHit rayhit;
  initRayHit(rayhit, org, dir, 0.0f, std::numeric_limits<float>::infinity(),
             time);

  Vec3fa L = Vec3fa(0.0f);
  Vec3fa Lw = Vec3fa(1.0f);
  /* Create vaccum medium.This helps for refractions passing through different
   * mediums... like glass(dielectric) material */
  Medium medium;
  medium.transmission = Vec3fa(1.0f);
  medium.eta = 1.f;

  DifferentialGeometry dg;

  /* iterative path tracer loop */
  for (int i = 0; i < g_max_path_length; i++) {
    /* terminate if contribution too low */
    if (max(Lw.x, max(Lw.y, Lw.z)) < 0.01f) break;

    /* intersect ray with scene */
    context.flags =
        (i == 0)
            ? RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_COHERENT
            : RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
    rtcIntersect1(g_scene, &context, &rayhit);
    const Vec3fa wo = -dir;

    /* if nothing hit the path is terminated, this could be an light lookup
     * insteead */
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) break;

    Vec3fa Ng = Vec3fa(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
    Vec3fa Ns = normalize(Ng);

    /* compute differential geometry */
    for (int i = 0; i < RTC_MAX_INSTANCE_LEVEL_COUNT; i++)
      dg.instIDs[i] = rayhit.hit.instID[i];

    dg.geomID = rayhit.hit.geomID;
    dg.primID = rayhit.hit.primID;
    dg.u = rayhit.hit.u;
    dg.v = rayhit.hit.v;

    dg.P = org + rayhit.ray.tfar * dir;
    dg.Ng = Ng;
    dg.Ns = Ns;

    /* Next is material discovery.
     * Note: the full pathtracer program includes lookup of materials from a
     * scenegraph object. This could include texture lookup transformations that
     * are based on vertex-to-texture assignments. This could include a required
     * tranformation of normals for geometries that have been instanced. Below
     * is a simple option for materials.
     */

    /* default albedo is a pink color for debug */
    Vec3fa albedo = Vec3fa(0.9f, 0.7f, 0.7f);
    enum class MaterialType materialType = MaterialType::MATERIAL_MATTE;
    materialType =
        g_geomIDs[rayhit.hit.geomID].materialTable[rayhit.hit.primID];
    albedo = g_geomIDs[rayhit.hit.geomID].primColorTable[rayhit.hit.primID];
    /* An albedo as well as a material type is used */

    /* Reference epsilon value to move away from the plane, avoid artifacts */
    dg.eps =
        32.0f * 1.19209e-07f *
        max(max(abs(dg.P.x), abs(dg.P.y)), max(abs(dg.P.z), rayhit.ray.tfar));

    dg.Ng = face_forward(dir, normalize(dg.Ng));
    dg.Ns = face_forward(dir, normalize(dg.Ns));

    /* weight scaling based on material sample */
    Vec3fa c = Vec3fa(1.0f);

    /* scale down mediums that do not transmit as much light */
    const Vec3fa transmission = medium.transmission;
    if (transmission != Vec3fa(1.0f))
      c = c * Vec3fa(pow(transmission.x, rayhit.ray.tfar),
                     pow(transmission.y, rayhit.ray.tfar),
                     pow(transmission.z, rayhit.ray.tfar));

    Sample3f wi1;
    Vec2f randomMatSample(distrib(reng), distrib(reng));
    c = c * Material__sample(albedo, materialType, Lw, wo, dg, wi1, medium,
                             randomMatSample);

    /* Search for each light in the scene from our hit point. Aggregate the
     * radiance if hit point is not occluded */
    context.flags =
        RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
    for (const Light& light : g_lights) {
      Vec2f randomLightSample(distrib(reng), distrib(reng));
      Light_SampleRes ls = sample_light(light, dg, randomLightSample);
      /* If the sample probability density evaluation is 0 then no need to
       * consider this shadow ray */
      if (ls.pdf <= 0.0f) continue;

      RTCRayHit shadow;
      initRayHit(shadow, dg.P, ls.dir, dg.eps, ls.dist, time);
      rtcOccluded1(g_scene, &context, &shadow.ray);
      if (shadow.ray.tfar >= 0.0f) {
        L = L + Lw * ls.weight *
                    Material__eval(albedo, materialType, wo, dg, ls.dir);
      }
    }

    if (wi1.pdf <= 1E-4f) break;
    Lw = Lw * c / wi1.pdf;

    /* setup secondary ray */
    float sign = dot(wi1.v, dg.Ng) < 0.0f ? -1.0f : 1.0f;
    dg.P = dg.P + sign * dg.eps * dg.Ng;
    org = dg.P;
    dir = normalize(wi1.v);
    initRayHit(rayhit, org, dir, dg.eps, std::numeric_limits<float>::infinity(),
               time);
  }

  return L;
}