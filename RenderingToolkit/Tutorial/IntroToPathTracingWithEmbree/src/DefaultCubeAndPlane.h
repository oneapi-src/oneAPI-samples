#pragma once
#ifndef FILE_DEFAULTCUBEANDPLANE_SEEN
#define FILE_DEFAULTCUBEANEPLANE_SEEN

#include "definitions.h"

Vec3fa* g_cube_face_colors = nullptr;
Vec3fa* g_cube_vertex_colors = nullptr;
Vec3fa* g_ground_face_colors = nullptr;
Vec3fa* g_ground_vertex_colors = nullptr;

static std::vector<enum class MaterialType> cubeMats = {
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE
};

static std::vector<enum class MaterialType> groundMats = {
    MaterialType::MATERIAL_MATTE,
    MaterialType::MATERIAL_MATTE
};


/* adds a cube to the scene */
unsigned int addCube(RTCScene _scene, RTCDevice _device) {
    /* create a triangulated cube with 12 triangles and 8 vertices */
    RTCGeometry mesh = rtcNewGeometry(_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    /* create face and vertex color arrays */
    g_cube_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 12, 16);
    g_cube_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 8, 16);

    /* set vertices and vertex colors */
    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 8);
    g_cube_vertex_colors[0] = Vec3fa(0, 0, 0);
    vertices[0].x = -1;
    vertices[0].y = -1;
    vertices[0].z = -1;
    g_cube_vertex_colors[1] = Vec3fa(0, 0, 1);
    vertices[1].x = -1;
    vertices[1].y = -1;
    vertices[1].z = +1;
    g_cube_vertex_colors[2] = Vec3fa(0, 1, 0);
    vertices[2].x = -1;
    vertices[2].y = +1;
    vertices[2].z = -1;
    g_cube_vertex_colors[3] = Vec3fa(0, 1, 1);
    vertices[3].x = -1;
    vertices[3].y = +1;
    vertices[3].z = +1;
    g_cube_vertex_colors[4] = Vec3fa(1, 0, 0);
    vertices[4].x = +1;
    vertices[4].y = -1;
    vertices[4].z = -1;
    g_cube_vertex_colors[5] = Vec3fa(1, 0, 1);
    vertices[5].x = +1;
    vertices[5].y = -1;
    vertices[5].z = +1;
    g_cube_vertex_colors[6] = Vec3fa(1, 1, 0);
    vertices[6].x = +1;
    vertices[6].y = +1;
    vertices[6].z = -1;
    g_cube_vertex_colors[7] = Vec3fa(1, 1, 1);
    vertices[7].x = +1;
    vertices[7].y = +1;
    vertices[7].z = +1;

    /* set triangles and face colors */
    int tri = 0;
    Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 12);

    // left side
    g_cube_face_colors[tri] = Vec3fa(1, 0, 0);
    triangles[tri].v0 = 0;
    triangles[tri].v1 = 1;
    triangles[tri].v2 = 2;
    tri++;
    g_cube_face_colors[tri] = Vec3fa(1, 0, 0);
    triangles[tri].v0 = 1;
    triangles[tri].v1 = 3;
    triangles[tri].v2 = 2;
    tri++;

    // right side
    g_cube_face_colors[tri] = Vec3fa(0, 1, 0);
    triangles[tri].v0 = 4;
    triangles[tri].v1 = 6;
    triangles[tri].v2 = 5;
    tri++;
    g_cube_face_colors[tri] = Vec3fa(0, 1, 0);
    triangles[tri].v0 = 5;
    triangles[tri].v1 = 6;
    triangles[tri].v2 = 7;
    tri++;

    // bottom side
    g_cube_face_colors[tri] = Vec3fa(1.0f);
    triangles[tri].v0 = 0;
    triangles[tri].v1 = 4;
    triangles[tri].v2 = 1;
    tri++;
    g_cube_face_colors[tri] = Vec3fa(1.0f);
    triangles[tri].v0 = 1;
    triangles[tri].v1 = 4;
    triangles[tri].v2 = 5;
    tri++;

    // top side
    g_cube_face_colors[tri] = Vec3fa(1.0f);
    triangles[tri].v0 = 2;
    triangles[tri].v1 = 3;
    triangles[tri].v2 = 6;
    tri++;
    g_cube_face_colors[tri] = Vec3fa(1.0f);
    triangles[tri].v0 = 3;
    triangles[tri].v1 = 7;
    triangles[tri].v2 = 6;
    tri++;

    // front side
    g_cube_face_colors[tri] = Vec3fa(0, 0, 1);
    triangles[tri].v0 = 0;
    triangles[tri].v1 = 2;
    triangles[tri].v2 = 4;
    tri++;
    g_cube_face_colors[tri] = Vec3fa(0, 0, 1);
    triangles[tri].v0 = 2;
    triangles[tri].v1 = 6;
    triangles[tri].v2 = 4;
    tri++;

    // back side
    g_cube_face_colors[tri] = Vec3fa(1, 1, 0);
    triangles[tri].v0 = 1;
    triangles[tri].v1 = 5;
    triangles[tri].v2 = 3;
    tri++;
    g_cube_face_colors[tri] = Vec3fa(1, 1, 0);
    triangles[tri].v0 = 3;
    triangles[tri].v1 = 5;
    triangles[tri].v2 = 7;
    tri++;

    rtcSetGeometryVertexAttributeCount(mesh, 1);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
        RTC_FORMAT_FLOAT3, g_cube_vertex_colors, 0,
        sizeof(Vec3fa), 8);

    rtcCommitGeometry(mesh);
    unsigned int geomID = rtcAttachGeometry(_scene, mesh);
    rtcReleaseGeometry(mesh);

    MatAndPrimColorTable mpTable;
    mpTable.materialTable = cubeMats;
    mpTable.primColorTable = g_cube_face_colors;
    g_geomIDs.insert(std::make_pair(geomID, mpTable));

    return geomID;
}

/* adds a ground plane to the scene */
unsigned int addGroundPlane(RTCScene _scene, RTCDevice _device) {
    /* create a triangulated plane with 2 triangles and 4 vertices */
    RTCGeometry mesh = rtcNewGeometry(_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    /* create face and vertex color arrays */
    g_ground_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 2, 16);
    g_ground_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 4, 16);

    /* Moving the plane up to the bottom of the cube shows more global illumination color bleed 
    Try y = -1 to see it!
    */
    /* The color of the ground plane is changed to white to see global illumination effects */
    /* set vertices */
    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 4);
    g_ground_vertex_colors[0] = Vec3fa(1, 0, 0);
    vertices[0].x = -10;
    vertices[0].y = -2;
    vertices[0].z = -10;
    g_ground_vertex_colors[1] = Vec3fa(1, 0, 1);
    vertices[1].x = -10;
    vertices[1].y = -2;
    vertices[1].z = +10;
    g_ground_vertex_colors[2] = Vec3fa(1, 1, 0);
    vertices[2].x = +10;
    vertices[2].y = -2;
    vertices[2].z = -10;
    g_ground_vertex_colors[3] = Vec3fa(1, 1, 1);
    vertices[3].x = +10;
    vertices[3].y = -2;
    vertices[3].z = +10;

    /* set triangles */
    Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 2);

    g_ground_face_colors[0] = Vec3fa(1, 1, 1);
    triangles[0].v0 = 0;
    triangles[0].v1 = 1;
    triangles[0].v2 = 2;
    g_ground_face_colors[1] = Vec3fa(1, 1, 1);
    triangles[1].v0 = 1;
    triangles[1].v1 = 3;
    triangles[1].v2 = 2;

    rtcSetGeometryVertexAttributeCount(mesh, 1);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
        RTC_FORMAT_FLOAT3, g_ground_vertex_colors, 0,
        sizeof(Vec3fa), 4);

    rtcCommitGeometry(mesh);
    unsigned int geomID = rtcAttachGeometry(_scene, mesh);
    rtcReleaseGeometry(mesh);

    MatAndPrimColorTable mpTable;
    mpTable.materialTable = groundMats;
    mpTable.primColorTable = g_ground_face_colors;
    g_geomIDs.insert(std::make_pair(geomID, mpTable));

    return geomID;
}

void cleanCubeAndPlane() {
    if (g_cube_face_colors) alignedFree(g_cube_face_colors);
    g_cube_face_colors = nullptr;
    if (g_cube_vertex_colors) alignedFree(g_cube_vertex_colors);
    g_cube_vertex_colors = nullptr;
    if (g_ground_face_colors) alignedFree(g_ground_face_colors);
    g_ground_face_colors = nullptr;
    if (g_ground_vertex_colors) alignedFree(g_ground_vertex_colors);
    g_ground_vertex_colors = nullptr;

}

void cubeAndPlaneCameraLightSetup(AffineSpace3fa& camera, std::vector<Light>& lights, unsigned int width, unsigned int height) {
    Vec3fa defaultLightDirection = normalize(Vec3fa(-1.0f, -1.0f, -1.0f));
    Vec3fa defaultLightIntensity = { 1.0f, 1.0f, 1.0f };

    camera = positionCamera(Vec3fa(1.5f, 1.5f, -1.5f), Vec3fa(0, 0, 0),
        Vec3fa(0, 1, 0), 90.0f, width, height);

    /* Our light from the triangle geometry sample */
    /*
    Light infDirectionalLight;
    infDirectionalLight.dir = defaultLightDirection;
    infDirectionalLight.intensity = defaultLightIntensity;
    infDirectionalLight.type = LightType::INFINITE_DIRECTIONAL_LIGHT;
    lights.push_back(infDirectionalLight);
    */
    
    Light pointLight;
    /* Note that the magnitude of the light can be tricky. Lights such as the point light fall off at the inverse square of the distance. When designing a sandbox renderer, you may need to scale your light up or down to see your scene. */
    pointLight.intensity = 500.f * Vec3fa(1.f, 1.f, 1.f);

    /* The point light that mimicks the direction of the directional light */
    pointLight.pos = Vec3fa(10.0f, 10.0f, 10.0f);
    pointLight.type = LightType::POINT_LIGHT;
    lights.push_back(pointLight);
}

#endif /* !FILE_DEFAULTCUBEANDPLANE_SEEN */