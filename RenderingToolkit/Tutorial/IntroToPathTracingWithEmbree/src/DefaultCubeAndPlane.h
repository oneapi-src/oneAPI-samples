#pragma once
#ifndef FILE_DEFAULTCUBEANDPLANE_SEEN
#define FILE_DEFAULTCUBEANEPLANE_SEEN

#include "definitions.h"

Vec3fa* g_cube_face_colors = nullptr;
Vec3fa* g_cube_vertex_colors = nullptr;
Vec3fa* g_ground_face_colors = nullptr;
Vec3fa* g_ground_vertex_colors = nullptr;

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
    return geomID;
}

/* adds a ground plane to the scene */
unsigned int addGroundPlane(RTCScene _scene, RTCDevice _device) {
    /* create a triangulated plane with 2 triangles and 4 vertices */
    RTCGeometry mesh = rtcNewGeometry(_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    /* create face and vertex color arrays */
    g_ground_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 2, 16);
    g_ground_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 4, 16);

    /* set vertices */
    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 4);
    g_ground_vertex_colors[0] = Vec3fa(1, 0, 0);
    vertices[0].x = -10;
    vertices[0].y = -1;
    vertices[0].z = -10;
    g_ground_vertex_colors[1] = Vec3fa(1, 0, 1);
    vertices[1].x = -10;
    vertices[1].y = -1;
    vertices[1].z = +10;
    g_ground_vertex_colors[2] = Vec3fa(1, 1, 0);
    vertices[2].x = +10;
    vertices[2].y = -1;
    vertices[2].z = -10;
    g_ground_vertex_colors[3] = Vec3fa(1, 1, 1);
    vertices[3].x = +10;
    vertices[3].y = -1;
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

void cubeAndPlaneCameraLightSetup(AffineSpace3fa& _camera, std::vector<Light>& _lights, unsigned int _width, unsigned int _height) {
    Vec3fa defaultLightDirection = normalize(Vec3fa(-1.0f, -1.0f, -1.0f));
    Vec3fa defaultLightIntensity = { 1.0f, 1.0f, 1.0f };

    _camera = positionCamera(Vec3fa(1.5f, 1.5, -1.5f), Vec3fa(0, 0, 0),
        Vec3fa(0, 1, 0), 90.0f, _width, _height);
    _lights.resize(1);
    _lights[0].dir = defaultLightDirection;
    _lights[0].intensity = defaultLightIntensity;
    _lights[0].type = LightType::INFINITE_DIRECTIONAL_LIGHT;

}

#endif /* !FILE_DEFAULTCUBEANDPLANE_SEEN */