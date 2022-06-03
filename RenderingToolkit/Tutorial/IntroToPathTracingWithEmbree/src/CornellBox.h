#pragma once

#include <embree3/rtcore.h>

#include <vector>
#include <rkcommon/math/vec.h>

using Vec3fa = rkcommon::math::vec_t<float, 3, 1>;

#ifdef _WIN32
#define alignedMalloc(a, b) _aligned_malloc(a, b)
#define alignedFree(a) _aligned_free(a)
#else
#include <mm_malloc.h>
#define alignedMalloc(a, b) _mm_malloc(a, b)
#define alignedFree(a) _mm_free(a)
#endif

/* from tutorial_device.h */
/* vertex and triangle layout */
struct Vertex {
    float x, y, z, r;
};

struct Quad {
    int v0, v1, v2, v3;
};

struct Triangle {
    int v0, v1, v2;
};

// mesh data
static std::vector<Vertex> cornellBoxVertices = {
    // Floor
    {1.00f, -1.00f, -1.00f, 0.0f},
    {-1.00f, -1.00f, -1.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    // Ceiling
    {1.00f, 1.00f, -1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, -1.00f, 0.0f},
    // Backwall
    {1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    // RightWall
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, -1.00f, 0.0f},
    {-1.00f, 1.00f, -1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    // LeftWall
    {1.00f, -1.00f, -1.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, -1.00f, 0.0f},
    // ShortBox Top Face
    {-0.53f, -0.40f, -0.75f, 0.0f},
    {-0.70f, -0.40f, -0.17f, 0.0f},
    {-0.13f, -0.40f, -0.00f, 0.0f},
    {0.05f, -0.40f, -0.57f, 0.0f},
    // ShortBox Left Face
    {0.05f, -1.00f, -0.57f, 0.0f},
    {0.05f, -0.40f, -0.57f, 0.0f},
    {-0.13f, -0.40f, -0.00f, 0.0f},
    {-0.13f, -1.00f, -0.00f, 0.0f},
    // ShortBox Front Face
    {-0.53f, -1.00f, -0.75f, 0.0f},
    {-0.53f, -0.40f, -0.75f, 0.0f},
    {0.05f, -0.40f, -0.57f, 0.0f},
    {0.05f, -1.00f, -0.57f, 0.0f},
    // ShortBox Right Face
    {-0.70f, -1.00f, -0.17f, 0.0f},
    {-0.70f, -0.40f, -0.17f, 0.0f},
    {-0.53f, -0.40f, -0.75f, 0.0f},
    {-0.53f, -1.00f, -0.75f, 0.0f},
    // ShortBox Back Face
    {-0.13f, -1.00f, -0.00f, 0.0f},
    {-0.13f, -0.40f, -0.00f, 0.0f},
    {-0.70f, -0.40f, -0.17f, 0.0f},
    {-0.70f, -1.00f, -0.17f, 0.0f},
    // ShortBox Bottom Face
    {-0.53f, -1.00f, -0.75f, 0.0f},
    {-0.70f, -1.00f, -0.17f, 0.0f},
    {-0.13f, -1.00f, -0.00f, 0.0f},
    {0.05f, -1.00f, -0.57f, 0.0f},
    // TallBox Top Face
    {0.53f, 0.20f, -0.09f, 0.0f},
    {-0.04f, 0.20f, 0.09f, 0.0f},
    {0.14f, 0.20f, 0.67f, 0.0f},
    {0.71f, 0.20f, 0.49f, 0.0f},
    // TallBox Left Face
    {0.53f, -1.00f, -0.09f, 0.0f},
    {0.53f, 0.20f, -0.09f, 0.0f},
    {0.71f, 0.20f, 0.49f, 0.0f},
    {0.71f, -1.00f, 0.49f, 0.0f},
    // TallBox Front Face
    {0.71f, -1.00f, 0.49f, 0.0f},
    {0.71f, 0.20f, 0.49f, 0.0f},
    {0.14f, 0.20f, 0.67f, 0.0f},
    {0.14f, -1.00f, 0.67f, 0.0f},
    // TallBox Right Face
    {0.14f, -1.00f, 0.67f, 0.0f},
    {0.14f, 0.20f, 0.67f, 0.0f},
    {-0.04f, 0.20f, 0.09f, 0.0f},
    {-0.04f, -1.00f, 0.09f, 0.0f},
    // TallBox Back Face
    {-0.04f, -1.00f, 0.09f, 0.0f},
    {-0.04f, 0.20f, 0.09f, 0.0f},
    {0.53f, 0.20f, -0.09f, 0.0f},
    {0.53f, -1.00f, -0.09f, 0.0f},
    // TallBox Bottom Face
    {0.53f, -1.00f, -0.09f, 0.0f},
    {-0.04f, -1.00f, 0.09f, 0.0f},
    {0.14f, -1.00f, 0.67f, 0.0f},
    {0.71f, -1.00f, 0.49f, 0.0f} };

static std::vector<Quad> cornellBoxIndices = {
    {0, 1, 2, 3}, // Floor
    {4, 5, 6, 7}, // Ceiling
    {8, 9, 10, 11}, // Backwall
    {12, 13, 14, 15}, // RightWall
    {16, 17, 18, 19}, // LeftWall
    {20, 21, 22, 23}, // ShortBox Top Face
    {24, 25, 26, 27}, // ShortBox Left Face
    {28, 29, 30, 31}, // ShortBox Front Face
    {32, 33, 34, 35}, // ShortBox Right Face
    {36, 37, 38, 39}, // ShortBox Back Face
    {40, 41, 42, 43}, // ShortBox Bottom Face
    {44, 45, 46, 47}, // TallBox Top Face
    {48, 49, 50, 51}, // TallBox Left Face
    {52, 53, 54, 55}, // TallBox Front Face
    {56, 57, 58, 59}, // TallBox Right Face
    {60, 61, 62, 63}, // TallBox Back Face
    {64, 65, 66, 67} // TallBox Bottom Face
};

static std::vector<Vec3fa> cornellBoxColors = {
    // Floor
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // Ceiling
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // Backwall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // RightWall
    {0.140f, 0.450f, 0.091f},
    {0.140f, 0.450f, 0.091f},
    {0.140f, 0.450f, 0.091f},
    {0.140f, 0.450f, 0.091f},
    // LeftWall
    {0.630f, 0.065f, 0.05f},
    {0.630f, 0.065f, 0.05f},
    {0.630f, 0.065f, 0.05f},
    {0.630f, 0.065f, 0.05f},
    // ShortBox Top Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Left Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Front Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Right Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Back Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // ShortBox Bottom Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // TallBox Top Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // TallBox Left Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // TallBox Front Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // TallBox Right Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // TallBox Back Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // TallBox Bottom Face
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f} };

int addCornell(RTCScene scene, RTCDevice device, Vec3fa** g_cornell_face_colors, Vec3fa** g_cornell_vertex_colors)
{
    /* create a mesh for all the quads in the Cornell Box scene */
    RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD);
    *g_cornell_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * cornellBoxIndices.size(), 16);
    *g_cornell_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * cornellBoxVertices.size(), 16);
    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), cornellBoxVertices.size());
    for (auto i = 0; i < cornellBoxVertices.size(); i++) {
        vertices[i] = cornellBoxVertices[i];
        (*g_cornell_vertex_colors)[i] = cornellBoxColors[i];
    }
    
    /* set quads */
    Quad* quads = (Quad*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(Quad), cornellBoxIndices.size());

    for (auto i = 0; i < cornellBoxIndices.size(); i++) {
        quads[i] = cornellBoxIndices[i];
        (*g_cornell_face_colors)[i] = cornellBoxColors[i * 4];
    }

    rtcSetGeometryVertexAttributeCount(mesh, 1);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
        RTC_FORMAT_FLOAT3, *g_cornell_vertex_colors, 0,
        sizeof(Vec3fa), cornellBoxVertices.size());

    rtcCommitGeometry(mesh);
    unsigned int geomID = rtcAttachGeometry(scene, mesh);
    rtcReleaseGeometry(mesh);
    return geomID;

}