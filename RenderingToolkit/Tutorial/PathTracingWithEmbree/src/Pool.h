#pragma once
#ifndef FILE_POOL_SEEN
#define FILE_POOL_SEEN

#include <embree3/rtcore.h>

#include <vector>

#include "definitions.h"
#include "Materials.h"
#include "Lights.h"

/* Added for pathtracer */
Vec3fa* g_pool_face_colors = nullptr;
Vec3fa* g_pool_vertex_colors = nullptr;
Vec3fa* g_water_face_colors = nullptr;
Vec3fa* g_water_vertex_colors = nullptr;

// mesh data
static std::vector<Vertex> poolBoxVertices = {
    // South Wall
    {1.00f, -1.00f, -5.00f, 0.0f},
    {-1.00f, -1.00f, -5.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    // North Wall
    {1.00f, 1.00f, -5.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, -5.00f, 0.0f},

    // Foundation
    {1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    // East Wall
    {-1.00f, -1.00f, 1.00f, 0.0f},
    {-1.00f, -1.00f, -5.00f, 0.0f},
    {-1.00f, 1.00f, -5.00f, 0.0f},
    {-1.00f, 1.00f, 1.00f, 0.0f},
    // West Wall
    {1.00f, -1.00f, -5.00f, 0.0f},
    {1.00f, -1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, 1.00f, 0.0f},
    {1.00f, 1.00f, -5.00f, 0.0f},

    // Unseen wall
    {1.00f, -1.00f, -5.00f, 0.0f},
    {-1.00f, -1.00f, -5.00f, 0.0f},
    {-1.00f, 1.00f, -5.00f, 0.0f},
    {1.00f, 1.00f, -5.00f, 0.0f}

};

static std::vector<Quad> poolBoxIndices = {
    {0, 1, 2, 3},      // Floor
    {4, 5, 6, 7},      // Ceiling
    {8, 9, 10, 11},    // Backwall
    {12, 13, 14, 15},  // RightWall
    {16, 17, 18, 19},  // LeftWall
    {20, 21, 22, 23}   // Unseen wall
};

static std::vector<Vec3fa> poolBoxColors = {
    // South Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // North Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // Foundation "Egyptian Blue"
    {0.078f, 0.204f, 0.643f},
    {0.078f, 0.204f, 0.643f},
    {0.078f, 0.204f, 0.643f},
    {0.078f, 0.204f, 0.643f},
    // East Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    // West Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},    
    // West Wall
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
    {0.725f, 0.710f, 0.68f},
 
};

static std::vector<enum class MaterialType> poolBoxMats = {
    // Floor
    MaterialType::MATERIAL_MATTE,
    /* Swap in thr below material to make the ceiling a matte material*/
    // Ceiling
    MaterialType::MATERIAL_MATTE,
    // Backwall
    MaterialType::MATERIAL_MATTE,
    // RightWall
    MaterialType::MATERIAL_MATTE,
    // LeftWall
    MaterialType::MATERIAL_MATTE,
    // Unseen Wall
    MaterialType::MATERIAL_MATTE

};

static std::vector<enum class MaterialType> waterMats;


int addPool(RTCScene scene, RTCDevice device) {
    /* create a mesh for all the quads in the pool Box scene */
    RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_QUAD);
    g_pool_face_colors =
        (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * poolBoxIndices.size(), 16);
    g_pool_vertex_colors =
        (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * poolBoxVertices.size(), 16);
    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
        poolBoxVertices.size());
    for (auto i = 0; i < poolBoxVertices.size(); i++) {
        vertices[i] = poolBoxVertices[i];
        g_pool_vertex_colors[i] = poolBoxColors[i];
    }

    /* set quads */
    Quad* quads = (Quad*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0,
        RTC_FORMAT_UINT4, sizeof(Quad),
        poolBoxIndices.size());

    for (auto i = 0; i < poolBoxIndices.size(); i++) {
        quads[i] = poolBoxIndices[i];
        g_pool_face_colors[i] = poolBoxColors[i * 4];
    }

    rtcSetGeometryVertexAttributeCount(mesh, 1);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
        RTC_FORMAT_FLOAT3, *g_pool_vertex_colors, 0,
        sizeof(Vec3fa), poolBoxVertices.size());

    rtcCommitGeometry(mesh);
    unsigned int geomID = rtcAttachGeometry(scene, mesh);
    rtcReleaseGeometry(mesh);

    MatAndPrimColorTable mpTable;
    mpTable.materialTable = poolBoxMats;
    mpTable.primColorTable = g_pool_face_colors;
    g_geomIDs.insert(std::make_pair(geomID, mpTable));

    return geomID;
}

void cleanpool() {
    if (g_pool_face_colors) alignedFree(g_pool_face_colors);
    g_pool_face_colors = nullptr;
    if (g_pool_vertex_colors) alignedFree(g_pool_vertex_colors);
    g_pool_vertex_colors = nullptr;
}

unsigned int addWater(RTCScene _scene, RTCDevice _device) {
    /* create a triangulated water with 12 triangles and 8 vertices */
    RTCGeometry mesh = rtcNewGeometry(_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    size_t latticeWidth = 200;
    int numTriangles = 2 * (latticeWidth - 1) * (latticeWidth - 1);
    /* create face and vertex color arrays */
    g_water_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * numTriangles, 16);
    g_water_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * latticeWidth * latticeWidth, 16);


    /* set vertices and vertex colors */
    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), latticeWidth * latticeWidth);

    //Bubbles color
    Vec3fa waterColor = Vec3fa(0.906f, 0.9961f, 1.0f);

    //Top of the water will be z = FarPlaneZ - waterdepth... modulated up or down by some sin waves
    float waterDepth = 1.4f;
    float horizScale = 0.03f;
    //float vertScale = 0.03f;
    int horizPeriods = 4;
    //int vertPeriods = 6;

    for (int j = 0; j < latticeWidth; j++) {
        for (int i = 0; i < latticeWidth; i++)  {

            g_water_vertex_colors[j * latticeWidth + i] = waterColor;
        //Assigning vertices from bottom right to top left
        vertices[j * latticeWidth + i].x = 1.0f - (float(i) / float(latticeWidth - 1)) * 2.0f;
        vertices[j * latticeWidth + i].y = 1.0f - (float(j) / float(latticeWidth - 1)) * 2.0f;
        vertices[j * latticeWidth + i].z = 1.0f - waterDepth
            + horizScale * waterDepth * sinf(horizPeriods * (float(i) / (latticeWidth - 1)) * 2.f * float(M_PI));
        //+  vertScale * waterDepth * sinf(vertPeriods*(float(j) / float(latticeWidth - 1)) * 2.f * float(M_PI));
    }
}


    /* set triangles and face colors */

    int tri = 0;
    Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
        mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), numTriangles);

    for (int j = 0; j < latticeWidth - 1; j++) {
        for (int i = 0; i < latticeWidth - 1; i++) {

            g_water_face_colors[tri] = waterColor;
            waterMats.push_back(MaterialType::MATERIAL_WATER);
            triangles[tri].v0 = j * latticeWidth + i;
            triangles[tri].v1 = j * latticeWidth + (i + 1);
            triangles[tri].v2 = (j + 1) * latticeWidth + i;
            tri++;

            g_water_face_colors[tri] = waterColor;
            waterMats.push_back(MaterialType::MATERIAL_WATER);
            triangles[tri].v0 = j * latticeWidth + (i + 1);
            triangles[tri].v1 = (j + 1) * latticeWidth + i;
            triangles[tri].v2 = (j + 1) * latticeWidth + (i + 1);
            tri++;
        }
    }


    rtcSetGeometryVertexAttributeCount(mesh, 1);
    rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
        RTC_FORMAT_FLOAT3, g_water_vertex_colors, 0,
        sizeof(Vec3fa), latticeWidth*latticeWidth);

    rtcCommitGeometry(mesh);
    unsigned int geomID = rtcAttachGeometry(_scene, mesh);
    rtcReleaseGeometry(mesh);

    MatAndPrimColorTable mpTable;
    mpTable.materialTable = waterMats;
    mpTable.primColorTable = g_water_face_colors;
    g_geomIDs.insert(std::make_pair(geomID, mpTable));

    return geomID;
}



void poolCameraLightSetup(AffineSpace3fa& camera, std::vector<Light>& lights,
    unsigned int width, unsigned int height) {

     /* A camera position that connects the field of vision angle of the camera to
      * the bounds of the pool box */
    float fov = 30.0f;
    float fovrad = fov * M_PI / 180.0f;
    float half_fovrad = fovrad * 0.5f;
    // Careful to not move the camera outside the box! :-D
    camera = positionCamera(Vec3fa(0.0, 0.0, -1.0f - 1.f / tanf(half_fovrad)),
        Vec3fa(0, 0, 0), Vec3fa(0, 1, 0), fov, width, height);


    Light pointLight;
    /* The magnitude of the light can be tricky. Lights such as the point light
     * fall off at the inverse square of the distance. When designing a sandbox
     * renderer, you may need to scale your light up or down to see your scene. */
    pointLight.intensity = 5.f * Vec3fa(0.9922, 0.9843, 0.8275);

     /* A somewhat central position for the point light within the box. This is
      * similar to the position for the interactive pathtracer program shipped with
      * Intel Embree */
    //pointLight.pos =
    //    Vec3fa(2.f * 213.0f / 556.0f - 1.f, 2.f * 300.f / 558.8f - 1.f,
    //        2.f * 227.f / 559.2f - 1.f);
    pointLight.pos = Vec3fa(0.0f, 0.0f, -2.0f);
    pointLight.type = LightType::POINT_LIGHT;
    lights.push_back(pointLight);
}

void cleanPool() {
    if (g_pool_face_colors) alignedFree(g_pool_face_colors);
    g_pool_face_colors = nullptr;
    if (g_pool_vertex_colors) alignedFree(g_pool_vertex_colors);
    g_pool_vertex_colors = nullptr;

}

void cleanWater() {
    if (g_water_face_colors) alignedFree(g_water_face_colors);
    g_water_face_colors = nullptr;
    if (g_water_vertex_colors) alignedFree(g_water_vertex_colors);
    g_water_vertex_colors = nullptr;

}

#endif /* !FILE_POOL_SEEN */