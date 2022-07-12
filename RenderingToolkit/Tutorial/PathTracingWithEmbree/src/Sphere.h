#pragma once
#ifndef FILE_SPHERE_SEEN
#define FILE_SPHERE_SEEN

#include "definitions.h"

#include <embree3/rtcore.h>
#include <vector>

static std::vector<enum class MaterialType> sphereMats = {
    //Just one material for our sphere primitive (Defined as singular Vec4 point for embree)
    MaterialType::MATERIAL_GLASS
};

Vec3fa g_sphere_face_colors = { 1.f, 1.f, 1.f };

int addSphere(RTCScene scene, RTCDevice device, const Vec3fa& pos, float radius)
{
    RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(Vertex), 1);

    Vertex p = { pos.x, pos.y, pos.z, radius };
    vertices[0] = p;

    rtcCommitGeometry(mesh);
    unsigned int geomID = rtcAttachGeometry(scene, mesh);
    rtcReleaseGeometry(mesh);

    MatAndPrimColorTable mpTable;
    mpTable.materialTable = sphereMats;
    mpTable.primColorTable = &g_sphere_face_colors;
    g_geomIDs.insert(std::make_pair(geomID, mpTable));

    return geomID;
}

/* Only a place holder. Nothing is here because we do not attach any attributes for the sphere.  */
inline void cleanSphere()
{

}

#endif /* FILE_SPHERE_SEEN */