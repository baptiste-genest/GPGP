#ifndef MESHSAMPLING_H
#define MESHSAMPLING_H

#include "StochasticGeometryProcessing.h"
#include "Mesh.h"
#include <geometrycentral/surface/surface_point.h>
// #include <polyscope/point_cloud.h>

#include "PointCloud.h"

namespace SGP {


//using PointsOnMesh = std::vector<PointOnMesh>;
using SurfacePoint = geometrycentral::surface::SurfacePoint;
using SurfacePoints = std::vector<SurfacePoint>;

SurfacePoints sampleMesh(const Mesh &M, int sampleNum, const scalars &face_weights);

inline OrientedWeightedPointCloud<3> toOrientedPointCloud(const Mesh &M, const SurfacePoints &X) {
    M.geometry->requireVertexNormals();
    Points<3> P(3,X.size());
    Points<3> normals(3,X.size());

    for (auto i : range(X.size())){
        P.col(i) = toVec(X[i].interpolate(M.geometry->vertexPositions));
        normals.col(i) = toVec(X[i].interpolate(M.geometry->vertexNormals)).normalized();
    }
    return {P,normals};
}


inline OrientedWeightedPointCloud<3> meshToOPC(const Mesh& M) {
    // each position is a face barycenter
    // normal is face normal
    OrientedWeightedPointCloud<3> pc;
    // int V = M.topology->nVertices();
    int F = M.topology->nFaces();
    int N = F;//+V;
    pc.positions.resize(3, N);
    pc.normals.resize(3, N);
    pc.weights.resize(N);
    M.geometry->requireVertexNormals();
    auto DA = M.dualAreas();
    auto FA = M.faceAreas();
    for (auto f : M.topology->faces()) {
        pc.positions.col(f.getIndex()) = M.faceBarycenter(f);
        pc.normals.col(f.getIndex()) = toVec(M.geometry->faceNormal(f));
        pc.weights[f.getIndex()] = FA[f.getIndex()];
    }
    /*
    for (auto v : M.topology->vertices()) {
        pc.positions.col(v.getIndex()) = toVec(M.geometry->vertexPositions[v]);
        pc.normals.col(v.getIndex()) = toVec(M.geometry->vertexNormals[v]);
        pc.weights[v.getIndex()] = DA[v.getIndex()]*3/4;
    }
*/
    return pc;
}



// polyscope::PointCloud* display(std::string label,const Mesh& M,const SurfacePoints& X);

}

#endif // MESHSAMPLING_H
