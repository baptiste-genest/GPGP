#ifndef UTILS_H
#define UTILS_H

#include "StochasticGeometryProcessing.h"
#include "Mesh.h"
#include "polyscope/surface_mesh.h"
#include "Grid.h"

namespace SGP {


Vec sampleOnMesh(const polyscope::SurfaceMesh* S,const ScalarGrid3& G,const AffineMap3& map = AffineMap3::Identity());

Vec sampleOnCloud(const Points<3>& S,const ScalarGrid3& G,const AffineMap3& map = AffineMap3::Identity());

vec MostAlignedVertex(polyscope::SurfaceMesh* M,const vec& d);

int GetClosestPoint(const Points<dim>& P,const Vector<dim>& x);
int MostAlignedPoint(const Points<dim>& M, const Vector<dim> &d);

vecs sampleOnMesh(const polyscope::SurfaceMesh* S,const VectorGrid3& G);

vec ClosestVertex(polyscope::SurfaceMesh* M,const vec& x);

void write_vti(const std::string &filename,
               const Vec &data,
               int nx, int ny, int nz,
               double dx, double dy, double dz);

void export_mesh_to_obj(std::string filename,polyscope::SurfaceMesh* mesh);

geometrycentral::surface::Vertex MostAlignedVertex(const Mesh& M, const vec &d);
geometrycentral::surface::Vertex ClosestVertex(const Mesh& M, const vec &d);


};


#endif // UTILS_H
