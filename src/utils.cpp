#include "utils.h"


SGP::Vec SGP::sampleOnMesh(const polyscope::SurfaceMesh *S, const ScalarGrid3 &G, const AffineMap3 &map) {
    int n = S->vertexPositions.data.size();
    Vec samples = Vec::Zero(n);
    bool square = G.getSizes().maxCoeff() == G.getSizes().minCoeff();
    for (auto i : range(n)) {
        auto p = S->vertexPositions.data[i];
        vec x(p.x, p.y, p.z);
        if (square)
            samples(i) = Grid3D::lerp(G, x);
        else
            samples(i) = Grid3D::lerpNonSquare(G, x, map);
    }
    return samples;
}

SGP::vec SGP::MostAlignedVertex(polyscope::SurfaceMesh *M, const vec &d) {
    vec z;
    scalar hz = -1000;
    for (auto v : range(M->vertexPositions.size())) {
        auto q = M->vertexPositions.getValue(v);
        vec p = vec(q.x,q.y,q.z);
        if (p.dot(d) > hz) {
            z = p;
            hz = p.dot(d);
        }
    }
    return z;
}

int SGP::GetClosestPoint(const Points<dim> &P, const Vector<dim> &x){
    int c = 0;
    scalar min_dist = 1e20;
    for (auto i : range(P.cols())){
        scalar dist = (P.col(i)-x).squaredNorm();
        if (dist < min_dist){
            min_dist = dist;
            c = i;
        }
    }
    return c;
}

SGP::vecs SGP::sampleOnMesh(const polyscope::SurfaceMesh *S, const VectorGrid3 &G) {
    int n = S->vertexPositions.data.size();
    vecs samples = vecs(n);
    for (auto i : range(n)) {
        auto p = S->vertexPositions.data[i];
        vec x(p.x, p.y, p.z);
        samples[i] = Grid3D::lerp(G, x);
    }
    return samples;
}

SGP::vec SGP::ClosestVertex(polyscope::SurfaceMesh *M, const vec &x) {
    vec z;
    scalar hz = 1000;
    for (auto v : range(M->vertexPositions.size())) {
        auto q = M->vertexPositions.getValue(v);

        vec p = vec(q.x,q.y,q.z);
        if ((p-x).norm() < hz) {
            z = p;
            hz = (p-x).norm();
        }
    }
    return z;
}

SGP::Vec SGP::sampleOnCloud(const Points<3> &S, const ScalarGrid3 &G, const AffineMap3 &map)
{
    int n = S.cols();
    Vec samples = Vec::Zero(n);
    bool square = G.getSizes().maxCoeff() == G.getSizes().minCoeff();
    for (auto i : range(n)) {
        vec x = S.col(i);
        if (square)
            samples(i) = Grid3D::lerp(G, x);
        else
            samples(i) = Grid3D::lerpNonSquare(G, x, map);
    }
    return samples;

}

void SGP::write_vti(const std::string &filename, const Vec &data, int nx, int ny, int nz, double dx, double dy, double dz)
{
    std::ofstream file(filename);
    file << R"(<?xml version="1.0"?>)" << "\n";
    file << R"(<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">)" << "\n";
    file << "<ImageData WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1
         << "\" Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";
    file << "<Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1 << "\">\n";
    file << "<PointData Scalars=\"scalars\">\n";
    file << "<DataArray type=\"Float64\" Name=\"scalars\" format=\"ascii\">\n";

    for (size_t i = 0; i < data.size(); ++i)
        file << data[i] << " ";
    file << "\n</DataArray>\n";
    file << "</PointData>\n";
    file << "<CellData/>\n";
    file << "</Piece>\n";
    file << "</ImageData>\n";
    file << "</VTKFile>\n";
}

void SGP::export_mesh_to_obj(std::string filename, polyscope::SurfaceMesh *mesh) {
    std::ofstream file(filename);

    // Write vertices
    for (auto i : range(mesh->nVertices())){
        auto v = mesh->vertexPositions.data.at(i);
        file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    for (auto i : range(mesh->nFaces())){
        file << "f ";
        file << mesh->faceIndsEntries[3*i] + 1 << " " << mesh->faceIndsEntries[3*i+1]+1 << " " << mesh->faceIndsEntries[3*i+2]+1;
        file << "\n";
    }
}

geometrycentral::surface::Vertex SGP::MostAlignedVertex(const Mesh &M, const vec &d) {
    Vertex z;
    scalar hz = -1000;
    for (auto v : M.topology->vertices()) {
        auto q = M.geometry->vertexPositions[v];
        vec p = vec(q.x,q.y,q.z);
        if (p.dot(d) > hz) {
            z = v;
            hz = p.dot(d);
        }
    }
    return z;
}

int SGP::MostAlignedPoint(const Points<dim> &M, const Vector<dim> &d)
{
    int c = 0;
    scalar hz = -1000;
    for (auto i : range(M.cols())) {
        if (M.col(i).dot(d) > hz) {
            c = i;
            hz = M.col(i).dot(d);
        }
    }
    return c;

}

geometrycentral::surface::Vertex SGP::ClosestVertex(const Mesh &M, const vec &d)
{
    Vertex z;
    scalar hz = 1000;
    for (auto v : M.topology->vertices()) {
        auto q = M.geometry->vertexPositions[v];
        vec p = vec(q.x,q.y,q.z);
        if ((p-d).norm() < hz) {
            z = v;
            hz = (p-d).norm();
        }
    }
    return z;
}
