#include "Grid.h"

SGP::PointGrid3 SGP::Grid3D::getGrid(int n) {
    PointGrid3 G(n,n,n);
    scalar dx = 1.0 / (n-1);
    StopWatch profiler;
#pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                G(i,j,k) = (vec(i*dx,j*dx,k*dx) - vec(0.5,0.5,0.5))*2;
    return G;
}

SGP::ScalarGrid3 SGP::Grid3D::ToGrid(const Vec &X, int N) {
    ScalarGrid3 G(N,N,N);
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<N;k++){
                SliceIndex3 index = {i,j,k};
                G(index) = X(G.getIndex(index));
            }
    return G;
}

SGP::vec SGP::Grid3D::lerp(const VectorGrid3 &G, const vec &x) {
    // trilinear interpolation, assumes x in [-1,1]^3
    int N = G.getSizes()[0];
    vec xi = (x + vec(1,1,1)) / 2 * (N-1); // convert to [0,N-1]^3
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    int k0 = std::floor(xi(2));
    int i1 = std::ceil(xi(0));
    int j1 = std::ceil(xi(1));
    int k1 = std::ceil(xi(2));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    scalar zd = xi(2) - k0;
    vec c000 = G(i0,j0,k0);
    vec c001 = G(i0,j0,k1);
    vec c010 = G(i0,j1,k0);
    vec c011 = G(i0,j1,k1);
    vec c100 = G(i1,j0,k0);
    vec c101 = G(i1,j0,k1);
    vec c110 = G(i1,j1,k0);
    vec c111 = G(i1,j1,k1);
    vec c00 = c000 * (1 - xd) + c100 * xd;
    vec c01 = c001 * (1 - xd) + c101 * xd;
    vec c10 = c010 * (1 - xd) + c110 * xd;
    vec c11 = c011 * (1 - xd) + c111 * xd;
    vec c0 = c00 * (1 - yd) + c10 * yd;
    vec c1 = c01 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
}

SGP::scalar SGP::Grid3D::lerp(const ScalarGrid3 &G, const vec &x) {
    // trilinear interpolation, assumes x in [-1,1]^3
    // int N = G.getSizes()[0];
    auto S = G.getSizes();
    int Nmax = S.maxCoeff();
    //TODO : FIX for non uniform res
    vec xi = ((x + vec(1,1,1)) / 2)*(Nmax-1); // convert to [0,N-1]^3
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    int k0 = std::floor(xi(2));
    int i1 = std::ceil(xi(0));
    int j1 = std::ceil(xi(1));
    int k1 = std::ceil(xi(2));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    scalar zd = xi(2) - k0;
    scalar c000 = G(i0,j0,k0);
    scalar c001 = G(i0,j0,k1);
    scalar c010 = G(i0,j1,k0);
    scalar c011 = G(i0,j1,k1);
    scalar c100 = G(i1,j0,k0);
    scalar c101 = G(i1,j0,k1);
    scalar c110 = G(i1,j1,k0);
    scalar c111 = G(i1,j1,k1);
    scalar c00 = c000 * (1 - xd) + c100 * xd;
    scalar c01 = c001 * (1 - xd) + c101 * xd;
    scalar c10 = c010 * (1 - xd) + c110 * xd;
    scalar c11 = c011 * (1 - xd) + c111 * xd;
    scalar c0 = c00 * (1 - yd) + c10 * yd;
    scalar c1 = c01 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
}


SGP::ScalarGrid3 SGP::Grid3D::Splat(const PointGrid3 &G, const vec &x)
{
    int N = G.getSizes()[0];
    vec xi = (x + vec(1,1,1)) / 2 * (N-1); // convert to [0,N-1]^3
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    int k0 = std::floor(xi(2));
    int i1 = std::ceil(xi(0));
    int j1 = std::ceil(xi(1));
    int k1 = std::ceil(xi(2));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    scalar zd = xi(2) - k0;

    ScalarGrid3 S = ScalarGrid3(G.getSizes());

    S(i0,j0,k0) += (1 - xd) * (1 - yd) * (1 - zd);
    S(i0,j0,k1) += (1 - xd) * (1 - yd) * zd;
    S(i0,j1,k0) += (1 - xd) * yd * (1 - zd);
    S(i0,j1,k1) += (1 - xd) * yd * zd;
    S(i1,j0,k0) += xd * (1 - yd) * (1 - zd);
    S(i1,j0,k1) += xd * (1 - yd) * zd;
    S(i1,j1,k0) += xd * yd * (1 - zd);
    S(i1,j1,k1) += xd * yd * zd;

    return S;
}

SGP::scalar SGP::Grid3D::lerpNonSquare(const ScalarGrid3 &G, const vec &x, const AffineMap3 &map)
{
    AffineMap3 inv = map.inverse();
    vec xi = inv * x;
    int i0 = std::floor(xi(0));
    int j0 = std::floor(xi(1));
    int k0 = std::floor(xi(2));
    int i1 = std::ceil(xi(0));
    int j1 = std::ceil(xi(1));
    int k1 = std::ceil(xi(2));
    scalar xd = xi(0) - i0;
    scalar yd = xi(1) - j0;
    scalar zd = xi(2) - k0;
    scalar c000 = G(i0,j0,k0);
    scalar c001 = G(i0,j0,k1);
    scalar c010 = G(i0,j1,k0);
    scalar c011 = G(i0,j1,k1);
    scalar c100 = G(i1,j0,k0);
    scalar c101 = G(i1,j0,k1);
    scalar c110 = G(i1,j1,k0);
    scalar c111 = G(i1,j1,k1);
    scalar c00 = c000 * (1 - xd) + c100 * xd;
    scalar c01 = c001 * (1 - xd) + c101 * xd;
    scalar c10 = c010 * (1 - xd) + c110 * xd;
    scalar c11 = c011 * (1 - xd) + c111 * xd;
    scalar c0 = c00 * (1 - yd) + c10 * yd;
    scalar c1 = c01 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
}

SGP::PointGrid3 SGP::Grid3D::getGrid(const SliceIndex3 &sizes, const GridEmbedder &embedder)
{
    PointGrid3 G(sizes);
    StopWatch profiler;
    for(int i=0;i<sizes[0];i++)
        for(int j=0;j<sizes[1];j++)
            for(int k=0;k<sizes[2];k++)
                G(i,j,k) = embedder*vec(i,j,k);
    return G;
}

int SGP::Grid3D::closestOnGrid(const vec &x, const PointGrid3 &G) {
    scalar d = 1000;
    int closest = -1;
    for (auto i : range(G.getSize())) {
        vec p = G.at(i);
        scalar dd = (p - x).squaredNorm();
        if (dd < d) {
            d = dd;
            closest = i;
        }
    }
    return closest;
}

SGP::Vec SGP::Grid3D::Vectorize(const VectorGrid3 &X)
{
    Vec V(X.getSize()*dim);
    for (auto i : range(X.getSize())) {
        auto x = X.at(i);
        for (auto j : range(dim))
            V(dim*i + j) = x(j);
    }
    return V;
}

std::vector<SGP::SliceIndex3> SGP::Grid3D::getNeighbors(const SliceIndex3 &I,const PointGrid3& G)
{
    std::vector<SliceIndex3> neighbors;
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++) {
                if (di == 0 && dj == 0 && dk == 0)
                    continue;
                SliceIndex3 neighbor = {I(0) + di, I(1) + dj, I(2) + dk};
                if (G.validIndex(neighbor))
                    neighbors.push_back(neighbor);
            }
    return neighbors;
}

SGP::VectorGrid3 SGP::Grid3D::ToVecGrid(const Vec &X, int N)
{
    VectorGrid3 G(N,N,N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
                SliceIndex3 index = {i,j,k};
                vec x;
                for (auto d : range(dim))
                    x(d) = X(dim*G.getIndex(index) + d);
                G(index) = x;
            }
    return G;
}

SGP::PointGrid2 SGP::Grid2D::getGrid(int n)
{
    PointGrid2 G(n,n);
    scalar dx = 1.0 / (n-1);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
                G(i,j) = (vec2(i*dx,j*dx) - vec2(0.5,0.5))*2;
    return G;
}
