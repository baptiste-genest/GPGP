#ifndef GRID_H
#define GRID_H

#include "StochasticGeometryProcessing.h"
#include "StochasticCalculus.h"


namespace SGP {


struct Grid2D {

    static PointGrid2 getGrid(int n);

};

struct Grid3D {

static PointGrid3 getGrid(int n);
static PointGrid3 getGrid(const SliceIndex3& sizes,const GridEmbedder& embedder);

const int d = 3;

static ScalarGrid3 ToGrid(const Vec& X,int N);
static ScalarGrid3 Splat(const PointGrid3& G,const vec& x);

static scalar lerp(const ScalarGrid3& G,const vec& x);
static vec lerp(const VectorGrid3& G,const vec& x);
static scalar lerpNonSquare(const ScalarGrid3& G,const vec& x,const AffineMap3& map);
static vec lerpNonSquare(const VectorGrid3& G,const vec& x,const AffineMap3& map);

static int closestOnGrid(const vec& x,const PointGrid3& G);

static Vec Vectorize(const VectorGrid3& X);

static VectorGrid3 ToVecGrid(const Vec& X,int N);

static std::vector<SliceIndex3> getNeighbors(const SliceIndex3& I,const PointGrid3& G);

template<class PosReader>
static Vec Sample(PosReader r,int nb,const ScalarGrid3& G,const AffineMap3& map = AffineMap3::Identity()) {
    Vec samples = Vec::Zero(nb);
    for (auto i : range(nb)) {
        vec p = r(i);
        samples(i) = lerpNonSquare(G,p,map);
    }
    return samples;
}

};

}


#endif // GRID_H
