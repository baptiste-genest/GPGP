#ifndef STOCHASTICCALCULUS_H
#define STOCHASTICCALCULUS_H

#include "StochasticGeometryProcessing.h"
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include "SellingsAlgorithm.h"
#include "gaussians.h"
#include <queue>
#include <set>

namespace SGP {

template<int N,class T>
Tensor<N,T> BlockConvolve(const Tensor<N,T>& G) {
    // average filter, 3x3x3
    auto S = G.getSizes();
    Tensor<N,T> rslt(S);
    for (auto i : range(S[0])) {
        for (auto j : range(S[1])) {
            for (auto k : range(S[2])) {
                SliceIndex index = {i,j,k};
                int count = 1;
                T sum = G(index)*count;
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        for (int dk = -1; dk <= 1; dk++) {
                            if (!di && !dj && !dk)
                                continue;
                            SliceIndex neighbor = {i+di,j+dj,k+dk};
                            if (G.validIndex(neighbor)) {
                                sum += G(neighbor);
                                count++;
                            }
                        }
                    }
                }
                rslt(index) = sum / count;
            }
        }
    }
    return rslt;
}


template<int N,class T>
Tensor<N,T> CenterConvolve(const Tensor<N,T>& G) {
    // average filter, 3x3x3
    auto S = G.getSizes();
    for (auto& s : S)
        s -= 1;
    Tensor<N,T> rslt(S);
    for (auto i : range(S[0])) {
        for (auto j : range(S[1])) {
            for (auto k : range(S[2])) {
                SliceIndex index = {i,j,k};
                T sum = G(index);
                int count = 1;
                for (int di = 0; di <= 1; di++) {
                    for (int dj = 0; dj <= 1; dj++) {
                        for (int dk = 0; dk <= 1; dk++) {
                            if (!di && !dj && !dk)
                                continue;
                            SliceIndex neighbor = {i+di,j+dj,k+dk};
                            if (G.validIndex(neighbor)) {
                                sum += G(neighbor);
                                count++;
                            }
                        }
                    }
                }
                rslt(index) = sum / count;
            }
        }
    }
    return rslt;
}

struct SparseNarrowBand;

using GridEmbedder = Eigen::Transform<scalar,dim,Eigen::Affine>;


class StochasticCalculus {
public:

    using ValueGradientGaussian = MultivariateGaussian<dim+1>;
    enum BoundaryCondition {
        Dirichlet,
        Neumann
    };

    static scalar max_mu,min_mu;

    struct ComputeNode {
        scalar mu,field;
        SquareMatrix<dim> G;
        VoronoiReduction<dim> stencil;
        int id;
        GaussianValueGradient pred;
    };
    static int NQMC;
    static scalar computeSurfaceProb(const ValueGradientGaussian& x,scalar iso);
    static SquareMatrix<dim> computeDiffusionTensor(const ValueGradientGaussian& p,scalar iso);

protected:
    static SliceIndex delta(int i);

    GridElement bbox;

    using NodeMap = std::unordered_map<size_t,ComputeNode>;

    NodeMap nodes;

    int padding_count = 0;



    std::optional<size_t> getHash(const GridElement& I) const {
        if (!checkInBbox(I))
            return std::nullopt;
        //get id in bbox
        size_t id = I(0);
        size_t mul = bbox(0);
        for (int i = 1; i < dim; i++) {
            id += I(i)*mul;
            mul *= bbox(i);
        }
        return id;
    }

    bool checkInBbox(const GridElement& I) const {
        for (int i = 0; i < dim; i++)
            if (I(i) < 0 || I(i) >= bbox(i))
                return false;
        return true;
    }

    SliceIndex getGridCoord(int h) const {
        // get grid coord in bbox from id
        SliceIndex I;
        for (int i = 0; i < dim; i++) {
            I(i) = h % bbox(i);
            h /= bbox(i);
        }
        return I;
    }

    GridEmbedder embedder;

    scalar iso;

    static scalar getTikhonovParameter(const smat& L){
        Vec R = Vec::Random(L.cols());
        for (int i = 0; i < 10; i++){
            R = L*R;
            R.normalize();
        }
        return R.dot(L*R);
    }

    scalar dx = -1;


    std::optional<size_t> getRestrictionIndex(const SliceIndex& I) const {
        auto h = getHash(I);
        if (!h)
            return std::nullopt;
        auto it = nodes.find(*h);
        if (it == nodes.end())
            return std::nullopt;
        return it->second.id;
    }

    bool visibilityCheck(const SliceIndex& Ip,const SliceIndex& q) const;

    std::map<std::string,smat> operatorCache;

    void computeVoronoiStencil();



public:

    // builders
    StochasticCalculus() {}
    StochasticCalculus(const GaussianGrid& random_field,scalar iso,scalar eps,bool padding = false);
    StochasticCalculus(const SparseNarrowBand& SNB);// : nodes(nodes),bbox(bbox) {}


    // getters
    const GridEmbedder& getEmbedder() const {
        return embedder;
    }

    scalar getDx() const {
        if (dx < 0)
            // error
            throw std::runtime_error("dx not set");
        return dx;
    }

    inline const NodeMap& getNodes() const {return nodes;}

    int bboxSize() const {
        //return number of cubes in bbox
        return bbox.prod();
    }

    int getNumVariables() const {return nodes.size();}

    ScalarGrid getProbField() const;
    ScalarGrid getValueField(scalar fill = 0) const;

    Vec getGPISMean() const;
    Vec getMu() const;
    Points<dim> embedNarrowBand() const;

    scalar expectedMass() {
        smat M = buildMassMatrix();
        return M.sum();
    }


    // narrow band interaction
    ScalarGrid extend(const Vec& x,scalar fill = 0);
    VectorGrid extendVec(const Vec& x);
    ScalarGrid extendFill(const Vec& x);

    Vec restrict(const ScalarGrid& x);
    Vec restrictVec(const VectorGrid& x);
    Vec restrictMat(const TensorGrid& x);

    scalar lerp(const Vector<dim>& x,const Vec& f) const;
    Vector<dim> lerpVec(const Vector<dim>& x,const Vec& f) const;

    // operator building
    smat buildTangentTensorMatrix();
    smat buildFiniteDifferenceGradient(bool forward = true,BoundaryCondition bc = Dirichlet);
    smat buildDiagMu(bool vec = false);
    smat buildDiagInvMu(bool vec = false);
    smat buildGradient(BoundaryCondition bc = Dirichlet);
    smat buildIntegratedDivergence(BoundaryCondition bc = Dirichlet);
    smat buildVoronoiLaplace(scalar offset = 1e-6,BoundaryCondition bc = Dirichlet);
    smat buildNaiveLaplacian(scalar offset = 1e-4,BoundaryCondition bc = Dirichlet);
    smat buildMassMatrix();


};

struct SparseNarrowBand {
    GridMap<StochasticCalculus::ComputeNode> elements;
    GridElement bbox;
    scalar dx;
    scalar iso;
    GridEmbedder embedder;
};

inline mat Rot90(const vec& x) {
    //build cross product matrix
    mat R = mat::Zero(3,3);
    R(0,1) = -x(2);
    R(0,2) = x(1);
    R(1,0) = x(2);
    R(1,2) = -x(0);
    R(2,0) = -x(1);
    R(2,1) = x(0);
    return R;
}

}

#endif // STOCHASTICCALCULUS_H
