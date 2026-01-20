#ifndef STOCHASTICBARNESHUTT_H
#define STOCHASTICBARNESHUTT_H

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <Eigen/Dense>

#include "StochasticGeometryProcessing.h"
#include "gaussians.h"
#include "GaussianPointCloud.h"


namespace SGP {

class StochasticBarnesHutt : public GPIS {

public:

    using HashKey = size_t;
    using position = Vector<dim>;
    const scalar factor = std::sqrt(2/M_PI);

    using kernel = Eigen::Matrix<scalar,dim+1,dim>;
    using full_kernel = Eigen::Matrix<scalar,dim+1,2*dim>;

    struct PSRKernels {
        kernel Kn = kernel::Zero(),Kp = kernel::Zero();
        full_kernel KF = full_kernel::Zero();
    };

private:
    static constexpr int nb_children = 1 << dim;

    struct StochasticBarnesHuttNode {

        position center;
        scalar halfSize,radius;
        std::optional<GaussianDipole<dim>> point;

        scalar weight = 0;
        int nb_below = 0;

        bool isLeaf = true;
        GaussianDipole<dim> weightedCenter;
        int active_child_mask = 0;
        // std::array<HashKey,nb_children> children;

        HashKey key = 0;

        StochasticBarnesHuttNode(HashKey k,const position& center_, scalar halfSize_) : key(k), center(center_), halfSize(halfSize_) {    }
        StochasticBarnesHuttNode() {}

        bool contains(const position& position) const;

        int getOctant(const position& position) const noexcept;

        HashKey getChildKey(HashKey oct) const;

        bool isChildActive(int octant) const {
            return (active_child_mask & (1 << octant)) != 0;
        }

        void setChildActive(int octant) {
            active_child_mask |= (1 << octant);
        }

    };

    scalar beta = 2.3;
    scalar s;

    // struct IdentityHash {
    //     size_t operator()(uint64_t x) const noexcept { return (size_t)x; }
    // };

    // std::unordered_map<HashKey,StochasticBarnesHuttNode,IdentityHash> nodes;
    std::unordered_map<HashKey,StochasticBarnesHuttNode> nodes;
    // std::map<HashKey,StochasticBarnesHuttNode> nodes;

    void insertAtNode(HashKey node_key,const GaussianDipole<dim>& b);
    void compute(HashKey key,const position& p,GaussianValueGradient& rslt) const;

    PSRKernels computeKernels(const Vector<dim>& x,const GaussianDipole<dim>& p) const;

    scalar PSRPotential(const Vector<dim>& p,const GaussianDipole<dim>& q) const;

public:

    void sanityCheck();

    StochasticBarnesHutt(scalar s,scalar b,int size = -1) : s(s),beta(b) {
        // if (size > 0)
        //     nodes.reserve(size+1);
        nodes[1] = StochasticBarnesHuttNode(1,position::Zero(),1);
    }

    StochasticBarnesHutt(const GaussianDipoles<dim>& input,scalar s,scalar b) : StochasticBarnesHutt(s,b,input.size()) {
        for (auto i : range(input.size()))
            insert(input[i]);

        precomputeMoments();
        computeRadius();
    }


    void insert(const GaussianDipole<dim>& b);
    void computeRadius();
    void precomputeMoments();


    GaussianValueGradient predict(const position& b) const override;

    void getCenters(std::vector<Vector<dim>>& C) const;

private:
    // void subdivide(StochasticBarnesHuttNode& node);
    void inner_computeMaxRadius(HashKey key,std::vector<Vector<dim>>& points_below);

    void insertToChildren(HashKey key,const GaussianDipole<dim>& b);
};

}

#endif // STOCHASTICBARNESHUTT_H
