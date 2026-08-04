#ifndef STOCHASTICBARNESHUTT_H
#define STOCHASTICBARNESHUTT_H

#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <Eigen/Dense>

#include "StochasticGeometryProcessing.h"
#include "gaussians.h"
#include "GaussianPointCloud.h"
#include "StochasticPoissonSurfaceReconstruction.h"


namespace SGP {

class StochasticBarnesHutt : public GPIS {

public:

    using HashKey = size_t;
    using position = Vector<dim>;

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

        std::array<std::array<SquareMatrix<dim>,dim>,dim> Cmom;
        std::array<SquareMatrix<dim>,dim> Ccross;
        SquareMatrix<dim> Cnn = SquareMatrix<dim>::Zero();

        HashKey key = 0;

        void addDipole(const GaussianDipole<dim>& b) {
            scalar w = b.n.norm();
            weight += w;
            weightedCenter.p += b.p*w;
            weightedCenter.n += b.n;

            SquareMatrix<dim> Cpp = b.FullCov.template block<dim,dim>(0,0);
            SquareMatrix<dim> Cpn = b.FullCov.template block<dim,dim>(0,dim);
            for (int j = 0; j < dim; ++j) {
                for (int l = 0; l < dim; ++l)
                    Cmom[j][l] += b.n(j)*b.n(l)*Cpp;
                Ccross[j] += b.n(j)*Cpn;
            }
            Cnn += b.FullCov.template block<dim,dim>(dim,dim);
        }

        SquareMatrix<dim+1> assembleCovariance(const JointKernelTerms<dim>& t) const {
            PoissonKernel<dim> Kn = t.Kn();
            std::array<PoissonKernel<dim>,dim> P;
            for (int j = 0; j < dim; ++j)
                P[j] = t.P(j);

            SquareMatrix<dim+1> C = Kn*Cnn*Kn.transpose();
            for (int j = 0; j < dim; ++j) {
                SquareMatrix<dim+1> cross = P[j]*Ccross[j]*Kn.transpose();
                C += cross + cross.transpose();
                for (int l = 0; l < dim; ++l)
                    C += P[j]*Cmom[j][l]*P[l].transpose();
            }
            return SquareMatrix<dim+1>(0.5*(C + C.transpose()));
        }

        void zeroMoments() {
            for (auto& row : Cmom)
                for (auto& m : row)
                    m.setZero();
            for (auto& m : Ccross)
                m.setZero();
            Cnn.setZero();
        }

        StochasticBarnesHuttNode(HashKey k,const position& center_, scalar halfSize_) : key(k), center(center_), halfSize(halfSize_) { zeroMoments(); }
        StochasticBarnesHuttNode() { zeroMoments(); }

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

    scalar beta;
    scalar s;

    std::unordered_map<HashKey,StochasticBarnesHuttNode> nodes;

    void insertAtNode(HashKey node_key,const GaussianDipole<dim>& b);
    void compute(HashKey key,const position& p,GaussianValueGradient& rslt) const;

public:

    StochasticBarnesHutt(scalar s,scalar b,int size = -1) : s(s),beta(b) {
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

private:
    void inner_computeMaxRadius(HashKey key,std::vector<Vector<dim>>& points_below);

    void insertToChildren(HashKey key,const GaussianDipole<dim>& b);
};

}

#endif // STOCHASTICBARNESHUTT_H
