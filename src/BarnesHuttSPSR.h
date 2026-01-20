#ifndef BARNESHUTTSPSR_H
#define BARNESHUTTSPSR_H

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <Eigen/Dense>

#include "StochasticGeometryProcessing.h"
#include "gaussians.h"
#include "GaussianPointCloud.h"

namespace SGP {

using pred = MultivariateGaussian<SGP::dim+1>;

class BarnesHuttStochasticPSR : public GPIS {
public:
    using position = Vector<dim>;

    position center;
    scalar halfSize,radius;
    std::optional<GaussianDipole<dim>> point;

    scalar weight = 0;
    int nb_below = 0;

    bool isLeaf = true;

    GaussianDipole<dim> weightedCenter;

    using nodeptr = std::unique_ptr<BarnesHuttStochasticPSR>;

    static constexpr int nb_children = 1 << dim;

    std::array<nodeptr,nb_children> children;

    scalar beta = 2.3;

    using Moment = Eigen::Matrix<scalar,dim,dim>;

    Moment order1 = Moment::Zero();
    Moment weightedMoment = Moment::Zero();


    scalar s;


    BarnesHuttStochasticPSR(const position& center_, scalar halfSize_,scalar s,scalar b);
    BarnesHuttStochasticPSR(scalar reg,scalar beta)
        : BarnesHuttStochasticPSR(position::Zero(),1,reg,beta){    }

    bool contains(const position& position) const;

    int getOctant(const position& position) const noexcept;

    void insert(const GaussianDipole<dim>& b);


    void computeRadius();

    const scalar factor = std::sqrt(2/M_PI);

    using kernel = Eigen::Matrix<scalar,dim+1,dim>;
    using full_kernel = Eigen::Matrix<scalar,dim+1,2*dim>;

    struct PSRKernels {
        kernel Kn = kernel::Zero(),Kp = kernel::Zero();
        full_kernel KF = full_kernel::Zero();
    };

    PSRKernels computeKernels(const Vector<dim>& x,const GaussianDipole<dim>& p) const;

    scalar PSRPotential(const Vector<dim>& p,const GaussianDipole<dim>& q) const;

    struct BarnesHuttOutput {
        scalar value = 0;
        MultivariateGaussian<dim+1> prediction;
    };

    void compute(const position& p,GaussianValueGradient& rslt) const;
    void computeIter(const position& p,GaussianValueGradient& rslt) const;

    void precomputeMoments();

    GaussianValueGradient predict(const position& b) const override;

    void getCenters(std::vector<Vector<dim>>& C) const;

    scalar getMeanValue(const GaussianDipoles<dim>& P) const;

private:
    void inner_computeMaxRadius(std::vector<Vector<dim>>& points_below);
    void subdivide();

    void insertToChildren(const GaussianDipole<dim>& b);
};

}
#endif // BARNESHUTTSPSR_H
