#ifndef STOCHASTICPOISSONSURFACERECONSTRUCTION_H
#define STOCHASTICPOISSONSURFACERECONSTRUCTION_H

#include "StochasticGeometryProcessing.h"
#include "gaussians.h"
#include "PointCloud.h"
#include "GaussianPointCloud.h"
#include "Mesh.h"


namespace SGP {

template<int D>
using PoissonKernel = Eigen::Matrix<scalar,D+1,D>;

template<int D>
using JointPoissonKernel = Eigen::Matrix<scalar,D+1,2*D>;


constexpr scalar PI = 3.14159265358979323846;
constexpr scalar SQRT_PI = 1.77245385090551602729;
constexpr scalar EPS_R = 1e-6; // threshold for small-radius series

namespace PSR2D {

// Factorized function: computes grad_u, hess_u, and hess_of_dot in one pass
JointPoissonKernel<2> ComputeJointPoissonKernel(const vec2 &x, const vec2 &p, const vec2 &n, scalar s);

}


namespace PSR3D {

// Computes gradient, Hessian of u, and Hessian of dot product in one call
JointPoissonKernel<3> ComputeJointPoissonKernel(const vec &x, const vec &p, const vec &n, scalar s);

}

template<int D>
JointPoissonKernel<D> ComputeJointPoissonKernel(const Vector<D> &x, const Vector<D> &p, const Vector<D> &n, scalar s) {
    if constexpr (D == 2)
        return PSR2D::ComputeJointPoissonKernel(x,p,n,s);
    else
        return PSR3D::ComputeJointPoissonKernel(x,p,n,s);
}


struct StochasticPoissonSurfaceReconstruction : public GPIS
{
    GaussianDipoles<dim>* input;

    scalar eps = 1e-5;

    scalar s2,s;

    StochasticPoissonSurfaceReconstruction(GaussianDipoles<dim>* I,scalar reg) : input(I), s2(reg*reg),s(reg){

    }

    scalar PSR(const Vector<dim>& x) const {
        scalar rslt = 0;
        scalar s = std::sqrt(s2);
        for (auto i : range(input->size())) {
            const Vector<dim>& p = input->getPos(i);
            Vector<dim> d = x - p;
            scalar r = d.norm();
            if (r < 1e-6)
                continue;
            Vector<dim> rhat = d.stableNormalized();
            scalar er = std::erf(r/(std::sqrt(2*s2)));
            scalar dr = (std::sqrt(2)*r*std::exp(-r*r/(2*s2))/(std::sqrt(std::pow(M_PI,3)*s2)) - er/M_PI)/(4*r*r);
            //            rslt += A[i] * dr*rhat.dot(input.normals.col(i))*s;
            rslt += dr*rhat.dot(input->getMoment(i))*s;
        }
        return rslt;
    }

    scalar getIsoLevel() const {
        scalar avg_on_samples = 0;
        for (auto i : range(input->size())) {
            avg_on_samples += PSR(input->getPos(i));
        }
        avg_on_samples /= input->size();
        return avg_on_samples;
    }

    /*
    struct SPSRprediction {
        scalar value;
        Vector<dim> grad;
        scalar prob;
    };
*/

    using SPSRprediction = MultivariateGaussian<dim+1>;

    using kernel = Eigen::Matrix<scalar,dim+1,dim>;


    SPSRprediction predict(const Vector<dim>& x) const override {
        Vector<dim+1> mean = Vector<dim+1>::Zero();
        SquareMatrix<dim+1> Cov = SquareMatrix<dim+1>::Zero();

        for (auto i : range(input->size())) {
            auto FK = ComputeJointPoissonKernel(x,input->getPos(i),input->getMoment(i),s);
            Vector<2*dim> X = Vector<2*dim>::Zero();
            X.tail(dim) = input->getMoment(i);

            mean += FK*X;
            Cov += FK*input->getJointCovariance(i)*FK.transpose();
        }
        return SPSRprediction(mean, Cov);
    }



};

}

#endif // STOCHASTICPOISSONSURFACERECONSTRUCTION_H
