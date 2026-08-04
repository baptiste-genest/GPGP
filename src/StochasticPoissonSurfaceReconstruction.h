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

template<int D>
struct JointKernelTerms {
    scalar r = 0;
    Vector<D> evec = Vector<D>::Zero();
    Vector<D> grad_u = Vector<D>::Zero();
    SquareMatrix<D> hess_u = SquareMatrix<D>::Zero();
    scalar A = 0;
    scalar Aprime = 0;
    scalar s = 1;
    bool degenerate = true;

    SquareMatrix<D> hessOfDot(const Vector<D>& n) const {
        if (degenerate)
            return SquareMatrix<D>::Zero();
        SquareMatrix<D> I = SquareMatrix<D>::Identity();
        SquareMatrix<D> eeT = evec*evec.transpose();
        scalar edotn = evec.dot(n);
        SquareMatrix<D> H = Aprime*edotn*eeT
                          + (A/r)*edotn*(I - eeT)
                          + (A/r)*(evec*n.transpose() + n*evec.transpose());
        return 0.5*(H + H.transpose());
    }

    PoissonKernel<D> Kn() const {
        PoissonKernel<D> K;
        K.row(0) = grad_u.transpose();
        K.template block<D,D>(1,0) = hess_u;
        return s*K;
    }

    PoissonKernel<D> Kp(const Vector<D>& n) const {
        PoissonKernel<D> K;
        K.row(0) = (hess_u*n).transpose();
        K.template block<D,D>(1,0) = hessOfDot(n);
        return s*K;
    }

    PoissonKernel<D> P(int j) const { return Kp(Vector<D>::Unit(j)); }

    JointPoissonKernel<D> joint(const Vector<D>& n) const {
        JointPoissonKernel<D> FK;
        FK.template block<D+1,D>(0,0) = Kp(n);
        FK.template block<D+1,D>(0,D) = Kn();
        return FK;
    }
};

namespace PSR2D {
JointKernelTerms<2> ComputeJointKernelTerms(const vec2 &x, const vec2 &p, scalar s);
}

namespace PSR3D {
JointKernelTerms<3> ComputeJointKernelTerms(const vec &x, const vec &p, scalar s);
}

template<int D>
JointKernelTerms<D> ComputeJointKernelTerms(const Vector<D> &x, const Vector<D> &p, scalar s) {
    if constexpr (D == 2)
        return PSR2D::ComputeJointKernelTerms(x,p,s);
    else
        return PSR3D::ComputeJointKernelTerms(x,p,s);
}

template<int D>
JointPoissonKernel<D> ComputeJointPoissonKernel(const Vector<D> &x, const Vector<D> &p, const Vector<D> &n, scalar s) {
    return ComputeJointKernelTerms<D>(x,p,s).joint(n);
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
