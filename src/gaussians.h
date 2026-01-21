#ifndef GAUSSIANS_H
#define GAUSSIANS_H

#include "StochasticGeometryProcessing.h"
#include "Eigen/Cholesky"
#include "PointCloud.h"
#include "sampling.h"
#include "QMC.h"

namespace SGP {


template<int dim>
Points<dim> GaussianQMC(int n) {
    throw std::runtime_error("GaussianQMC not implemented for dim != 2,3");
}

template<>
inline Points<3> GaussianQMC(int n) {
    Points<3> QMC(3,n);
    QMC = SGP::QMC3D::firstN(n);
    return QMC;
}

template<>
inline Points<2> GaussianQMC(int n) {
    Points<2> QMC(2,n);
    QMC = SGP::QMC2D::firstN(n);
    return QMC;
}

struct Gaussian {
    scalar mean;
    scalar s2;
    static scalar pdf(scalar m,scalar cov, scalar x) {
        scalar diff = x - m;
        scalar exponent = diff * diff / (2 * cov);
        scalar norm = std::sqrt(2 * M_PI * cov);
        return std::exp(-exponent) / norm;
    }
    scalar pdf(scalar x) const {
        return pdf(mean,s2,x);
    }

    static scalar cdf(scalar m,scalar cov,scalar x) {
        scalar stddev = std::sqrt(cov);
        scalar z = (x - m) / (stddev * std::sqrt(2.0));
        return 0.5 * (1.0 + std::erf(z));
    }
    scalar cdf(scalar x) const {
        return cdf(mean,s2,x);
    }
};

template<int dim>
struct MultivariateGaussian
{
    using MeanType = Vector<dim>;
    using CovType = SquareMatrix<dim>;
    MeanType mean;
    CovType sig;
    bool factorized = false;


public:

    template<class func>
    scalar evalQMC(int n,const func& f) const {
        Points<dim> samples = sample(n);
        scalar result = 0;

        for (int i = 0; i < samples.cols(); ++i) {
            result += f(samples.col(i));
        }
        return result / samples.cols();
    }

    template<class outval,class func>
    outval evalQMC(int n,const func& f) const {
        Points<dim> samples = sample(n);
        outval result = outval::Zero();

        for (int i = 0; i < samples.cols(); ++i) {
            result += f(samples.col(i));
        }
        return result / samples.cols();
    }

    Points<dim> sample(int n) const {
        Points<dim> samples = GaussianQMC<dim>(n);
        Eigen::LLT<CovType> llt(sig);
//        if (!factorized){
//            factorized = true;
//            llt.compute(sig);
//        }
        samples = llt.matrixL() * samples;
        samples.colwise() += mean;
        return samples;
    }

    MultivariateGaussian() {
        mean = MeanType::Zero(dim);
        sig = CovType::Zero(dim,dim);
    }
    MultivariateGaussian(const MeanType& m,const CovType& s):mean(m),sig(s) {}

    const MeanType& getMean() const { return mean; }
    const CovType& getCov() const { return sig; }

    MultivariateGaussian<dim-1> conditionOnFirstCoord(scalar c) const {
        using Cond = MultivariateGaussian<dim-1>;
        scalar meanw = mean(0);
        Vector<dim-1> mean_grad = mean.tail(dim-1);
        Vector<dim-1> Cxy = sig.block(1,0,dim-1,1);
        scalar var = sig(0,0);

        Vector<dim-1> cond_mean = mean_grad + Cxy*(c - meanw)/(1e-6 + var);

        Eigen::Matrix<scalar,dim-1,dim-1> cond_cov = sig.block(1,1,dim-1,dim-1);
        cond_cov -= Cxy*Cxy.transpose()/(1e-6 + var);
        return MultivariateGaussian<dim-1>(cond_mean,cond_cov);
    }

    Gaussian conditionOnLastCoords(const Vector<dim-1>& c) const {
        using Cond = MultivariateGaussian<dim-1>;
        scalar meanx = mean(0);
        Vector<dim-1> mean_grad = mean.tail(dim-1);
        Vector<dim-1> Cxy = sig.block(1,0,dim-1,1);
        SquareMatrix<dim-1> Cyy_inv = sig.block(1,1,dim-1,dim-1);
        Cyy_inv = Cyy_inv.completeOrthogonalDecomposition().pseudoInverse();
        scalar mean_c = meanx + Cxy.dot(Cyy_inv*(c - mean_grad));
        scalar var_c = sig(0,0) - Cxy.dot(Cyy_inv*Cxy);
        return Gaussian(mean_c,var_c);
    }

    MultivariateGaussian<dim-1> dropFirstCoord() const {
        Vector<dim-1> new_mean = mean.tail(dim-1);
        SquareMatrix<dim-1> new_cov = sig.block(1,1,dim-1,dim-1);
        return MultivariateGaussian<dim-1>(new_mean, new_cov);
    }

    scalar pdf(const Vector<dim>& x) {
        Vector<dim> diff = x - mean;
        // solve for Cov^-1 * diff
        Vector<dim> adj = sig.colPivHouseholderQr().solve(diff);
        scalar exponent = diff.dot(adj);
        scalar norm = std::sqrt(std::pow(2 * M_PI, dim) * sig.determinant());
        return std::exp(-0.5 * exponent) / norm;
    }


    static scalar pdf(const MeanType& m, const CovType& precision, const Vector<dim>& x) {
        Vector<dim> diff = x - m;
        scalar exponent = diff.transpose() * precision * diff;
        scalar norm = std::sqrt(std::pow(2 * M_PI, dim) / (1e-8+ precision.determinant()));
        return std::exp(-0.5*exponent) / norm;
    }

    MultivariateGaussian<dim> operator+(const MultivariateGaussian<dim>& other) const {
        MultivariateGaussian<dim> rslt;
        rslt.mean = mean + other.mean;
        rslt.sig = sig + other.sig;
        return rslt;
    }

    void operator+=(const MultivariateGaussian<dim>& other) {
        mean += other.mean;
        sig += other.sig;
    }

    void addNoise(scalar s) {
        sig += s*s*CovType::Identity(dim,dim);
    }
};




inline scalar normalCDF(scalar x, scalar mean, scalar variance) {
    scalar stddev = std::sqrt(variance);
    scalar z = (x - mean) / (stddev * std::sqrt(2.0));
    return 0.5 * (1.0 + std::erf(z));
}

using GaussianGrid = Tensor<dim,MultivariateGaussian<dim+1>>;

}

#endif // GAUSSIANS_H
