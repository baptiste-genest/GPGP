#ifndef STOCHASTICAPSS_H
#define STOCHASTICAPSS_H

#include "GaussianPointCloud.h"
#include "gaussians.h"

namespace SGP {


template<int dim>
struct GaussianVector {
    using MeanType = Vector<dim>;
    using CovType = Eigen::Matrix<scalar,dim,dim>;
    using GaussianType = MultivariateGaussian<dim>;

    MeanType mean;
    CovType covariance;

    GaussianVector() {
        mean = MeanType::Zero(dim);
        covariance = CovType::Zero(dim,dim);
    }

    static GaussianVector<1> Gaussian1D(scalar mean, scalar variance) {
        using gauss = MultivariateGaussian<1>;
        return GaussianVector<1>(Vector<1>(mean), Eigen::Matrix<scalar,1,1>(variance));
    }

    GaussianVector(const MeanType& m, const CovType& c)
        : mean(m), covariance(c) {}

    GaussianVector operator+(const GaussianVector& other) const {
        return GaussianVector(mean + other.mean, covariance + other.covariance);
    }

    GaussianVector operator-(const GaussianVector& other) const {
        return GaussianVector(mean - other.mean, covariance + other.covariance);
    }
    GaussianVector operator*(scalar factor) const {
        return GaussianVector(mean * factor, covariance * factor * factor);
    }

    GaussianVector operator/(scalar factor) const {
        return GaussianVector(mean / factor, covariance / (factor * factor));
    }

    GaussianVector& operator+=(const GaussianVector& other) {
        mean += other.mean;
        covariance += other.covariance;
        return *this;
    }

    GaussianVector& operator-=(const GaussianVector& other) {
        mean -= other.mean;
        covariance += other.covariance;
        return *this;
    }

    GaussianVector& operator*=(scalar factor) {
        mean *= factor;
        covariance *= factor * factor;
        return *this;
    }

    GaussianVector& operator/=(scalar factor) {
        mean /= factor;
        covariance /= (factor * factor);
        return *this;
    }
    GaussianVector<1> dot(const Vector<dim>& x) const {
        using gauss = MultivariateGaussian<1>;
        return Gaussian1D(mean.dot(x),x.dot(covariance*x));
    }
};

template<int D>
inline GaussianVector<D> operator*(const GaussianVector<1>& g,const Vector<D>& x) {
    return GaussianVector<D>(g.mean(0) * x, g.covariance(0,0) * SquareMatrix<dim>(x * x.transpose()));
}

template<int N,int D>
inline GaussianVector<N> operator*(const Eigen::Matrix<scalar,N,D>& x,const GaussianVector<D>& g) {
    return GaussianVector<N>(x * g.mean, x * g.covariance * x.transpose());
}

inline scalar pdf(const GaussianVector<1>& g,scalar s) {
    //    scalar p = std::exp(-std::pow(m - iso,2)/(2*s2))/std::sqrt(2*M_PI*s2);
    scalar s2 = g.covariance(0,0);
    scalar diff = s - g.mean(0);
    return std::exp(-diff*diff/2/s2) / std::sqrt(2*M_PI*s2);
}

struct StochasticAPSS {
    GaussianDipoles<dim>* input;
    scalar eps = 1e-5;

    Vec A;

    std::vector<GaussianVector<dim>> normals;

    scalar getMeanValue() const;

    static Vector<dim> delta(int i);

    template<class func>
    static Vector<dim> grad(const Vector<dim>& x,const func& f,scalar eps = 1e-6) {
        Vector<dim> g = Vector<dim>::Zero();
        for (auto i : range(dim)) {
            Vector<dim> d = x + eps*delta(i);
            g(i) = (f(d) - f(x))/eps;
        }
        return g;
    }

    using gaussian = GaussianVector<1>;
    gaussian pred(const Vector<dim>& x) const;


    scalar evalField(const Vector<dim>& x) const;


    int nb_points;

    StochasticAPSS(GaussianDipoles<dim>* pc);

};

scalar InlineAPPS(const std::vector<Vector<dim>>& p,const std::vector<Vector<dim>>& n,const Vector<dim>& x);

}

#endif // STOCHASTICAPSS_H
