#ifndef GAUSSIANPOINTCLOUD_H
#define GAUSSIANPOINTCLOUD_H

#include "StochasticGeometryProcessing.h"
#include "Mesh.h"
#include "gaussians.h"

namespace SGP {

template<int D = dim>
struct GaussianDipole
{
    Vector<D> p = Vector<D>::Zero(),n = Vector<D>::Zero();
    // SquareMatrix<dim> CovMoment = SquareMatrix<dim>::Zero();//,PosMoment;
    // SquareMatrix<dim> CovPos = SquareMatrix<dim>::Zero();//,PosMoment;
    SquareMatrix<2*D> FullCov = SquareMatrix<2*D>::Zero();

    void operator+=(const GaussianDipole& other) {
        p += other.p;
        n += other.n;
        // CovMoment += other.CovMoment;
        FullCov += other.FullCov;
    }
};

template<int D = dim>
class PosList {
public:
    virtual Vector<D> getPos(int i) const = 0;
    virtual int size() const = 0;
};


template<int D = dim>
class EigenPosList : public PosList<D> {
public:
    Points<D> pts;

    Vector<D> getPos(int i) const override {
        return pts.col(i);
    }

    int size() const override { return pts.cols();  }

    EigenPosList(const Points<D>& P) : pts(P) {}
    EigenPosList() {}
};

template<int D = dim>
class GaussianDipoles : public PosList<D> {
    std::vector<GaussianDipole<D>> dipoles;

public:


    const std::vector<GaussianDipole<D>>& getDipoles() const {return dipoles;}

    inline int size() const override {return dipoles.size();}

    inline Vector<D>  getPos(int i) const override {
        return dipoles[i].p;
    }
    inline const Vector<D>&  getMoment(int i) const {
        return dipoles[i].n;
    }

    inline SquareMatrix<D> getPosCovariance(int i) const {
        return dipoles[i].FullCov.block(0,0,D,D);
    }
    inline SquareMatrix<D> getMomentCovariance(int i) const {
        return dipoles[i].FullCov.block(D,D,D,D);
    }

    inline const SquareMatrix<2*D>& getJointCovariance(int i) const {
        // spdlog::info("unimplemented getMomentCovariance, returning FullCov block");

        return dipoles[i].FullCov;
        // return dipoles[i].CovMoment;
    }


    inline const GaussianDipole<D>& operator[](int i) const {
        return dipoles[i];
    }

    friend GaussianDipoles<3> GaussianDipolesFromLidarScan(const Points<3>& P,const Points<3>& W,const Vec& S,scalar sig_min,scalar sig_max);
    friend GaussianDipoles<3> GaussianDipolesFromTriangleSoup(const Mesh& M,scalar sigma);
    friend GaussianDipoles<2> GaussianDipolesFromPolyline(const Points<2>& P,scalar sigma);

    static GaussianDipoles<D> Subsample(const GaussianDipoles<D>& P,int n,std::vector<int>* kept = nullptr) {
        // random shuffle then take first n
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<int> indices(P.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);
        GaussianDipoles<D> rslt;
        rslt.dipoles.reserve(std::min(n,(int)P.size()));
        for (int i = 0; i < n && i < P.size(); i++)
            rslt.dipoles.push_back(P.dipoles[indices[i]]);

        if (kept) {
            kept->clear();
            kept->insert(kept->end(),indices.begin(),indices.begin()+std::min(n,(int)P.size()));
        }
        return rslt;
    }

    GaussianDipoles() {}
    GaussianDipoles(const Points<D>& P) {
        int n = P.cols();
        dipoles.resize(n);
        for (int i = 0; i < n; i++) {
            dipoles[i].p = P.col(i);
            dipoles[i].n = Vector<D>::Zero();
            dipoles[i].FullCov = SquareMatrix<2*D>::Identity()*1e-6;
        }
    }
    GaussianDipoles(const Points<D>& P,const Points<D>& W,const std::vector<SquareMatrix<2*D>>& Covs) {
        int n = P.cols();
        dipoles.resize(n);
        for (int i = 0; i < n; i++) {
            dipoles[i].p = P.col(i);
            dipoles[i].n = W.col(i);
            dipoles[i].FullCov = Covs[i];
        }
    }

    void save(const std::string& filename)
    {
        std::ofstream out(filename);
        if (!out)
            throw std::runtime_error("Cannot open file");

        out << D << " " << size() << "\n";

        for (int i = 0; i < size(); ++i) {
            const auto& d = dipoles[i];

            for (int k = 0; k < D; ++k)
                out << d.p(k) << " ";
            out << "\n";

            for (int k = 0; k < D; ++k)
                out << d.n(k) << " ";
            out << "\n";

            for (int r = 0; r < 2 * D; ++r) {
                for (int c = 0; c < 2 * D; ++c)
                    out << d.FullCov(r, c) << " ";
                out << "\n";
            }
        }
    }

    GaussianDipoles(const std::string& filename)
    {
        std::ifstream in(filename);
        if (!in)
            throw std::runtime_error("Cannot open file");

        int dim;
        size_t count;
        in >> dim >> count;

        if (dim != D)
            throw std::runtime_error("Dimension mismatch");

        GaussianDipoles<D> G;
        dipoles.resize(count);

        for (size_t i = 0; i < count; ++i) {
            auto& d = dipoles[i];

            for (int k = 0; k < D; ++k)
                in >> d.p(k);

            for (int k = 0; k < D; ++k)
                in >> d.n(k);

            for (int r = 0; r < 2 * D; ++r)
                for (int c = 0; c < 2 * D; ++c)
                    in >> d.FullCov(r, c);
        }
    }


    scalar estimateScale() const {
        scalar s = 0;
        for ( auto i : range(size()))
            s += std::sqrt(4./std::sqrt(3.)*getMoment(i).norm());
        return s/size();
    }

    void normalize(scalar f) {
        // normalize points to fit in box of size [-f,f]
        // properly scale moments and covariances

        Vector<D> min_pt = Vector<D>::Constant(std::numeric_limits<scalar>::max());
        Vector<D> max_pt = Vector<D>::Constant(std::numeric_limits<scalar>::lowest());
        for ( auto i : range(size())) {
            Vector<D> p = getPos(i);
            min_pt = min_pt.cwiseMin(p);
            max_pt = max_pt.cwiseMax(p);
        }
        Vector<D> center = 0.5*(min_pt + max_pt);
        scalar scale = (max_pt - min_pt).maxCoeff() / (2.*f);
        for ( auto i : range(size())) {
            dipoles[i].p = (dipoles[i].p - center)/scale;
            dipoles[i].n /= scale;
            dipoles[i].FullCov /= (scale*scale);
        }

    }

    Points<dim> getPositions() const {
        Points<D> P(D,size());
        for ( auto i : range(size()))
            P.col(i) = getPos(i);
        return P;
    }

    Points<dim> getMoments() const {
        Points<D> W(D,size());
        for ( auto i : range(size()))
            W.col(i) = getMoment(i);
        return W;
    }



};

GaussianDipoles<3> GaussianDipolesFromTriangleSoup(const Mesh& M,scalar sigma);
GaussianDipoles<2> GaussianDipolesFromPolyline(const Points<2>& P,scalar sigma);
GaussianDipoles<3> GaussianDipolesFromLidarScan(const Points<3>& P,const Points<3>& W,const Vec& S,scalar sig_min,scalar sig_max);

scalar GetAverageIso(const PosList<dim>& GD,const GPIS& gp);


}

#endif // GAUSSIANPOINTCLOUD_H
