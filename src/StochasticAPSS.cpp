#include "StochasticAPSS.h"


SGP::scalar SGP::StochasticAPSS::getMeanValue() const
{
    scalar m = 0;
    for (auto i : range(nb_points))
        m += evalField(input->getPos(i));
    return m/nb_points;
}

SGP::Vector<SGP::dim> SGP::StochasticAPSS::delta(int i) {
    Vector<dim> d = Vector<dim>::Zero(dim);
    d(i) = 1;
    return d;
}

SGP::StochasticAPSS::gaussian SGP::StochasticAPSS::pred(const Vector<dim> &x) const {
    Vec w(nb_points);
    GaussianVector<dim> avgn;
    Vector<dim> wp = Vector<dim>::Zero();
    scalar wpp = 0;
    for (auto i : range(nb_points)) {
        const Vector<dim>& p = input->getPos(i);
        w[i] = A[i]/std::pow((p-x).squaredNorm() + 0.001,2);
        wp += w[i]*p;
        wpp += w[i]*p.squaredNorm();
    }
    scalar sum = w.sum();
    if (sum < 1e-6)
        return gaussian();
    sum += 1e-6;
    scalar denum = 2*(wpp - wp.squaredNorm()/sum) + 1e-4;

    gaussian uq,uc;
    for (auto i : range(nb_points)) {
        const Vector<dim>& p = input->getPos(i);
        avgn += normals[i]*(w[i]/sum);
        uq += normals[i].dot(p*(w[i]/denum));
    }
    uq -= avgn.dot(wp/denum);
    GaussianVector<dim> ul = avgn - uq*Vector<dim>(wp*2/(sum));
    uc = ul.dot(-wp/sum) - uq*(wpp/sum);
    return gaussian((uc + ul.dot(x) + uq*x.squaredNorm()));
}

SGP::scalar SGP::StochasticAPSS::evalField(const Vector<dim> &x) const {
    return pred(x).mean(0);
}

SGP::StochasticAPSS::StochasticAPSS(GaussianDipoles<dim> *pc) : input(pc) {
    nb_points = input->size();
    normals.resize(nb_points);
    A = Vec(nb_points);
    for (auto i : range(nb_points)) {
        normals[i].mean = input->getMoment(i);
        normals[i].covariance = input->getMomentCovariance(i);
        A[i] = input->getMoment(i).norm();
        normals[i] /= A[i];
    }

}

SGP::scalar SGP::InlineAPPS(const std::vector<Vector<dim>> &p, const std::vector<Vector<dim>> &moments, const Vector<dim> &x)
{
    int nb_points = p.size();
    Vec w(nb_points);
    Vector<dim> wp = Vector<dim>::Zero(dim);
    Vector<dim> avgn = Vector<dim>::Zero(dim);
    scalar wpp = 0;
    for (auto i : range(nb_points)) {
        w[i] = moments[i].norm()/std::pow((p[i]-x).squaredNorm() + 0.001,2);
        wp += w[i]*p[i];
        wpp += w[i]*p[i].squaredNorm();
    }
    scalar sum = w.sum();
    sum += 1e-6;
    scalar denum = 2*(wpp - wp.squaredNorm()/sum) + 1e-4;

    scalar uq = 0,uc = 0;
    for (auto i : range(nb_points)) {
        avgn += moments[i].normalized()*(w[i]/sum);
        uq += moments[i].normalized().dot(p[i]*(w[i]/denum));
    }
    uq -= avgn.dot(wp/denum);
    Vector<dim> ul = avgn - uq*(wp*2/(sum));
    uc = ul.dot(-wp/sum) - uq*(wpp/sum);
    return uc + ul.dot(x) + uq*x.squaredNorm();
}
