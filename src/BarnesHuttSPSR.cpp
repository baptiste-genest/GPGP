#include "BarnesHuttSPSR.h"

SGP::BarnesHuttStochasticPSR::BarnesHuttStochasticPSR(const position &center_, scalar halfSize_, scalar s, scalar b)
    : center(center_), halfSize(halfSize_),s(s),beta(b){
}

bool SGP::BarnesHuttStochasticPSR::contains(const position &position) const {
    for (int i = 0; i < dim; ++i)
        if (std::abs(position[i] - center[i]) > halfSize)
            return false;
    return true;
}

int SGP::BarnesHuttStochasticPSR::getOctant(const position &position) const noexcept {
    int index = 0;
    for (int i = 0; i < dim; ++i)
        if (position[i] >= center[i])
            index |= (1 << i);
    return index;
}

void SGP::BarnesHuttStochasticPSR::insert(const GaussianDipole<dim> &b) {
    if (!contains(b.p)) return;

    if (isLeaf && !point.has_value()) {
        point = b;
        weightedCenter = b;
        weightedCenter.p *= b.n.norm();
        weight += b.n.norm();
//        weightedCenter = b;
//        weightedCenter.n *= b.weight;
//        weightedMoment = b.p*b.n.transpose() * b.weight;
//        radius = 0;
        return;
    }

    if (isLeaf) {
        subdivide();
        insertToChildren(point.value());
        point = {};
        isLeaf = false;
    }
    insertToChildren(b);
    nb_below++;

    // Update center of weight
    weight += b.n.norm();
//    weightedCenter += b;
    weightedCenter.p += b.p*b.n.norm();
    weightedCenter.n += b.n;
    // weightedCenter.CovMoment += b.CovMoment;
    // weightedCenter.CovPos += b.CovPos;
    weightedCenter.FullCov += b.FullCov;
}

void SGP::BarnesHuttStochasticPSR::inner_computeMaxRadius(std::vector<Vector<dim> > &points_below) {
    if (point) {
        points_below.push_back(point.value().p);
        return;
    }
    std::vector<Vector<dim>> buffer;
    for (const auto& child : children) {
        if (child) child->inner_computeMaxRadius(buffer);
    }
    // compute max radius from points_below
//    scalar old_radius = radius;
    radius = 0;
    if (!buffer.empty()) {
        Vector<dim> max_point = buffer[0];
        for (const auto& p : buffer) {
            if ((p - weightedCenter.p).norm() > (max_point - weightedCenter.p).norm()) {
                max_point = p;
            }
        }
        radius = std::max(radius, (max_point - weightedCenter.p).norm());
        // spdlog::info("old radius {} new radius {}",old_radius,radius);
    }
    points_below.insert(points_below.end(), buffer.begin(), buffer.end());
}

void SGP::BarnesHuttStochasticPSR::computeRadius() {
    std::vector<Vector<dim>> buffer;
    buffer.reserve(nb_below);
    inner_computeMaxRadius(buffer);
}

SGP::BarnesHuttStochasticPSR::PSRKernels SGP::BarnesHuttStochasticPSR::computeKernels(const Vector<dim> &x, const GaussianDipole<dim> &p) const {
    Vector<dim> d = x - p.p;
    scalar r = d.norm();
    if (r < 1e-6)
        return {};
//        return {kernel::Zero(),kernel::Zero()};
    Vector<dim> rhat = d.stableNormalized();

    scalar s2 = s*s;

    scalar a = factor*r* std::exp(-r*r/(2*s2));
    scalar b = std::erf(r/(std::sqrt(2)*s));

    scalar dr = (a/s - b)/(4*M_PI*r*r);
    scalar d2r = (2*b - a*(r*r+2*s2)/(s2*s))/(4*M_PI*r*r*r);
    Vector<dim> k = dr*rhat;
    SquareMatrix<dim> H = (d2r - dr/r)*rhat*rhat.transpose() + dr/r*SquareMatrix<dim>::Identity();
    kernel Kn,Kp = kernel::Zero();

    full_kernel FK = full_kernel::Zero();


    Kp.block(0,0,1,dim) = (H*p.n).transpose();
    SquareMatrix<dim> Hn = (d2r/r - dr/r/r)*(rhat.dot(p.n))*SquareMatrix<dim>::Identity() +
             (3*dr/r/r - d2r)*(rhat.dot(p.n))*rhat*rhat.transpose() +
             (d2r/r - dr/r/r)*(rhat*p.n.transpose() + p.n*rhat.transpose());
    Kp.block(1,0,dim,dim) = Hn;

//    Kp *= s;

    Kn.block(0,0,1,dim) = k.transpose();
    Kn.block(1,0,dim,dim) = H;
//    Kn *= s;

    FK.block(0,0,dim+1,dim) = Kp;
    FK.block(0,dim,dim+1,dim) = Kn;

    return {Kn,Kp,FK};
}

void SGP::BarnesHuttStochasticPSR::compute(const position &p, GaussianValueGradient &rslt) const {
    scalar dist = (p - weightedCenter.p).norm();

    if (point || dist > beta*radius) {
        auto K = computeKernels(p,weightedCenter);

        Vector<2*dim> X = Vector<2*dim>::Zero();
        X.tail(dim) = weightedCenter.n;
        Vector<dim+1> mean = K.KF*X;
        SquareMatrix<dim+1> Cov = K.KF*weightedCenter.FullCov*K.KF.transpose();

//        Vector<dim+1> mean = K.Kn*weightedCenter.n;
//        SquareMatrix<dim+1> Cov = K.Kn*weightedCenter.CovMoment*K.Kn.transpose();
//        Cov += K.Kp*weightedCenter.CovPos*K.Kp.transpose();
        MultivariateGaussian<dim+1> P = MultivariateGaussian<dim+1>(
            mean,
            Cov
            );
        rslt += P;
//        rslt.value += PSRPotential(p, weightedCenter);
    } else {
        for (const auto& child : children)
            if (child) child->compute(p, rslt);
    }
}

// void SGP::BarnesHuttStochasticPSR::computeIter(const position &p, GaussianValueGradient &rslt) const
// {
//     const BarnesHuttStochasticPSR* stack[128];
//     int sp = 0; stack[sp++] = this;
//     while (sp) {
//         auto n = stack[--sp];
//         scalar dist = (p - n->weightedCenter.p).norm();
//         if (n->point || dist > beta*n->radius) {
//             auto K = computeKernels(p,n->weightedCenter);

//             Vector<2*dim> X = Vector<2*dim>::Zero();
//             X.tail(dim) = n->weightedCenter.n;
//             Vector<dim+1> mean = K.KF*X;
//             SquareMatrix<dim+1> Cov = K.KF*n->weightedCenter.FullCov*K.KF.transpose();

//             MultivariateGaussian<dim+1> P = MultivariateGaussian<dim+1>(
//                 mean,
//                 Cov
//                 );
//             rslt.prediction += P;
//             rslt.value += mean(0);
//         } else {
//             for (const auto& child : n->children)
//                 if (child) stack[sp++] = child.get();;
//         }
//     }
// }

void SGP::BarnesHuttStochasticPSR::precomputeMoments() {
    if (weight == 0)
        return;
    weightedCenter.p /= weight;
    //    built = true;
    //    order1 = weightedMoment - weightedCenter.n* weightedCenter.p.transpose();
    for (const auto& child : children) {
        if (child)
            child->precomputeMoments();
    }
}

SGP::GaussianValueGradient SGP::BarnesHuttStochasticPSR::predict(const position &b) const {
    GaussianValueGradient rslt;
    // computeIter(b, result);
    compute(b, rslt);
    // rslt.prediction.addNoise(0.01);
    return rslt;
}

void SGP::BarnesHuttStochasticPSR::getCenters(std::vector<Vector<dim> > &C) const {
    if (point)
        C.push_back(point.value().p);
    if (!isLeaf) {
        for (const auto& child : children) {
            if (child) child->getCenters(C);
        }
    }
}

SGP::scalar SGP::BarnesHuttStochasticPSR::getMeanValue(const GaussianDipoles<dim> &P) const {
    scalar rslt = 0;
    for (auto i : range(P.size())) {
        rslt += predict(P.getPos(i)).mean(0);
    }
    return rslt/P.size();
}

void SGP::BarnesHuttStochasticPSR::subdivide() {
    for (int i = 0; i < (1 << dim); ++i) {
        position offset = position::Zero();
        for (int d = 0; d < dim; ++d)
            offset[d] = ((i & (1 << d)) ? 0.5 : -0.5) * halfSize;
        // children[i] = std::make_shared<BarnesHuttStochasticPSR>(center + offset, halfSize / 2,s,beta);
        children[i] = std::make_unique<BarnesHuttStochasticPSR>(center + offset, halfSize / 2,s,beta);
    }
}

void SGP::BarnesHuttStochasticPSR::insertToChildren(const GaussianDipole<dim> &b) {
    int oct = getOctant(b.p);
    if (children[oct])
        children[oct]->insert(b);
    else {
        spdlog::error("child not initialized");
    }
}

/*
 *
    weightedCenter.weight += b.weight;
    weightedCenter.p = (weightedCenter.p * (weightedCenter.weight - b.weight) + b.p * b.weight) / weightedCenter.weight;
    weightedCenter.n += b.n * b.weight;
    weightedMoment += b.p*b.n.transpose() * b.weight;
    weightedCenter.CovMoment += b.CovMoment;
    radius = std::max(radius, (b.p - weightedCenter.p).norm());

scalar SGP::BarnesHuttStochasticPSR::PSRPotential(const Vector<dim> &p, const GaussianDipole &q) const {
    const scalar s2 = s*s;

    scalar r = (p - q.p).norm();
    if (r < 1e-6)
        return 0;

    scalar a = factor * r * std::exp(-r*r/(2*s2));
    scalar b = std::erf(r/(std::sqrt(2)*s));

    scalar dr = (a/s - b)/(4*M_PI*r*r);
    scalar d2r = (2*b - a*(r*r+2*s2)/(s2*s))/(4*M_PI*r*r*r);
    Vector<dim> rhat = (p - q.p).stableNormalized();
    mat H = (d2r - dr/r)*rhat*rhat.transpose() + dr/r*mat::Identity();

//    scalar taylor1 = (H*order1).trace();

    vec k = dr*rhat;
    return k.dot(q.n);// + taylor1;
}

*/
