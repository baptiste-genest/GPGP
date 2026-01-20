#include "StochasticBarnesHutt.h"
#include "StochasticPoissonSurfaceReconstruction.h"

bool SGP::StochasticBarnesHutt::StochasticBarnesHuttNode::contains(const position &position) const {
    for (int i = 0; i < dim; ++i)
        if (std::abs(position[i] - center[i]) > halfSize)
            return false;
    return true;
}

int SGP::StochasticBarnesHutt::StochasticBarnesHuttNode::getOctant(const position &position) const noexcept {
    if (!contains(position))
        spdlog::error("octant query on point outside of range");
    int index = 0;
    for (int i = 0; i < dim; ++i)
        if (position[i] >= center[i])
            index |= (1 << i);
    return index;
}

/*
void SGP::StochasticBarnesHutt::subdivide(StochasticBarnesHuttNode& n) {
    for (int i = 0; i < (1 << dim); ++i) {
        position offset = position::Zero();
        for (int d = 0; d < dim; ++d)
            offset[d] = ((i & (1 << d)) ? 0.5 : -0.5) * n.halfSize;
        auto child_key = n.getChildKey(i);
        if (nodes.contains(child_key)) {
            spdlog::info("conflict {}",n.key);
        }
        nodes[child_key] = StochasticBarnesHuttNode(child_key,n.center + offset, n.halfSize / 2);
    }
}
*/

void SGP::StochasticBarnesHutt::insertToChildren(HashKey key, const GaussianDipole<dim> &b) {
    StochasticBarnesHuttNode& node = nodes[key];
    HashKey oct = node.getOctant(b.p);
    auto child_id = node.getChildKey(oct);
    // auto child_it = nodes.find(child_id);
    // if (child_it != nodes.end())
    if (node.isChildActive(oct))
        insertAtNode(child_id,b);
    else {
        node.setChildActive(oct);
        position offset = position::Zero();
        for (int d = 0; d < dim; ++d)
            offset[d] = ((oct & (1 << d)) ? 0.5 : -0.5) * node.halfSize;
        nodes[child_id] = StochasticBarnesHuttNode(child_id,node.center + offset, node.halfSize / 2);
        insertAtNode(child_id,b);
        // spdlog::error("could not find child");
    }
}

void SGP::StochasticBarnesHutt::insertAtNode(HashKey key, const GaussianDipole<dim> &b)
{
    StochasticBarnesHuttNode& node = nodes[key];
    if (!node.contains(b.p)) {
        spdlog::error("did not find point in zone");
        return;
    }

    if (node.isLeaf && !node.point.has_value()) {
        node.point = b;
        node.weightedCenter = b;
        node.weightedCenter.p *= b.n.norm();
        node.weight += b.n.norm();
        return;
    }

    if (node.isLeaf) {
        auto old_point = node.point.value();
        node.point = {};
        node.isLeaf = false;
        insertToChildren(key,old_point);
    }
    insertToChildren(key,b);
    node = nodes[key];
    node.nb_below++;

    // Update center of weight
    node.weight += b.n.norm();
    node.weightedCenter.p += b.p*b.n.norm();
    node.weightedCenter.n += b.n;
    // node.weightedCenter.CovMoment += b.CovMoment;
    // node.weightedCenter.CovPos += b.CovPos;
    node.weightedCenter.FullCov += b.FullCov;
}

void SGP::StochasticBarnesHutt::insert(const GaussianDipole<dim> &b) {
    insertAtNode(1,b);
}

void SGP::StochasticBarnesHutt::inner_computeMaxRadius(HashKey key, std::vector<Vector<dim> > &points_below) {
    StochasticBarnesHuttNode& node = nodes[key];
    if (node.point) {
        points_below.push_back(node.point.value().p);
        return;
    }
    std::vector<Vector<dim>> buffer;

    for (int i = 0; i < (1 << dim); ++i) {
        auto id = node.getChildKey(i);
        if (node.isChildActive(i)) inner_computeMaxRadius(id,buffer);

    }
    // compute max radius from points_below
    // scalar old_radius = radius;
    node.radius = 0;
    if (!buffer.empty()) {
        Vector<dim> max_point = buffer[0];
        for (const auto& p : buffer) {
            if ((p - node.weightedCenter.p).norm() > (max_point - node.weightedCenter.p).norm()) {
                max_point = p;
            }
        }
        node.radius = std::max(node.radius, (max_point - node.weightedCenter.p).norm());
        // spdlog::info("old radius {} new radius {}",old_radius,radius);
    }
    points_below.insert(points_below.end(), buffer.begin(), buffer.end());
}


void SGP::StochasticBarnesHutt::computeRadius() {
    std::vector<Vector<dim>> buffer;
    buffer.reserve(nodes[1].nb_below);
    inner_computeMaxRadius(1,buffer);
}

SGP::StochasticBarnesHutt::PSRKernels SGP::StochasticBarnesHutt::computeKernels(const Vector<dim> &x, const GaussianDipole<dim> &p) const {
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

    Kn.block(0,0,1,dim) = k.transpose();
    Kn.block(1,0,dim,dim) = H;

    FK.block(0,0,dim+1,dim) = Kp;
    FK.block(0,dim,dim+1,dim) = Kn;

    return {Kn,Kp,FK};
}

void SGP::StochasticBarnesHutt::compute(HashKey key, const position &p, GaussianValueGradient &rslt) const {
    const StochasticBarnesHuttNode& node = nodes.at(key);
    scalar dist = (p - node.weightedCenter.p).norm();

    if (node.point || dist > beta*node.radius) {
        auto KF = ComputeJointPoissonKernel(p,node.weightedCenter.p,node.weightedCenter.n,s);
        // auto KF = computeKernels(p,node.weightedCenter).KF;

        Vector<2*dim> X = Vector<2*dim>::Zero();
        X.tail(dim) = node.weightedCenter.n;
        Vector<dim+1> mean = KF*X;
        SquareMatrix<dim+1> Cov = KF*node.weightedCenter.FullCov*KF.transpose();

        MultivariateGaussian<dim+1> P = MultivariateGaussian<dim+1>(
            mean,
            Cov
            );
        rslt += P;
    } else {
        for (int i = 0; i < (1 << dim); ++i){
            auto child_id = node.getChildKey(i);
            // auto child_it = nodes.find(child_id);
            // if (child_it != nodes.end()) compute(child_id,p, rslt);
            if (node.isChildActive(i)) compute(child_id,p, rslt);
        }
    }
}

void SGP::StochasticBarnesHutt::sanityCheck()
{
    for (const auto& [k,node] : nodes) {
        if (k != node.key) {
            spdlog::error("inconsistent id {} {}",k,node.key);
        }
        spdlog::info("nb_below {} weight {} radius {}",node.nb_below,node.weight,node.radius);
    }
}

void SGP::StochasticBarnesHutt::precomputeMoments() {
    for (auto& [k,node] : nodes) {
        if (node.weight == 0) continue;
        node.weightedCenter.p /= node.weight;
    }
}

SGP::GaussianValueGradient SGP::StochasticBarnesHutt::predict(const position &b) const {
    GaussianValueGradient rslt;
    compute(1,b, rslt);
    // rslt.prediction.addNoise(0.01);
    return rslt;
}

SGP::StochasticBarnesHutt::HashKey SGP::StochasticBarnesHutt::StochasticBarnesHuttNode::getChildKey(HashKey oct) const
{
    HashKey child_key = key;
    child_key <<= dim;
    child_key |= oct;
    return child_key;
}
