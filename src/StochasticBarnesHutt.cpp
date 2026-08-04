#include "StochasticBarnesHutt.h"

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

void SGP::StochasticBarnesHutt::insertToChildren(HashKey key, const GaussianDipole<dim> &b) {
    StochasticBarnesHuttNode& node = nodes[key];
    HashKey oct = node.getOctant(b.p);
    auto child_id = node.getChildKey(oct);
    if (node.isChildActive(oct))
        insertAtNode(child_id,b);
    else {
        node.setChildActive(oct);
        position offset = position::Zero();
        for (int d = 0; d < dim; ++d)
            offset[d] = ((oct & (1 << d)) ? 0.5 : -0.5) * node.halfSize;
        nodes[child_id] = StochasticBarnesHuttNode(child_id,node.center + offset, node.halfSize / 2);
        insertAtNode(child_id,b);
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
        node.addDipole(b);
        return;
    }

    if (node.isLeaf) {
        auto old_point = node.point.value();
        node.point = {};
        node.isLeaf = false;
        insertToChildren(key,old_point);
    }
    insertToChildren(key,b);
    node.nb_below++;

    node.addDipole(b);
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
    node.radius = 0;
    if (!buffer.empty()) {
        Vector<dim> max_point = buffer[0];
        for (const auto& p : buffer) {
            if ((p - node.weightedCenter.p).norm() > (max_point - node.weightedCenter.p).norm()) {
                max_point = p;
            }
        }
        node.radius = std::max(node.radius, (max_point - node.weightedCenter.p).norm());
    }
    points_below.insert(points_below.end(), buffer.begin(), buffer.end());
}


void SGP::StochasticBarnesHutt::computeRadius() {
    std::vector<Vector<dim>> buffer;
    buffer.reserve(nodes[1].nb_below);
    inner_computeMaxRadius(1,buffer);
}

void SGP::StochasticBarnesHutt::compute(HashKey key, const position &p, GaussianValueGradient &rslt) const {
    const StochasticBarnesHuttNode& node = nodes.at(key);
    scalar dist = (p - node.weightedCenter.p).norm();

    if (node.point || dist > beta*node.radius) {
        auto terms = ComputeJointKernelTerms<dim>(p,node.weightedCenter.p,s);

        Vector<dim+1> mean = terms.Kn()*node.weightedCenter.n;
        SquareMatrix<dim+1> Cov = node.assembleCovariance(terms);

        MultivariateGaussian<dim+1> P = MultivariateGaussian<dim+1>(
            mean,
            Cov
            );
        rslt += P;
    } else {
        for (int i = 0; i < (1 << dim); ++i){
            auto child_id = node.getChildKey(i);
            if (node.isChildActive(i)) compute(child_id,p, rslt);
        }
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
    return rslt;
}

SGP::StochasticBarnesHutt::HashKey SGP::StochasticBarnesHutt::StochasticBarnesHuttNode::getChildKey(HashKey oct) const
{
    HashKey child_key = key;
    child_key <<= dim;
    child_key |= oct;
    return child_key;
}
