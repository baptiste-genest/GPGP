#include "SellingsAlgorithm.h"

std::pair<int, int> SGP::Sellings3D::completeBasis(int i, int j) {
    if (i == 0 && j == 1)
        return {2,3};
    else if (i == 0 && j == 2)
        return {1,3};
    else if (i == 0 && j == 3)
        return {1,2};
    else if (i == 1 && j == 2)
        return {0,3};
    else if (i == 1 && j == 3)
        return {0,2};
    return {0,1};
}

void SGP::Sellings3D::SellingsFlip(SuperBase<3> &B, int i, int j) {
    // i, j distinct \in {1,2,3,4}
    // find k,l to complete the set
    auto [k,l] = completeBasis(i,j);
    const GridElement3 ei = B.col(i);
    const GridElement3 ej = B.col(j);
    const GridElement3 ek = B.col(k);
    const GridElement3 el = B.col(l);
    B << -ei,ej,ek+ei,el+ei;
}

std::pair<int, int> SGP::Sellings3D::criterion(const mat &D, const SuperBase<3> &base) {
    for (int j = 1;j<=3;j++) {
        for (int i = 0;i<j;i++) {
            vec ei = base.col(i).cast<scalar>();
            vec ej = base.col(j).cast<scalar>();
            if (ei.dot(D*ej) > 0)
                return {i,j};
        }
    }
    return {-1,-1};
}

bool SGP::Sellings3D::CheckSuperBase(const SuperBase<3> &B) {
    mat S = B.block(0,0,3,3).cast<scalar>();
    scalar D = std::abs(S.determinant());
    if (std::abs(1 - D) > 1e-6)
        return false;
    vec s = B.rowwise().sum().cast<scalar>();
    if (s.norm() > 1e-6)
        return false;
    return true;
}

SGP::SuperBase<3> SGP::Sellings3D::Init() {
    SuperBase<3> B;
    B.block(0,0,3,3) = Eigen::Matrix<int,3,3>::Identity();
    B.col(3) = -GridElement3::Ones();
    return B;
}

SGP::SuperBase<3> SGP::Sellings3D::SellingsAlgorithm(const mat &D) {
    SuperBase<3> B = Init();
    int max_iter = 1000;
    for (auto iter = 0;iter<max_iter;iter++){
        auto [i,j] = criterion(D,B);
        if (i == -1)
            return B;
        SellingsFlip(B,i,j);
    }
    spdlog::warn("Selling's algorithm did not converge after max iter");
    return B;
}

SGP::VoronoiReduction<3> SGP::Sellings3D::ComputeReductionFromSuperBase(const mat &D, const SuperBase<3> &B) {
    VoronoiReduction<3> rslt;

    mat Approx = mat::Zero();
    for (int j = 1;j<=3;j++) {
        for (int i = 0;i<j;i++) {
            auto [k,l] = completeBasis(i,j);
            vec ei = B.col(i).cast<scalar>();
            vec ej = B.col(j).cast<scalar>();
            GridElement3 vij = B.col(k).cross(B.col(l));
            scalar c = - ei.dot(D*ej);
            rslt.elements.push_back({vij,c});
            Approx += c * vij.cast<scalar>() * vij.transpose().cast<scalar>();
        }
    }
    if ((Approx - D).norm() > 1e-6)
        spdlog::warn("Approximation error: {}", (Approx - D).norm());
    return rslt;
}

SGP::VoronoiReduction<3> SGP::Sellings3D::compute(const mat &D) {
    return ComputeReductionFromSuperBase(D, SellingsAlgorithm(D));
}

void SGP::Sellings2D::SellingsFlip(SuperBase<2> &B, int i, int j)
{
    const GridElement2 ei = B.col(i);
    const GridElement2 ej = B.col(j);
    B << -ei,ej,ei-ej;
}

std::pair<int, int> SGP::Sellings2D::criterion(const mat2 &D, const SuperBase<2> &base)
{
    for (int j = 1;j<=2;j++) {
        for (int i = 0;i<j;i++) {
            vec2 ei = base.col(i).cast<scalar>();
            vec2 ej = base.col(j).cast<scalar>();
            if (ei.dot(D*ej) > 0)
                return {i,j};
        }
    }
    return {-1,-1};
}

bool SGP::Sellings2D::CheckSuperBase(const SuperBase<2> &B)
{
    mat2 S = B.block(0,0,2,2).cast<scalar>();
    scalar D = std::abs(S.determinant());
    if (std::abs(1 - D) > 1e-6)
        return false;
    vec2 s = B.rowwise().sum().cast<scalar>();
    if (s.norm() > 1e-6)
        return false;
    return true;
}

SGP::SuperBase<2> SGP::Sellings2D::Init()
{
    SuperBase<2> B;
    B.block(0,0,2,2) = Eigen::Matrix<int,2,2>::Identity();
    B.col(2) = -GridElement2::Ones();
    return B;
}

SGP::SuperBase<2> SGP::Sellings2D::SellingsAlgorithm(const mat2 &D)
{
    SuperBase<2> B = Init();
    int max_iter = 1000;
    for (auto iter = 0;iter<max_iter;iter++){
        auto [i,j] = criterion(D,B);
        if (i == -1)
            return B;
        SellingsFlip(B,i,j);
        // if (!CheckSuperBase(B))
        //     throw std::runtime_error("SellingsFlip produced an invalid superbase");
    }
    spdlog::warn("Selling's algorithm did not converge after max iter");
    return B;
}

SGP::VoronoiReduction<2> SGP::Sellings2D::ComputeReductionFromSuperBase(const mat2 &D, const SuperBase<2> &B)
{
    // std::cout << "Input:\n" << D << std::endl;
    // std::cout << "det : " << D.cast<scalar>().determinant() << std::endl;
    // std::cout << "Base:\n" << B << std::endl;
    VoronoiReduction<2> rslt;
    mat2 Approx = mat2::Zero();
    for (int j = 1;j<=2;j++) {
        for (int i = 0;i<j;i++) {
            Index k = completeBasis(i,j);
            // std::cout << "i,j,k: " << i << "," << j << "," << k << std::endl;
            GridElement2 vij;
            vij << B.col(k)(1),-B.col(k)(0);
            // std::cout << vij << std::endl;
            vec2 ei = B.col(i).cast<scalar>();
            vec2 ej = B.col(j).cast<scalar>();
            scalar c = - ei.dot(D*ej);
            // std::cout << c << std::endl;
            if (c < 1e-8)
                continue;
            rslt.elements.push_back({vij,c});
            Approx += c * vij.cast<scalar>() * vij.transpose().cast<scalar>();
        }
    }
    // std::cout << "Approx:\n" << Approx << std::endl;
    if ((Approx - D).norm() > 1e-6)
        spdlog::warn("Approximation error: {}", (Approx - D).norm());
    return rslt;
}

SGP::VoronoiReduction<2> SGP::Sellings2D::compute(const mat2 &D)
{
    return ComputeReductionFromSuperBase(D, SellingsAlgorithm(D));
}

int SGP::Sellings2D::completeBasis(int i, int j)
{
    return 3 - i - j;
}
