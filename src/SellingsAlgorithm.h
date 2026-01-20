#ifndef SELLINGSALGORITHM_H
#define SELLINGSALGORITHM_H

#include "StochasticGeometryProcessing.h"

namespace SGP {


template<int D>
struct VoronoiReduction {
    std::vector<std::pair<Eigen::Vector<int,D>,scalar>> elements;
};

template<int D>
using SuperBase = Eigen::Matrix<int,D,D+1>;

struct Sellings2D {

private:

    static int completeBasis(int i,int j);

    static void SellingsFlip(SuperBase<2>& B,int i,int j);

    static std::pair<int,int> criterion(const mat2& D,const SuperBase<2>& base);

    static bool CheckSuperBase(const SuperBase<2>& B);

    static SuperBase<2> Init();

    static SuperBase<2> SellingsAlgorithm(const mat2& D);

    static VoronoiReduction<2> ComputeReductionFromSuperBase(const mat2& D,const SuperBase<2>& B);

public:

    static VoronoiReduction<2> compute(const mat2& D);

};


struct Sellings3D {
private:

    static std::pair<int,int> completeBasis(int i,int j);

    static void SellingsFlip(SuperBase<3>& B,int i,int j);

    static std::pair<int,int> criterion(const mat& D,const SuperBase<3>& base);

    static bool CheckSuperBase(const SuperBase<3>& B);

    static SuperBase<3> Init();

    static SuperBase<3> SellingsAlgorithm(const mat& D);

    static VoronoiReduction<3> ComputeReductionFromSuperBase(const mat& D,const SuperBase<3>& B);

public:

    static VoronoiReduction<3> compute(const mat& D);

};


template<int dim>
inline VoronoiReduction<dim> ComputeVoronoiReduction(const SquareMatrix<dim>& D) {
    throw std::runtime_error("This function is not implemented for the given dimension.");
}

template<>
inline VoronoiReduction<3> ComputeVoronoiReduction<3>(const mat& D) {
    return Sellings3D::compute(D);
}

template<>
inline VoronoiReduction<2> ComputeVoronoiReduction<2>(const mat2& D) {
    return Sellings2D::compute(D);
}

}


#endif // SELLINGSALGORITHM_H
