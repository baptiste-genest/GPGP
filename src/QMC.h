#ifndef QMC_H
#define QMC_H

#include "StochasticGeometryProcessing.h"

// points.hpp
#pragma once

namespace SGP {

namespace QMC3D {

inline constexpr std::size_t kNumPoints = 8192;


using Mat3XdConstMap = Eigen::Map<const Points<3>>;

const double* data_ptr();

inline constexpr std::size_t size() { return kNumPoints; }

inline Mat3XdConstMap firstN(std::size_t n) {
    if (n > kNumPoints) {
        throw std::out_of_range("QMC3D::firstN: n exceeds number of precomputed points");
    }
    return Mat3XdConstMap(data_ptr(), 3, static_cast<Eigen::Index>(n));
}


} // namespace Points

namespace QMC2D {

inline constexpr std::size_t kNumPoints = 8192;


using Mat2XdConstMap = Eigen::Map<const Points<2>>;

const double* data_ptr();

inline constexpr std::size_t size() { return kNumPoints; }

inline Mat2XdConstMap firstN(std::size_t n) {
    if (n > kNumPoints) {
        throw std::out_of_range("QMC2D::firstN: n exceeds number of precomputed points");
    }
    return Mat2XdConstMap(data_ptr(), 2, static_cast<Eigen::Index>(n));
}


} // namespace Points

}


#endif // QMC_H
