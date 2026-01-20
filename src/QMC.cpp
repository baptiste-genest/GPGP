#include "QMC.h"

namespace SGP::QMC3D {

// 3*kNumPoints doubles, column-major as x0,y0,z0, x1,y1,z1, ...
alignas(64) static const double kData[3 * kNumPoints] = {
#include "../data/QMC/QMC3D.inc"
};

const double* data_ptr() { return kData; }

}


namespace SGP::QMC2D {

// 3*kNumPoints doubles, column-major as x0,y0,z0, x1,y1,z1, ...
alignas(64) static const double kData[2 * kNumPoints] = {
#include "../data/QMC/QMC2D.inc"
};

const double* data_ptr() { return kData; }

}
