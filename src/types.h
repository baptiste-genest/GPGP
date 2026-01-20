#pragma once
#include <numeric>
#include <vector>

#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ranges>

namespace SGP {

namespace fs = std::filesystem;

using scalar = double;
// using scalar = float;
using scalars = std::vector<scalar>;

using vec = Eigen::Vector3<scalar>;
using vec2 = Eigen::Vector2<scalar>;
using mat2 = Eigen::Matrix2<scalar>;
using mat = Eigen::Matrix3<scalar>;
using mat4 = Eigen::Matrix4<scalar>;
using vec4 = Eigen::Vector4<scalar>;

using Mat = Eigen::Matrix<scalar,-1,-1,Eigen::ColMajor>;
using Diag = Eigen::DiagonalMatrix<scalar,-1>;
using vecs = std::vector<vec>;
using vec2s = std::vector<vec2>;

using triplet = Eigen::Triplet<scalar>;

using ints = std::vector<int>;
using Vec = Eigen::Vector<scalar,-1>;
using Vecs = std::vector<Vec>;

using smat = Eigen::SparseMatrix<scalar>;

using Index = int;
using grid_Index = std::pair<Index,Index>;


using SPDSolver = Eigen::ConjugateGradient<smat, Eigen::Lower|Eigen::Upper>;



inline auto range(int i) {
    return std::views::iota(0,i);
}
inline auto range(int a,int b) {
    return std::views::iota(a,b);
}
inline std::array<int,2> signs() {
    return {-1,1};
}


inline ints rangeVec(int a,int b) {
    ints rslt(b-a);
    std::iota(rslt.begin(),rslt.end(),a);
    return rslt;
}

inline ints rangeVec(int i) {
    return rangeVec(0,i);
}


template<class T>
using twins = std::pair<T,T>;

inline smat Identity(int V) {
    smat I(V,V);
    I.setIdentity();
    return I;
   }

using Time = std::chrono::high_resolution_clock;
using TimeStamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using TimeTypeSec = float;
using DurationSec = std::chrono::duration<TimeTypeSec>;

inline TimeTypeSec TimeBetween(const TimeStamp& A,const TimeStamp& B){
    return DurationSec(B-A).count();
}

inline TimeTypeSec TimeFrom(const TimeStamp& A){
    return DurationSec(Time::now()-A).count();
}


template<class T>
bool Smin(T& a,T b) {
    if (b < a){
        a = b;
        return true;
    }
    return false;
}

template<class T>
bool Smax(T& a,T b) {
    if (a < b){
        a = b;
        return true;
    }
    return false;
}



}
