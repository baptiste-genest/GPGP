#ifndef STOCHASTICGEOMETRYPROCESSING_H
#define STOCHASTICGEOMETRYPROCESSING_H

#include <spdlog/spdlog.h>

#include "types.h"
#include "Tensor.h"
#include <iostream>
#include <unordered_set>
#include "Eigen/Geometry"

#include <fstream>

namespace SGP {
// struct OrientedPointCloud {
//     vecs positions,normals;
// };

#ifdef SGP2D
constexpr int dim = 2;
#else
constexpr int dim = 3;
#endif

template<int dim>
using Points = Eigen::Matrix<scalar,dim,-1,Eigen::ColMajor>;

template<int dim>
using Vector = Eigen::Vector<scalar,dim>;

template<int dim>
using SquareMatrix = Eigen::Matrix<scalar,dim,dim>;

template<int rows,int cols>
using Matrix = Eigen::Matrix<scalar,rows,cols>;

using AffineMap = Eigen::Transform<scalar,dim,Eigen::Affine>;
using AffineMap3 = Eigen::Transform<scalar,3,Eigen::Affine>;
using Translation = Eigen::Translation<scalar,dim>;
using Scaling = Eigen::UniformScaling<scalar>;

template<int dim>
void Normalize(Points<dim> &X, Vector<dim> offset = Vector<dim>::Zero(dim), scalar dilat = 1)
{
    if (dim == -1) {
        offset = Vector<dim>::Zero(X.rows());
    }
    Vector<dim> min = X.rowwise().minCoeff();
    Vector<dim> max = X.rowwise().maxCoeff();
    Vector<dim> scale = max - min;
    double f = dilat/scale.maxCoeff();
    Vector<dim> c = (min+max)*0.5;
    X.colwise() -= c;
    X *= f;
    X.colwise() += offset;
}


using TensorGrid = Tensor<dim,SquareMatrix<dim>>;
using VectorGrid = Tensor<dim,Vector<dim>>;
using PointGrid = Tensor<dim, Vector<dim>>;
using ScalarGrid = Tensor<dim,scalar>;
using SliceIndex = VectorGrid::SliceIndex;
using Triplet = Eigen::Triplet<scalar>;
using Triplets = std::vector<Triplet>;

using TensorGrid3 = Tensor<3,SquareMatrix<3>>;
using VectorGrid3 = Tensor<3,Vector<3>>;
using PointGrid3 = Tensor<3, Vector<3>>;
using ScalarGrid3 = Tensor<3,scalar>;
using SliceIndex3 = VectorGrid3::SliceIndex;


using GridElement = Eigen::Vector<int,dim>;
using GridElement3 = Eigen::Vector<int,3>;
using GridElement2 = Eigen::Vector<int,2>;


using PointGrid2 = Tensor<2, Vector<2>>;

template<class in,class out,int dim,class func>
Tensor<dim,out> Apply(const Tensor<dim,in>& G, const func& f) {
    Tensor<dim,out> rslt(G.getSizes());
    auto S = G.getSizes();
#pragma omp parallel for
    for(auto index : range(G.getSize()))
        rslt.at(index) = f(G.at(index));
    return rslt;
}



class StopWatch {
     std::map<std::string,scalar> profiler;
     TimeStamp clock;

public:
     StopWatch() {
         clock = Time::now();
     }

    void start() {
        clock = Time::now();
    }

     void reset() {
        profiler.clear();
    }

     TimeTypeSec tick(std::string label,bool verbose = false) {
        if (profiler.find(label) == profiler.end())
            profiler[label] = 0;
        TimeTypeSec t = TimeFrom(clock);
        profiler[label] += t;
        if (verbose) {
            spdlog::info("tick {} time {}",label,TimeFrom(clock));
        }
        clock = Time::now();
        return t;
    }

     void profile(bool relative = true) {
        std::cout << "         STOPWATCH REPORT            " << std::endl;
        scalar s = 0;
        std::vector<std::pair<std::string,scalar>> stamps;
        for (const auto& [key,value] : profiler){
            s += value;
            stamps.push_back({key,value});
        }
        if (!relative)
            s = 1;
        std::sort(stamps.begin(),stamps.end(),[](std::pair<std::string,scalar> a,std::pair<std::string,scalar> b) {
            return a.second > b.second;
        });
        for (const auto& x : stamps){
            if (relative)
            std::cout << x.first << " : " << 100*x.second/s << "%\n";
            else
                std::cout << x.first << " : " << x.second/s << "\n";
        }
        std::cout << "         END     REPORT            " << std::endl << std::endl;
    }

};



struct GridHash {
    std::size_t operator()(const GridElement& v) const noexcept {
        std::size_t seed = std::hash<Eigen::Index>()(v.size());
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            std::size_t h = std::hash<int>()(v[i]);
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
    /*
    std::size_t operator()(GridElement const& vec) const {
        std::size_t seed = vec.size();
        for(auto x : vec) {
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = (x >> 16) ^ x;
            seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
*/
};

struct GridEqual {
    bool operator()(const GridElement& a, const GridElement& b) const noexcept {
        return a == b;
    }
};

struct GridLess {
    bool operator()(const GridElement& a,
                    const GridElement& b) const {
        if (a.size() != b.size())           // shorter vector first
            return a.size() < b.size();
        for (Eigen::Index i = 0; i < a.size(); ++i) {
            if (a[i] != b[i])               // lexicographic compare
                return a[i] < b[i];
        }
        return false;                       // equal vectors are not less
    }
};

// template<class T>
// using GridMap = std::map<GridElement,T,GridLess>;

template<class T>
using GridMap = std::unordered_map<GridElement,T,GridHash,GridEqual>;

using GridSet = std::unordered_set<GridElement,GridHash,GridEqual>;

template<int D>
class MultivariateGaussian;

using GaussianValueGradient = MultivariateGaussian<dim+1>;

struct GPIS {

    virtual GaussianValueGradient predict(const Vector<dim>& x) const = 0;

};


template<int dim>
SquareMatrix<dim*dim> kron(const SquareMatrix<dim>& A,const SquareMatrix<dim>& B) {
    SquareMatrix<dim*dim> C;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            C.block(i*dim,j*dim,dim,dim) = A(i,j)*B;
    return C;
}

template<int D>
smat SparseDiag(const std::vector<SquareMatrix<D>>& d) {
    int n = d.size();
    Triplets triplets;
    triplets.reserve(n*D*D);
    for (int i = 0; i < n; i++) {
        const auto& M = d[i];
        for (int r = 0; r < D; r++)
            for (int c = 0; c < D; c++)
                triplets.push_back({i*D+r,i*D+c,M(r,c)});
    }
    smat rslt(n*D,n*D);
    rslt.setFromTriplets(triplets.begin(),triplets.end());
    return rslt;
}


inline void saveMatrix(std::string fileName, const Mat&  matrix)
{
    using namespace std;
    using namespace Eigen;
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

    ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}



inline Mat LoadMatrix(std::string fileToOpen)
{
    using namespace std;
    using namespace Eigen;
    vector<scalar> matrixEntries;
    ifstream matrixDataFile(fileToOpen);
    string matrixRowString;
    string matrixEntry;

    int matrixRowNumber = 0;
    while (getline(matrixDataFile, matrixRowString)) {
        stringstream matrixRowStringStream(matrixRowString);
        while (getline(matrixRowStringStream, matrixEntry, ','))
            matrixEntries.push_back(stod(matrixEntry));
        matrixRowNumber++;
    }
    return Map<Mat>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}



}

#endif // STOCHASTICGEOMETRYPROCESSING_H
