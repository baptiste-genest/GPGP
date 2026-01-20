#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include "StochasticGeometryProcessing.h"
#include "nanoflann.hpp"
#include <fstream>

namespace SGP {

using namespace nanoflann;
template<int dim>
struct OrientedWeightedPointCloud {
    Points<dim> positions,normals;
    Vec weights;

};

template<int D>
inline Points<D> ReadPointCloud(std::filesystem::path path) {
    std::ifstream infile(path);

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << path.filename() << std::endl;
        return {};
    }

    std::vector<double> data; // Store all values in a single contiguous array
    int rows = 0;
    std::string line;
    int dim = D;

    // First pass: Read the file and store numbers in a vector
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double num;
        int current_cols = 0;

        while (iss >> num) {
            data.push_back(num);
            ++current_cols;
        }


        if (dim == -1)
            dim = current_cols;
        if (current_cols != dim) {
            std::cerr << "Error: Wrong dimension when loading point cloud.\n";
            std::cerr << line << std::endl;
            return {};
        }
        ++rows;
    }

    // Second pass: Copy the data into an Eigen matrix
    // where each col is a point
    Points<D> pointCloud(dim, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < dim; ++j) {
            pointCloud(j, i) = data[i * dim + j];
        }
    }
    return pointCloud;
}


template<int dim>
struct PointCloudQuery {

    Points<dim> positions;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return positions.cols();
    }

    // Returns the 'dim'-th component of the 'index'-th point
    inline double kdtree_get_pt(const size_t idx, const size_t i) const {
        return positions(i, idx);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const {
        if (positions.cols() == 0) return false;

        Vector<dim> min_vals = positions.rowwise().minCoeff();
        Vector<dim> max_vals = positions.rowwise().maxCoeff();

        for (int d = 0; d < dim; ++d) {
            bb[d].low = min_vals[d];
            bb[d].high = max_vals[d];
        }

        return true;
    }

    using KDTree = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<scalar, PointCloudQuery<dim>>,
        PointCloudQuery<dim>,
        dim  // dimensionality
        >;

    KDTree kdtree;

    ints radiusQuery(const vec& x,scalar r) const {
        std::vector<std::pair<size_t, scalar>> ret_matches;

        nanoflann::SearchParameters params;
        params.sorted = false;
        const size_t n_matches = kdtree.radiusSearch(
            &x[0],  // raw pointer to query point
            r*r,
            ret_matches,
            params
            );
        ints indices(n_matches);
        for (size_t i = 0; i < n_matches; i++)
            indices[i] = ret_matches[i].first;
        return indices;
    }


    // Query the nearest neighbor of a 3D point
    std::pair<size_t, scalar> queryNN(const Eigen::Vector3d& query_pt) const {
        size_t ret_index;
        scalar out_dist_sqr;

        nanoflann::KNNResultSet<scalar> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        kdtree.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters());

        return {ret_index, out_dist_sqr};
    }

    PointCloudQuery() :
        kdtree(dim, *this, KDTreeSingleIndexAdaptorParams(10)) {}

    PointCloudQuery(const Points<dim>& positions,int max_leaf = 10)
        : positions(positions),
          kdtree(dim, *this, KDTreeSingleIndexAdaptorParams(max_leaf /* max leaf */)) {
        kdtree.buildIndex();
    }

};



}

#endif // POINTCLOUD_H
