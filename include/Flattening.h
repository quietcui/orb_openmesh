#ifndef FLATTENING_H
#define FLATTENING_H

#include <Eigen/Dense>
#include <vector>
#include "CutMesh.h"

Eigen::MatrixXd flatten_sphere(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        const std::vector<int>& cones,
        int orbifold_type,
        CutMesh& cutMesh
);

CutMesh generate_cut_mesh(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        const std::vector<std::vector<int>>& paths
);

#endif // FLATTENING_H