#ifndef FLATTENING_H
#define FLATTENING_H

#include "MeshTypes.h"
#include <vector>
#include <Eigen/Dense>

// 使用 OpenMesh 网格和顶点句柄
Eigen::MatrixXd flatten_sphere_openmesh(
        MyMesh& mesh,
        const std::vector<int>& cones_indices
);

#endif // FLATTENING_H