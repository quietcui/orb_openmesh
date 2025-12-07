#pragma once

#include "MeshTypes.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

// ---------- 工具函数：OpenMesh 点 -> Eigen 向量 ----------

inline Eigen::Vector3d to_eigen(const MyMesh::Point& p)
{
    return Eigen::Vector3d(
            static_cast<double>(p[0]),
            static_cast<double>(p[1]),
            static_cast<double>(p[2]));
}

// ---------- 构造 cotan Laplacian L（n x n 稀疏矩阵） ----------

void compute_cotangent_laplacian(const MyMesh& mesh,
                                 Eigen::SparseMatrix<double>& L);

// ---------- 收集一圈边界顶点（按顺时针 / 逆时针顺序） ----------

void collect_boundary_loop(const MyMesh& mesh,
                           std::vector<MyMesh::VertexHandle>& boundary);
