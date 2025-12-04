#ifndef MESH_UTILS_H
#define MESH_UTILS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <set> // 引入 set 用于 find_boundary_vertices

// 实用函数
double edge_length(const Eigen::MatrixXd& V, int i, int j);
std::vector<int> get_vertex_neighbors(const Eigen::MatrixXi& T, int vertex_index);

// 核心算法函数
std::vector<int> find_shortest_path(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        int start_v,
        int end_v
);

// 几何工具函数 (新增声明)
std::set<int> find_boundary_vertices( // <-- 🌟 新增声明 🌟
        const Eigen::MatrixXi& T,
        int num_vertices
);

void compute_cotangent_laplacian(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        Eigen::SparseMatrix<double>& L
);

#endif // MESH_UTILS_H