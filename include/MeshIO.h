#ifndef MESH_IO_H
#define MESH_IO_H

#include <Eigen/Dense>
#include <string>
#include <vector> // 需要 vector 来声明 cones

// ====================================================================
// Mesh I/O Declarations
// ====================================================================

// 加载函数 (必须声明 load_mesh)
bool load_mesh(
        const std::string& filename,
        Eigen::MatrixXd& V,
        Eigen::MatrixXi& T,
        std::vector<int>& cones
);

// 保存函数
bool write_mesh_obj(
        const std::string& filename,
        const Eigen::MatrixXd& V_flat, // 2D 平展坐标 (V')
        const Eigen::MatrixXi& T_cut   // 切割后的拓扑 (T')
);

#endif // MESH_IO_H