// CutMesh.h
#pragma once

#include <Eigen/Core>
#include <vector>

// 等价于 Matlab 的 CutMesh 类：表示“已经切开”的网格
struct CutMesh
{
    // 顶点 (nV x 3) 和三角形 (nF x 3)
    Eigen::MatrixXd V;   // double, nV x 3
    Eigen::MatrixXi T;   // int,    nF x 3

    // 每条 seam 的两侧路径：
    // Matlab: pathPairs{i} 是 m×2 矩阵，每行 [v_left, v_right]
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2>> pathPairs;

    // cutIndsToUncutInds[i]  = 原网格里对应的顶点索引（0-based）
    // uncutIndsToCutInds[j]  = 原网格顶点 j 在切后网格中的所有副本索引列表
    std::vector<int> cutIndsToUncutInds;               // size = nV_cut
    std::vector<std::vector<int>> uncutIndsToCutInds;  // size = nV_orig

    CutMesh() = default;

    CutMesh(const Eigen::MatrixXd& V_,
            const Eigen::MatrixXi& T_,
            const std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2>>& pathPairs_,
            const std::vector<int>& cut2uncut,
            const std::vector<std::vector<int>>& uncut2cut)
            : V(V_)
            , T(T_)
            , pathPairs(pathPairs_)
            , cutIndsToUncutInds(cut2uncut)
            , uncutIndsToCutInds(uncut2cut)
    {}
};
