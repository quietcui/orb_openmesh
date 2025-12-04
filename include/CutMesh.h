#ifndef CUTMESH_H
#define CUTMESH_H

#include <Eigen/Dense>
#include <vector>
#include <utility>

class CutMesh {
public:
    // V': 切割后的顶点, T': 切割后的三角形
    Eigen::MatrixXd V;
    Eigen::MatrixXi T;

    // 映射信息
    std::vector<std::vector<std::pair<int, int>>> pathPairs;     // 切割路径两侧的索引对
    std::vector<int> cutIndsToUncutInds;                     // 新索引 -> 旧索引
    std::vector<std::vector<int>> uncutIndsToCutInds;         // 旧索引 -> [新索引列表]

    CutMesh(
            const Eigen::MatrixXd& V_in,
            const Eigen::MatrixXi& T_in,
            const std::vector<std::vector<std::pair<int, int>>>& pathPairs_in,
            const std::vector<int>& cutIndsToUncutInds_in,
            const std::vector<std::vector<int>>& uncutIndsToCutInds_in
    );
    CutMesh() = default;
};

#endif // CUTMESH_H#ifndef CUTMESH_H
