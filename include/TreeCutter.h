// TreeCutter.h
#pragma once

#include <Eigen/Core>
#include <vector>
#include "CutMesh.h"

// 等价于 Matlab TreeCutter，用 cone-tree 把球面切开
class TreeCutter
{
public:
    // V: nV x 3, T: nF x 3（原始网格）
    // treeAdj: k x k 的 0/1 邻接矩阵（Matlab 里的 tree）
    // treeIndices: size = k，每个是 mesh 顶点索引（0-based）
    // root: tree 的根节点（tree 中的索引 0..k-1）
    TreeCutter(const Eigen::MatrixXd& V,
               const Eigen::MatrixXi& T,
               const Eigen::MatrixXi& treeAdj,
               const std::vector<int>& treeIndices,
               int root);

    // 只允许调用一次，执行整棵树的切缝
    void cutTree();

    // 得到切完后的 CutMesh
    CutMesh getCutMesh() const;

private:
    // 当前（会随切缝更新）的网格
    Eigen::MatrixXd V_; // nV x 3
    Eigen::MatrixXi T_; // nF x 3

    // 所有 seam 的对应路径
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2>> pathPairs_;

    // 索引映射
    std::vector<int> cutIndsToUncutInds_;               // size = 当前 nV
    std::vector<std::vector<int>> uncutIndsToCutInds_;  // size = 原始 nV

    // 树结构
    Eigen::MatrixXi treeStructure_;  // k x k，有向邻接矩阵
    std::vector<int> treeIndices_;   // size = k，对应每个 tree-node 的 mesh 顶点
    int treeRoot_;                   // 根节点 id（0..k-1）

    bool alreadyCut_ = false;

    // === 内部方法 ===
    void directTree();                 // BFS 把无向树变成有向树
    void cutTreeRecurse(int rootNode); // 递归切割整棵树

    // 按给定顶点路径切 mesh，返回 m×2 对应矩阵（类似 split_mesh_by_path）
    Eigen::Matrix<int, Eigen::Dynamic, 2>
    splitMeshByPath(const std::vector<int>& path);

    // 以 centerVertex 为“星型中心”复制中心点并更新 pathPairs（类似 splitCenterNode）
    void splitCenterNode(
            int centerVertex,
            std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2>>& starPathPairs);

    // 计算最短路径（避免边界），类似 Matlab 里的 shortestpath + boundary 删除
    std::vector<int> shortestPath(int source, int target) const;

    // 找当前 T_ 的边界顶点
    void findBoundaryVertices(std::vector<char>& isBoundary) const;
};
