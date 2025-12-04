// src/OrbifoldTutte.h
#pragma once

#include "Mesh.h"
#include <Eigen/Sparse>
#include <set>

// 支持的 orbifold 类型（这里只实现最简单实用的 TypeIV：4 个 π 锥点）
enum class OrbifoldType {
    TypeIV_4pi = 0  // 4 个 cone angle = π，对应论文的类型 (iv) :contentReference[oaicite:2]{index=2}
};

// 锥点约束：mesh 顶点索引 + 在目标平面上的位置
struct ConeConstraint {
    int vertexIndex;            // 顶点 id（0-based）
    Eigen::Vector2d targetUV;   // 目标平面坐标
};

struct OrbifoldParams {
    OrbifoldType type;
    std::vector<ConeConstraint> cones; // 大小为 3 或 4
};

// 结果：每个顶点的 UV
struct OrbifoldResult {
    std::vector<Eigen::Vector2d> uv; // size = mesh.vertices.size()
};

// 构造 type (iv) 的 basic tile：单位正方形 [0,1]^2，4 个角都是锥点
inline OrbifoldParams makeTypeIVSquareOrbifold(
        int v0, int v1, int v2, int v3)
{
    OrbifoldParams params;
    params.type = OrbifoldType::TypeIV_4pi;
    params.cones.clear();
    params.cones.push_back({v0, Eigen::Vector2d(0.0, 0.0)});
    params.cones.push_back({v1, Eigen::Vector2d(1.0, 0.0)});
    params.cones.push_back({v2, Eigen::Vector2d(1.0, 1.0)});
    params.cones.push_back({v3, Eigen::Vector2d(0.0, 1.0)});
    return params;
}

// 使用 cotan 权重 + 锥点 Dirichlet 约束的“orbifold‑Tutte”参数化（简化版）
inline OrbifoldResult computeOrbifoldTutte(
        const TriangleMesh& mesh,
        const OrbifoldParams& params)
{
    const int n = static_cast<int>(mesh.vertices.size());
    OrbifoldResult result;
    result.uv.assign(n, Eigen::Vector2d::Zero());

    if (n == 0) {
        std::cerr << "Mesh 为空\n";
        return result;
    }

    // 标记锥点
    std::vector<bool> isCone(n, false);
    std::vector<Eigen::Vector2d> coneUV(n, Eigen::Vector2d::Zero());
    for (const auto& c : params.cones) {
        if (c.vertexIndex < 0 || c.vertexIndex >= n) {
            std::cerr << "锥点索引 " << c.vertexIndex << " 越界，忽略\n";
            continue;
        }
        isCone[c.vertexIndex] = true;
        coneUV[c.vertexIndex] = c.targetUV;
    }

    // 收集自由变量（非锥点）
    std::vector<int> freeVertices;
    freeVertices.reserve(n);
    std::vector<int> globalToFreeIndex(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!isCone[i]) {
            globalToFreeIndex[i] = static_cast<int>(freeVertices.size());
            freeVertices.push_back(i);
        }
    }
    const int m = static_cast<int>(freeVertices.size());
    if (m == 0) {
        std::cerr << "所有顶点都是锥点？不合理\n";
        return result;
    }

    using SparseMat = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(m * 10); // 预估

    Eigen::VectorXd rhsU = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd rhsV = Eigen::VectorXd::Zero(m);

    // 遍历每个自由顶点，写出调和方程 ∑ w_ij (phi_i - phi_j) = 0
    for (int fi = 0; fi < m; ++fi) {
        int i = freeVertices[fi];
        const auto& nbrs = mesh.vertexNeighbors[i];
        double diag = 0.0;

        for (int j : nbrs) {
            // 找 cotan 权重
            std::uint64_t key = 0;
            if (i < j) key = (static_cast<std::uint64_t>(i) << 32) | static_cast<std::uint64_t>(j);
            else       key = (static_cast<std::uint64_t>(j) << 32) | static_cast<std::uint64_t>(i);

            auto it = mesh.edgeCotWeights.find(key);
            if (it == mesh.edgeCotWeights.end()) continue;
            double w = it->second;
            if (std::abs(w) < 1e-12) continue;

            diag += w;

            if (!isCone[j]) {
                int fj = globalToFreeIndex[j];
                // L_ij -= w
                triplets.emplace_back(fi, fj, -w);
            } else {
                // 锥点，移到右侧：rhs += w * phi_j
                rhsU[fi] += w * coneUV[j].x();
                rhsV[fi] += w * coneUV[j].y();
            }
        }
        // 对角线：L_ii += diag
        triplets.emplace_back(fi, fi, diag);
    }

    SparseMat A(m, m);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLDLT<SparseMat> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "稀疏矩阵分解失败（可能是网格不连通或权重有问题）\n";
        return result;
    }

    Eigen::VectorXd solU = solver.solve(rhsU);
    Eigen::VectorXd solV = solver.solve(rhsV);

    // 组装最终 UV
    for (int i = 0; i < n; ++i) {
        if (isCone[i]) {
            result.uv[i] = coneUV[i];
        } else {
            int fi = globalToFreeIndex[i];
            result.uv[i].x() = solU[fi];
            result.uv[i].y() = solV[fi];
        }
    }

    // 可选：对整体平移 / 旋转 / 缩放做一个简单归一化，让结果落在大致 [0,1]^2
    // 这里做一个平移使最小坐标到 0，并按最大跨度做缩放。
    Eigen::Vector2d minUV = result.uv[0];
    Eigen::Vector2d maxUV = result.uv[0];
    for (const auto& p : result.uv) {
        minUV = minUV.cwiseMin(p);
        maxUV = maxUV.cwiseMax(p);
    }
    Eigen::Vector2d extent = maxUV - minUV;
    double scale = std::max(extent.x(), extent.y());
    if (scale > 0) {
        for (auto& p : result.uv) {
            p = (p - minUV) / scale;
        }
    }

    return result;
}
