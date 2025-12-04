// src/OrbifoldParam.h
#pragma once

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <vector>
#include <unordered_map>
#include <array>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>

using MyMesh = OpenMesh::TriMesh_ArrayKernelT<>;

// ==== 辅助：输出带 UV 的 OBJ ====

// 将 mesh 顶点坐标 + 每个顶点的 uv 写成一个 OBJ
inline bool writeMeshWithUV(const std::string& path,
                            const MyMesh& mesh,
                            const std::vector<Eigen::Vector2d>& uv)
{
    if (uv.size() != mesh.n_vertices()) {
        std::cerr << "writeMeshWithUV: UV 数量与顶点数量不匹配\n";
        return false;
    }

    std::ofstream out(path);
    if (!out) {
        std::cerr << "writeMeshWithUV: 无法打开输出文件: " << path << "\n";
        return false;
    }

    out << "# Orbifold Tutte parameterization result\n";

    // 顶点坐标
    for (MyMesh::VertexIter v_it = mesh.vertices_begin();
         v_it != mesh.vertices_end(); ++v_it)
    {
        MyMesh::Point p = mesh.point(*v_it);
        out << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
    }

    // 顶点 UV
    for (std::size_t i = 0; i < uv.size(); ++i) {
        out << "vt " << uv[i].x() << " " << uv[i].y() << "\n";
    }

    // 面：假设 TriMesh，每个面三个顶点
    for (MyMesh::FaceIter f_it = mesh.faces_begin();
         f_it != mesh.faces_end(); ++f_it)
    {
        std::array<int, 3> idx{};
        int k = 0;
        for (MyMesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(*f_it);
             fv_it.is_valid(); ++fv_it)
        {
            if (k >= 3) break; // TriMesh 正常不会发生
            idx[k++] = fv_it->idx();
        }
        if (k != 3) continue;

        // OBJ 索引从 1 开始，顶点索引和纹理索引一一对应
        out << "f "
            << idx[0] + 1 << "/" << idx[0] + 1 << " "
            << idx[1] + 1 << "/" << idx[1] + 1 << " "
            << idx[2] + 1 << "/" << idx[2] + 1 << "\n";
    }

    return true;
}

// ==== Orbifold 参数化相关类型 ====

// 目前只支持 type (iv)：4 个 π 锥点，对应单位方形四角
enum class OrbifoldType {
    TypeIV_4pi = 0
};

struct OrbifoldParams {
    int v[4];  // 4 个锥点的顶点索引（0-based）
};

inline OrbifoldParams makeTypeIVSquareOrbifold(
        int v0, int v1, int v2, int v3)
{
    OrbifoldParams p;
    p.v[0] = v0;
    p.v[1] = v1;
    p.v[2] = v2;
    p.v[3] = v3;
    return p;
}

struct OrbifoldResult {
    std::vector<Eigen::Vector2d> uv; // 每个顶点的 UV
};
// 写一个“摊平后的” OBJ：把 uv 当作 (x,y)，z=0 写进 v
inline bool writeFlattenedOBJ(
        const std::string& path,
        const MyMesh& mesh,
        const std::vector<Eigen::Vector2d>& uv)
{
    if (uv.size() != mesh.n_vertices()) {
        std::cerr << "[writeFlattenedOBJ] UV 数量和顶点数量不一致\n";
        return false;
    }

    std::ofstream out(path);
    if (!out) {
        std::cerr << "[writeFlattenedOBJ] 无法写入文件: " << path << "\n";
        return false;
    }

    out << "# Flattened mesh by orbifold Tutte\n";

    // 顶点：用 uv 作为 2D 坐标，z=0
    for (size_t i = 0; i < uv.size(); ++i) {
        out << "v " << uv[i].x() << " " << uv[i].y() << " 0\n";
    }

    // 面：用 OpenMesh 的 face -> 顶点索引
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        std::vector<int> faceIdx;
        for (auto fv_it = mesh.cfv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            int idx = fv_it->idx(); // 0-based
            faceIdx.push_back(idx + 1); // OBJ 里是 1-based
        }
        if (faceIdx.size() == 3) {
            out << "f " << faceIdx[0] << " "
                << faceIdx[1] << " "
                << faceIdx[2] << "\n";
        }
        // 如果你的网格里有非三角面，可以在这里加判断
    }

    return true;
}

// ==== 核心：cotan Laplacian + 锥点 Dirichlet 约束 ====

inline OrbifoldResult computeOrbifoldTutte(
        const MyMesh& mesh,
        const OrbifoldParams& params)
{
    const int n = static_cast<int>(mesh.n_vertices());
    OrbifoldResult result;
    result.uv.assign(n, Eigen::Vector2d::Zero());

    if (n == 0) {
        std::cerr << "computeOrbifoldTutte: mesh 为空\n";
        return result;
    }

    // --- 构建一圈邻接 + cotan 权重 ---
    std::vector<std::unordered_map<int, bool>> nbrSet(n);
    std::unordered_map<std::uint64_t, double> edgeCot;

    auto edgeKey = [](int i, int j) -> std::uint64_t {
        if (i > j) std::swap(i, j);
        return (static_cast<std::uint64_t>(i) << 32) |
               static_cast<std::uint64_t>(j);
    };

    auto toEigen = [&](MyMesh::VertexHandle vh) -> Eigen::Vector3d {
        const MyMesh::Point& p = mesh.point(vh);
        return Eigen::Vector3d(
                static_cast<double>(p[0]),
                static_cast<double>(p[1]),
                static_cast<double>(p[2]));
    };

    auto accumulateAngle = [&](int ia, int ib, int ic,
                               const Eigen::Vector3d& pa,
                               const Eigen::Vector3d& pb,
                               const Eigen::Vector3d& pc)
    {
        Eigen::Vector3d u = pb - pa;
        Eigen::Vector3d v = pc - pa;
        // 手写 cross，避免 Eigen cross 的奇怪链接问题
        Eigen::Vector3d cr(
                u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x()
        );
        double area2 = cr.norm();
        if (area2 < 1e-15) return;
        double dot = u.dot(v);
        double cot = dot / area2;
        double w = 0.5 * cot;
        std::uint64_t key = edgeKey(ib, ic);
        edgeCot[key] += w;
    };

    for (MyMesh::FaceIter f_it = mesh.faces_begin();
         f_it != mesh.faces_end(); ++f_it)
    {
        MyMesh::FaceHandle fh = *f_it;
        std::array<MyMesh::VertexHandle, 3> vhs;
        int k = 0;
        for (MyMesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(fh);
             fv_it.is_valid() && k < 3; ++fv_it)
        {
            vhs[k++] = *fv_it;
        }
        if (k != 3) continue;

        int i0 = vhs[0].idx();
        int i1 = vhs[1].idx();
        int i2 = vhs[2].idx();

        Eigen::Vector3d p0 = toEigen(vhs[0]);
        Eigen::Vector3d p1 = toEigen(vhs[1]);
        Eigen::Vector3d p2 = toEigen(vhs[2]);

        // 三个角对应三条对边
        accumulateAngle(i0, i1, i2, p0, p1, p2); // 角在 p0，对边(i1,i2)
        accumulateAngle(i1, i2, i0, p1, p2, p0); // 角在 p1，对边(i2,i0)
        accumulateAngle(i2, i0, i1, p2, p0, p1); // 角在 p2，对边(i0,i1)

        // 邻接
        auto addNbr = [&](int a, int b) {
            if (a == b) return;
            nbrSet[a][b] = true;
            nbrSet[b][a] = true;
        };
        addNbr(i0, i1);
        addNbr(i1, i2);
        addNbr(i2, i0);
    }

    // 邻接列表
    std::vector<std::vector<int>> neighbors(n);
    for (int i = 0; i < n; ++i) {
        neighbors[i].reserve(nbrSet[i].size());
        for (auto& kv : nbrSet[i]) {
            neighbors[i].push_back(kv.first);
        }
    }

    // --- 锥点标记 + 目标位置 ---
    std::vector<bool> isCone(n, false);
    std::vector<Eigen::Vector2d> coneUV(n, Eigen::Vector2d::Zero());

    for (int k = 0; k < 4; ++k) {
        int idx = params.v[k];
        if (idx < 0 || idx >= n) {
            std::cerr << "锥点索引 " << idx << " 越界，忽略\n";
            continue;
        }
        isCone[idx] = true;
        switch (k) {
            case 0: coneUV[idx] = Eigen::Vector2d(0.0, 0.0); break;
            case 1: coneUV[idx] = Eigen::Vector2d(1.0, 0.0); break;
            case 2: coneUV[idx] = Eigen::Vector2d(1.0, 1.0); break;
            case 3: coneUV[idx] = Eigen::Vector2d(0.0, 1.0); break;
        }
    }

    // --- 收集自由变量（非锥点） ---
    std::vector<int> freeVertices;
    freeVertices.reserve(n);
    std::vector<int> globalToFree(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!isCone[i]) {
            globalToFree[i] = static_cast<int>(freeVertices.size());
            freeVertices.push_back(i);
        }
    }
    const int m = static_cast<int>(freeVertices.size());
    if (m == 0) {
        std::cerr << "所有顶点都是锥点？不合理\n";
        return result;
    }

    using SparseMat = Eigen::SparseMatrix<double>;
    using Triplet   = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(m * 10);

    Eigen::VectorXd rhsU = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd rhsV = Eigen::VectorXd::Zero(m);

    // --- 构建线性系统 ---
    for (int fi = 0; fi < m; ++fi) {
        int i = freeVertices[fi];
        const auto& nbrs = neighbors[i];
        double diag = 0.0;

        for (int j : nbrs) {
            std::uint64_t key = edgeKey(i, j);
            auto it = edgeCot.find(key);
            if (it == edgeCot.end()) continue;
            double w = it->second;
            if (std::abs(w) < 1e-12) continue;

            diag += w;

            if (!isCone[j]) {
                int fj = globalToFree[j];
                triplets.emplace_back(fi, fj, -w);
            } else {
                rhsU[fi] += w * coneUV[j].x();
                rhsV[fi] += w * coneUV[j].y();
            }
        }

        // 对角线
        triplets.emplace_back(fi, fi, diag);
    }

    SparseMat A(m, m);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLDLT<SparseMat> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "稀疏矩阵分解失败\n";
        return result;
    }

    Eigen::VectorXd solU = solver.solve(rhsU);
    Eigen::VectorXd solV = solver.solve(rhsV);

    // --- 组装 UV ---
    result.uv.assign(n, Eigen::Vector2d::Zero());
    for (int i = 0; i < n; ++i) {
        if (isCone[i]) {
            result.uv[i] = coneUV[i];
        } else {
            int fi = globalToFree[i];
            result.uv[i].x() = solU[fi];
            result.uv[i].y() = solV[fi];
        }
    }

    // --- 归一化到 [0,1]^2 ---
    Eigen::Vector2d minUV = result.uv[0];
    Eigen::Vector2d maxUV = result.uv[0];
    for (const auto& p : result.uv) {
        minUV = minUV.cwiseMin(p);
        maxUV = maxUV.cwiseMax(p);
    }
    Eigen::Vector2d extent = maxUV - minUV;
    double scale = std::max(extent.x(), extent.y());
    if (scale > 0.0) {
        for (auto& p : result.uv) {
            p = (p - minUV) / scale;
        }
    }

    return result;

}
