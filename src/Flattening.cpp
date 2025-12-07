// Flattening.cpp
#include "Flattening.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/IO/Options.hh>


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <stdexcept>
#include <cmath>

// =======================
// OpenMesh <-> Eigen
// =======================

static void mesh_to_eigen(
        const MyMesh& mesh,
        Eigen::MatrixXd& V,
        Eigen::MatrixXi& F)
{
    const int nV = static_cast<int>(mesh.n_vertices());
    const int nF = static_cast<int>(mesh.n_faces());

    V.resize(nV, 3);
    F.resize(nF, 3);

    // 顶点
    int idx = 0;
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it, ++idx) {
        auto p = mesh.point(*v_it);
        V(idx, 0) = p[0];
        V(idx, 1) = p[1];
        V(idx, 2) = p[2];
    }

    // 面
    idx = 0;
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it, ++idx) {
        int k = 0;
        for (auto fv_it = mesh.cfv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            F(idx, k++) = fv_it->idx();       // 顶点索引 0-based
            if (k >= 3) break;
        }
    }
}

static void eigen_to_mesh_flat(
        MyMesh& mesh,
        const Eigen::MatrixXd& flat_V,
        const CutMesh& cutMesh)
{
    // 注意：OpenMesh 中只有“原始”顶点数量，cutMesh 可能有复制的顶点。
    // 这里我们做一个简单的策略：
    //   对于每个原始顶点，只取 cutMesh.uncutIndsToCutInds[v] 的第一个副本的平面坐标。
    // 这样无法显式表现切缝，但可以看到整体展平的形状。
    int nOrigV = static_cast<int>(mesh.n_vertices());
    if (static_cast<int>(cutMesh.uncutIndsToCutInds.size()) != nOrigV) {
        std::cerr << "[eigen_to_mesh_flat] Warning: uncutIndsToCutInds size != n_orig_vertices\n";
    }

    int idx = 0;
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it, ++idx) {
        if (idx >= static_cast<int>(cutMesh.uncutIndsToCutInds.size()))
            break;

        const auto& copies = cutMesh.uncutIndsToCutInds[idx];
        if (copies.empty()) continue;
        int cutIdx = copies[0]; // 取第一个副本

        if (cutIdx < 0 || cutIdx >= static_cast<int>(flat_V.rows()))
            continue;

        double x = flat_V(cutIdx, 0);
        double y = flat_V(cutIdx, 1);

        mesh.set_point(*v_it, MyMesh::Point(static_cast<float>(x),
                                            static_cast<float>(y),
                                            0.0f));
    }
}

// =======================
// 辅助：构建 cotangent Laplacian（标量）
// =======================

static void build_cotangent_laplacian(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        Eigen::SparseMatrix<double>& L)
{
    const int nV = static_cast<int>(V.rows());
    const int nF = static_cast<int>(F.rows());

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nF * 9);

    // 先构建权重矩阵 W（对称），再构造 Laplacian L = diag(sum W_ij) - W
    std::vector<Eigen::Triplet<double>> w_triplets;

    auto cot_angle = [&](const Eigen::Vector3d& p0,
                         const Eigen::Vector3d& p1,
                         const Eigen::Vector3d& p2) {
        // cot(angle at p0) — 对边是 p1-p2
        Eigen::Vector3d u = p1 - p0;
        Eigen::Vector3d v = p2 - p0;
        double cosang = u.dot(v);
        double sin2 = (u.cross(v)).squaredNorm();
        if (sin2 <= 1e-16) return 0.0;
        double sinang = std::sqrt(sin2);
        return cosang / sinang;
    };

    Eigen::VectorXd diag = Eigen::VectorXd::Zero(nV);

    for (int fi = 0; fi < nF; ++fi) {
        int i0 = F(fi, 0);
        int i1 = F(fi, 1);
        int i2 = F(fi, 2);

        Eigen::Vector3d p0 = V.row(i0);
        Eigen::Vector3d p1 = V.row(i1);
        Eigen::Vector3d p2 = V.row(i2);

        double cot0 = cot_angle(p0, p1, p2); // at i0
        double cot1 = cot_angle(p1, p2, p0); // at i1
        double cot2 = cot_angle(p2, p0, p1); // at i2

        auto add_weight = [&](int a, int b, double w) {
            if (w == 0.0) return;
            diag(a) += w;
            diag(b) += w;
            w_triplets.emplace_back(a, b, -w);
            w_triplets.emplace_back(b, a, -w);
        };

        add_weight(i1, i2, cot0);
        add_weight(i2, i0, cot1);
        add_weight(i0, i1, cot2);
    }

    // 先构造 W
    Eigen::SparseMatrix<double> W(nV, nV);
    W.setFromTriplets(w_triplets.begin(), w_triplets.end());

    // D - W
    std::vector<Eigen::Triplet<double>> L_trip;
    L_trip.reserve(W.nonZeros() + nV);

    for (int k = 0; k < W.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(W, k); it; ++it) {
            L_trip.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int i = 0; i < nV; ++i) {
        L_trip.emplace_back(i, i, diag(i));
    }

    L.resize(nV, nV);
    L.setFromTriplets(L_trip.begin(), L_trip.end());

    // 和 Matlab 一样：检查负权重并 clamp
    double min_offdiag = 0.0;
    for (int k = 0; k < L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if (it.row() == it.col()) continue;
            if (it.value() < min_offdiag)
                min_offdiag = it.value();
        }
    }
    if (min_offdiag < 0.0) {
        std::cerr << "[build_cotangent_laplacian] Mesh is not Delaunay, "
                  << "clamping negative weights." << std::endl;

        double clamp = 1e-2;
        for (int k = 0; k < L.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                if (it.row() == it.col()) continue;
                if (it.value() < 0.0)
                    const_cast<double&>(it.valueRef()) = -clamp;
            }
        }

        // 重新调和对角线（行和为 0）
        Eigen::VectorXd rowSum = Eigen::VectorXd::Zero(nV);
        for (int k = 0; k < L.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                if (it.row() == it.col()) continue;
                rowSum(it.row()) += it.value();
            }
        }
        for (int i = 0; i < nV; ++i) {
            // 对角线 = - sum_offdiag
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
                if (it.row() == i && it.col() == i) {
                    const_cast<double&>(it.valueRef()) = -rowSum(i);
                }
            }
        }
    }
}

// =======================
// 辅助：从 CutMesh 求边界环（有序）
// =======================

static std::vector<int> ordered_boundary_cycle(const CutMesh& cm, int startVertex)
{
    // 找出边界边：出现一次的边
    int nV = static_cast<int>(cm.V.rows());
    int nF = static_cast<int>(cm.T.rows());

    struct EdgeKey {
        int a, b;
        bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(const EdgeKey& e) const {
            return std::hash<int>()(e.a * 73856093 ^ e.b * 19349663);
        }
    };

    std::unordered_map<EdgeKey, int, EdgeKeyHash> edgeCount;
    auto add_edge = [&](int x, int y) {
        if (x > y) std::swap(x, y);
        EdgeKey k{x, y};
        edgeCount[k] += 1;
    };

    for (int fi = 0; fi < nF; ++fi) {
        int v0 = cm.T(fi, 0);
        int v1 = cm.T(fi, 1);
        int v2 = cm.T(fi, 2);
        add_edge(v0, v1);
        add_edge(v1, v2);
        add_edge(v2, v0);
    }

    // 构建边界图邻接（无向）
    std::vector<std::vector<int>> adj(nV);
    for (auto& kv : edgeCount) {
        if (kv.second == 1) {
            int a = kv.first.a;
            int b = kv.first.b;
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
    }

    // 有些 mesh 可能有多个边界环，我们这里默认你是一个球切开变成单一环
    std::vector<int> boundary;

    int start = startVertex;
    if (start < 0 || start >= nV) {
        // 找一个度不为 0 的边界点
        for (int i = 0; i < nV; ++i) {
            if (!adj[i].empty()) { start = i; break; }
        }
    }
    if (start < 0 || start >= nV || adj[start].empty()) {
        std::cerr << "[ordered_boundary_cycle] no boundary found.\n";
        return boundary;
    }

    boundary.push_back(start);
    int prev = -1;
    int cur = start;

    while (true) {
        const auto& nbrs = adj[cur];
        int nxt = -1;
        for (int v : nbrs) {
            if (v == prev) continue;
            nxt = v;
            break;
        }
        if (nxt == -1) break;
        if (nxt == start) {
            // 闭合
            break;
        }
        boundary.push_back(nxt);
        prev = cur;
        cur = nxt;
        if (boundary.size() > (size_t)nV) break; // 防止死循环
    }

    return boundary;
}

// =======================
// 主函数：flatten_sphere
// =======================
void flatten_sphere(
        MyMesh& mesh,
        const std::vector<int>& cones,
        int orbifold_type,
        bool verbose)
{
    if (cones.size() < 3) {
        throw std::runtime_error("flatten_sphere: need at least 3 cones for sphere orbifolds.");
    }

    // 1. 把 OpenMesh 转成 Eigen (原始网格)
    Eigen::MatrixXd V_orig;
    Eigen::MatrixXi F_orig;
    mesh_to_eigen(mesh, V_orig, F_orig);

    if (verbose) {
        std::cout << "Mesh loaded for flattening: "
                  << V_orig.rows() << " vertices, "
                  << F_orig.rows() << " faces\n";
    }

    // 2. 构造 orbifold 的“singularities” (和 Matlab 一致)
    std::vector<int> singularities;
    switch (orbifold_type) {
        case 1: singularities = {4, 4}; break;      // type I
        case 2: singularities = {3, 3}; break;      // type II
        case 3: singularities = {6, 2}; break;      // type III
        case 4: singularities = {2, 2, 2}; break;   // type IV
        default:
            throw std::runtime_error("flatten_sphere: orbifold_type must be 1..4");
    }

    if (orbifold_type < 4 && cones.size() != 3) {
        throw std::runtime_error("flatten_sphere: orbifold types I–III require exactly 3 cones.");
    }
    if (orbifold_type == 4 && cones.size() != 4) {
        throw std::runtime_error("flatten_sphere: orbifold type IV requires 4 cones.");
    }

    // 3. 构造 cone-tree (和 Matlab: Flattener.cut() 里一样)
    const int k = static_cast<int>(cones.size());
    Eigen::MatrixXi treeAdj = Eigen::MatrixXi::Zero(k, k);
    int treeRoot = 0;
    if (k == 3) {
        // root = 3 (Matlab 1-based)，C++ 用 0-based -> 2
        treeRoot = 2;
        // fixedPairs = [3 1; 3 2]
        treeAdj(2, 0) = 1;
        treeAdj(0, 2) = 1;
        treeAdj(2, 1) = 1;
        treeAdj(1, 2) = 1;
    } else if (k == 4) {
        // fixedPairs = [1 3;3 4;4 2] (Matlab 1-based)
        treeRoot = 0; // root=1
        auto add_e = [&](int a, int b) {
            treeAdj(a, b) = 1;
            treeAdj(b, a) = 1;
        };
        add_e(0, 2); // 1-3
        add_e(2, 3); // 3-4
        add_e(3, 1); // 4-2
    }

    std::vector<int> treeIndices = cones; // 直接引用 cone 在 mesh 中的顶点索引

    // 4. 用 TreeCutter 对原始网格做切缝，得到 CutMesh
    if (verbose) {
        std::cout << "[flatten_sphere] Cutting mesh along cone tree...\n";
    }
    TreeCutter cutter(V_orig, F_orig, treeAdj, treeIndices, treeRoot);
    cutter.cutTree();
    CutMesh cmesh = cutter.getCutMesh();

    if (verbose) {
        std::cout << "[flatten_sphere] After cutting: "
                  << cmesh.V.rows() << " vertices, "
                  << cmesh.T.rows() << " faces, "
                  << cmesh.pathPairs.size() << " seam(s)\n";
    }

    // 5. 构造约束 PosConstraints（sphere orbifold 分支）
    const int nVcut = static_cast<int>(cmesh.V.rows());
    PosConstraints cons(nVcut);

    // 找 cut 网格的边界环（与 Matlab 的 TR.freeBoundary 类似）
    // startP = M_cut.uncutIndsToCutInds{inds(1)} 的第一个副本
    int startP = -1;
    {
        int cone0 = cones[0]; // 原始 index
        if (cone0 >= 0 && cone0 < static_cast<int>(cmesh.uncutIndsToCutInds.size()) &&
            !cmesh.uncutIndsToCutInds[cone0].empty()) {
            startP = cmesh.uncutIndsToCutInds[cone0][0];
        }
    }

    std::vector<int> all_binds = ordered_boundary_cycle(cmesh, startP);
    if (all_binds.empty()) {
        std::cerr << "[flatten_sphere] ERROR: cut mesh still has no boundary.\n";
        return;
    }

    // 旋转 all_binds，让它从 startP 开始
    auto itStart = std::find(all_binds.begin(), all_binds.end(), startP);
    if (itStart != all_binds.end()) {
        std::rotate(all_binds.begin(), itStart, all_binds.end());
    }

    // pathEnds = 所有 cut seam 的两端点
    std::vector<int> pathEnds;
    for (const auto& PP : cmesh.pathPairs) {
        if (PP.rows() == 0) continue;
        int r0 = 0;
        int r1 = PP.rows() - 1;
        // PP([1 end], :) in Matlab
        pathEnds.push_back(PP(r0, 0));
        pathEnds.push_back(PP(r0, 1));
        pathEnds.push_back(PP(r1, 0));
        pathEnds.push_back(PP(r1, 1));
    }
    std::sort(pathEnds.begin(), pathEnds.end());
    pathEnds.erase(std::unique(pathEnds.begin(), pathEnds.end()), pathEnds.end());

    // 找出边界上的“cone 顶点”（在 cut mesh 中）
    std::vector<int> cones_on_boundary;
    for (int v : all_binds) {
        if (std::binary_search(pathEnds.begin(), pathEnds.end(), v)) {
            cones_on_boundary.push_back(v);
        }
    }

    if (verbose) {
        std::cout << "[flatten_sphere] Boundary cycle size = "
                  << all_binds.size()
                  << ", boundary cones found = "
                  << cones_on_boundary.size() << "\n";
    }

    // 建立 map: cut 顶点 -> orbifold 中第几号 cone
    // (通过 cutIndsToUncutInds 映射回原始顶点，然后在 cones 中查位置)
    std::unordered_map<int,int> cutVertexToConeIndex;
    for (size_t i = 0; i < cones_on_boundary.size(); ++i) {
        int v_cut = cones_on_boundary[i];
        if (v_cut < 0 || v_cut >= static_cast<int>(cmesh.cutIndsToUncutInds.size()))
            continue;
        int v_orig = cmesh.cutIndsToUncutInds[v_cut];
        auto it = std::find(cones.begin(), cones.end(), v_orig);
        if (it != cones.end()) {
            int coneIdx = static_cast<int>(it - cones.begin()); // 0..k-1
            cutVertexToConeIndex[v_cut] = coneIdx;
        }
    }

    // 准备给每个 boundary 顶点一个“角度类型”，类似 Matlab 的 angs cell
    // 这里用 map<int,int>，值=0 表示“没有角度约束”（只做平移约束）
    std::unordered_map<int,int> angs; // cut vertex -> singularity value (2,3,4,6,..)

    // 给 cones_on_boundary 做位置 + 角度约束
    const int nbConesOnBoundary = static_cast<int>(cones_on_boundary.size());
    if (nbConesOnBoundary == 0) {
        std::cerr << "[flatten_sphere] WARNING: no boundary cones found; "
                  << "constraints may be insufficient.\n";
    }

    // 用一个简单的环上均匀分布（与 Matlab 类似）
    std::vector<Eigen::Vector2d> coords(nbConesOnBoundary);

    for (int i = 0; i < nbConesOnBoundary; ++i) {
        double theta = 2.0 * M_PI * (i+1) / nbConesOnBoundary + M_PI / 4.0;
        coords[i] = std::sqrt(2.0) * Eigen::Vector2d(std::cos(theta), std::sin(theta));
    }

    // tcoords：不同 orbifold 类型有特殊处理
    std::vector<Eigen::Vector2d> tcoords;
    if (orbifold_type == 4) {
        // type IV: 两个 cone 固定在竖直方向
        tcoords.resize(2);
        tcoords[0] = Eigen::Vector2d(0.0, -0.5);
        tcoords[1] = Eigen::Vector2d(0.0,  0.5);
    } else if (orbifold_type == 1 && singularities.size() >= 2 &&
               singularities[0] == 4 && singularities[1] == 4) {
        tcoords.resize(2);
        tcoords[0] = Eigen::Vector2d(-1.0, -1.0);
        tcoords[1] = Eigen::Vector2d( 1.0,  1.0);
    } else {
        tcoords = coords; // type II,III
    }

    // 实际把 cone 点绑在平面上的固定位置，并给前两个 cone 设置 angle
    // ====== 修正版：严格按 cones 的顺序给角点加约束 ======
    for (int v_cut : cones_on_boundary)
    {
        // 找到这个 cut 顶点对应的是 cones[] 里的第几个
        auto itCone = cutVertexToConeIndex.find(v_cut);
        if (itCone == cutVertexToConeIndex.end())
            continue;

        int coneIdx = itCone->second; // 在 cones 中的序号：0..k-1

        // 只有前几个 cones（取决于 singularities.size()）是“真正的角点”
        // 对于 type I：singularities = {4,4}，只有 cones[0]、cones[1] 有角度约束
        if (coneIdx < static_cast<int>(singularities.size()) &&
            coneIdx < static_cast<int>(tcoords.size()))
        {
            // 把这个 cone 坐标钉在预设的 tcoords[coneIdx] 上
            cons.addConstraint(v_cut, 1.0, tcoords[coneIdx]);

            // 记录它的 cone 角度，用于后面 seam 的旋转约束
            angs[v_cut] = singularities[coneIdx];
        }
        else
        {
            // 其余 cones（比如第三个）不加 angle 约束，只由 seam 约束决定
            // 如果你想它也“钉在某个位置”，可以在这里额外加位置约束
        }

        // type IV 的特殊处理：第二个 cone 要额外加一个约束
        if (orbifold_type == 4 && coneIdx == 1)
        {
            cons.addConstraint(v_cut, 1.0, Eigen::Vector2d(1.0, -0.5));
        }
    }


    // 6. 对每条 cut seam 加旋转/平移约束：T*x_s = x_t（or up to rotation）
    if (verbose) {
        std::cout << "[flatten_sphere] Adding seam constraints...\n";
    }

    for (const auto& PP : cmesh.pathPairs) {
        if (PP.rows() == 0) continue;

        // path1 = PP(:,0), path2 = PP(:,1)
        std::vector<int> path1, path2;
        path1.reserve(PP.rows());
        path2.reserve(PP.rows());
        for (int r = 0; r < PP.rows(); ++r) {
            path1.push_back(PP(r, 0));
            path2.push_back(PP(r, 1));
        }

        int sign = -1;
        if (path1.back() == path2.back()) {
            std::reverse(path1.begin(), path1.end());
            std::reverse(path2.begin(), path2.end());
            sign = 1;
        }

        // 取 path1 的起点，查它是否有角度约束
        int v0 = path1.front();
        int ang_val = 0;
        auto itAng = angs.find(v0);
        if (itAng != angs.end()) {
            ang_val = itAng->second;
        }

        if (ang_val == 0) {
            ang_val = 1; // 没有角度时，只做平移约束（相当于 rotation = Identity）
        }

        if (ang_val != 0) {
            ang_val *= sign;
            double theta = 2.0 * M_PI / static_cast<double>(ang_val);
            Eigen::Matrix2d R;
            R << std::cos(theta), -std::sin(theta),
                    std::sin(theta),  std::cos(theta);

            cons.addTransConstraints(path1, path2, R);
        }
    }

    // 7. 构建 Laplacian (2nVcut x 2nVcut)
    if (verbose) {
        std::cout << "[flatten_sphere] Building cotangent Laplacian...\n";
    }

    Eigen::SparseMatrix<double> L0;
    build_cotangent_laplacian(cmesh.V, cmesh.T, L0);

    Eigen::SparseMatrix<double> L(2 * nVcut, 2 * nVcut);
    std::vector<Eigen::Triplet<double>> Ltrip;
    Ltrip.reserve(L0.nonZeros() * 2);
    for (int kcol = 0; kcol < L0.outerSize(); ++kcol) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L0, kcol); it; ++it) {
            int i = it.row();
            int j = it.col();
            double v = it.value();
            // x 分量
            Ltrip.emplace_back(2*i,   2*j,   v);
            // y 分量
            Ltrip.emplace_back(2*i+1, 2*j+1, v);
        }
    }
    L.setFromTriplets(Ltrip.begin(), Ltrip.end());

    // 8. 求解 KKT 系统，得到平面坐标向量 x
    Eigen::MatrixXd A = cons.getA();
    Eigen::VectorXd b = cons.getB();

    if (verbose) {
        std::cout << "[flatten_sphere] Number of constraints: "
                  << cons.numConstraints() << "\n";
        std::cout << "[flatten_sphere] Solving linear system (KKT)...\n";
    }
    Eigen::VectorXd x = computeFlatteningCxx(L, A, b);

    if (verbose) {
        std::cout << "[flatten_sphere] Linear system solved.\n";
    }

    // 9. 拆成 flat_V (nVcut x 2)
    Eigen::MatrixXd flat_V(nVcut, 2);
    for (int i = 0; i < nVcut; ++i) {
        flat_V(i, 0) = x(2*i);
        flat_V(i, 1) = x(2*i + 1);
    }

    // 9.25 统计每个原始锥点在 cut 网格中的所有副本，并打印
    std::vector<std::vector<int>> coneCopies(cones.size());
    for (int v = 0; v < nVcut; ++v) {
        int orig = cmesh.cutIndsToUncutInds[v];  // 这个 cut 顶点来自哪个原始顶点
        for (size_t ci = 0; ci < cones.size(); ++ci) {
            if (orig == cones[ci]) {
                coneCopies[ci].push_back(v);
            }
        }
    }

    std::cout << "[DEBUG] Cone copy counts:\n";
    for (size_t ci = 0; ci < cones.size(); ++ci) {
        std::cout << "  original cone " << cones[ci]
                  << " -> " << coneCopies[ci].size()
                  << " cut copies: ";
        for (int cv : coneCopies[ci]) std::cout << cv << " ";
        std::cout << "\n";
    }

    // =====================================================
    // 9.5 生成调试网格：标记锥点 & 切缝上的点，并输出 flattened_debug.off
    //     - 锥点: 红色 (255,0,0)
    //     - 切缝上的点: 蓝色 (0,0,255)
    //     - 其他点: 灰色 (200,200,200)
    // =====================================================
    try {
        // 标记 cut 网格上的点
        std::vector<bool> isConeCut(nVcut, false);
        std::vector<bool> isSeamCut(nVcut, false);

        // 1) 用 coneCopies 标记所有锥点副本
        for (size_t ci = 0; ci < coneCopies.size(); ++ci) {
            for (int cutIdx : coneCopies[ci]) {
                if (cutIdx >= 0 && cutIdx < nVcut) {
                    isConeCut[cutIdx] = true;
                }
            }
        }

        // 2) 所有 seam 路径上的点标记为 seam
        for (const auto& PP : cmesh.pathPairs) {
            for (int r = 0; r < PP.rows(); ++r) {
                int v1 = PP(r, 0);
                int v2 = PP(r, 1);
                if (v1 >= 0 && v1 < nVcut) isSeamCut[v1] = true;
                if (v2 >= 0 && v2 < nVcut) isSeamCut[v2] = true;
            }
        }

        // 3) 构造调试用的 OpenMesh 网格（用 cut 网格，顶点数 = nVcut）
        MyMesh debugMesh;
        debugMesh.request_vertex_colors();

        std::vector<MyMesh::VertexHandle> vhandles(nVcut);
        for (int i = 0; i < nVcut; ++i) {
            float x2d = static_cast<float>(flat_V(i, 0));
            float y2d = static_cast<float>(flat_V(i, 1));
            vhandles[i] = debugMesh.add_vertex(MyMesh::Point(x2d, y2d, 0.0f));
        }

        // 添加三角面
        int nFcut = static_cast<int>(cmesh.T.rows());
        for (int fi = 0; fi < nFcut; ++fi) {
            int i0 = cmesh.T(fi, 0);
            int i1 = cmesh.T(fi, 1);
            int i2 = cmesh.T(fi, 2);
            if (i0 < 0 || i0 >= nVcut ||
                i1 < 0 || i1 >= nVcut ||
                i2 < 0 || i2 >= nVcut) {
                continue;
            }
            std::vector<MyMesh::VertexHandle> face_vhandles;
            face_vhandles.reserve(3);
            face_vhandles.push_back(vhandles[i0]);
            face_vhandles.push_back(vhandles[i1]);
            face_vhandles.push_back(vhandles[i2]);
            debugMesh.add_face(face_vhandles);
        }

        // 设置颜色
        for (int i = 0; i < nVcut; ++i) {
            MyMesh::Color c(200, 200, 200); // 默认灰色
            if (isSeamCut[i]) {
                c = MyMesh::Color(0, 0, 255); // seam: 蓝色
            }
            if (isConeCut[i]) {
                c = MyMesh::Color(255, 0, 0); // cone: 红色（覆盖 seam）
            }
            debugMesh.set_color(vhandles[i], c);
        }

        // 写出调试 OFF（带顶点颜色）
        OpenMesh::IO::Options wopt;
        wopt += OpenMesh::IO::Options::VertexColor;

        std::string debug_name = "flattened_debug.off";
        if (verbose) {
            std::cout << "[flatten_sphere] Writing debug mesh with cone & seam colors to: "
                      << debug_name << "\n";
        }
        if (!OpenMesh::IO::write_mesh(debugMesh, debug_name, wopt)) {
            std::cerr << "[flatten_sphere] Warning: cannot write debug mesh "
                      << debug_name << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[flatten_sphere] Exception while writing debug mesh: "
                  << e.what() << "\n";
    }

    // 10. 把展平结果写回原始 OpenMesh（简单版：每个原始顶点取第一个 cut 副本）
    eigen_to_mesh_flat(mesh, flat_V, cmesh);

    if (verbose) {
        std::cout << "Flattening finished. Mesh vertices updated to 2D.\n";
    }
}




