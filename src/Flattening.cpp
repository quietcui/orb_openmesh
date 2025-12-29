// Flattening.cpp
//
// 本实现是对作者MATLAB代码的逐行近似转换，
// 源自：
//   https://github.com/noamaig/euclidean_orbifolds
// 具体对应Flattener.flatten_sphere()与
// computeFlattening()中的逻辑。
//
// 关键约定（与 MATLAB 仓库一致）：
// - 余切拉普拉斯算子 L0 具有 **正非对角权重** 和
//   **负对角权重**（每行求和为 0）。这与作者的
//   cotmatrix.m 输出结果一致。
// - 若存在负余切权重（非德劳内网格），则将其限制为
//   小正常数（1e-2），随后重新计算对角线。
// - 边界锥检测方式完全遵循MATLAB：
//     pathEnds = unique([pathPairs{*}(1,:) pathPairs{*}(end,:)])
//     p        = all_binds( ismember(all_binds, pathEnds) )
//   并将首个边界顶点旋转至 inds(1) 的切割索引处起始。

#include "Flattening.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/IO/Options.hh>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static constexpr double kPi = 3.141592653589793238462643383279502884;

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

    int idx = 0;
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it, ++idx)
    {
        auto p = mesh.point(*v_it);
        V(idx, 0) = p[0];
        V(idx, 1) = p[1];
        V(idx, 2) = p[2];
    }

    idx = 0;
    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it, ++idx)
    {
        int k = 0;
        for (auto fv_it = mesh.cfv_iter(*f_it); fv_it.is_valid(); ++fv_it)
        {
            F(idx, k++) = fv_it->idx();
            if (k >= 3) break;
        }
    }
}

static void eigen_to_mesh_flat(
        MyMesh& mesh,
        const Eigen::MatrixXd& flat_V,
        const CutMesh& cutMesh)
{
    // For each original vertex, choose the first cut copy (same as earlier code).
    const int nOrigV = static_cast<int>(mesh.n_vertices());
    if (static_cast<int>(cutMesh.uncutIndsToCutInds.size()) != nOrigV)
    {
        std::cerr << "[eigen_to_mesh_flat] Warning: uncutIndsToCutInds size != n_orig_vertices\n";
    }

    int idx = 0;
    for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it, ++idx)
    {
        if (idx >= static_cast<int>(cutMesh.uncutIndsToCutInds.size()))
            break;

        const auto& copies = cutMesh.uncutIndsToCutInds[idx];
        if (copies.empty()) continue;

        const int cutIdx = copies[0];
        if (cutIdx < 0 || cutIdx >= static_cast<int>(flat_V.rows()))
            continue;

        const double x = flat_V(cutIdx, 0);
        const double y = flat_V(cutIdx, 1);
        mesh.set_point(*v_it, MyMesh::Point(static_cast<float>(x),
                                            static_cast<float>(y),
                                            0.0f));
    }
}

// =======================
// Cotangent Laplacian (MATLAB cotmatrix.m convention)
// =======================

using SparseRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

static double cot_angle(const Eigen::Vector3d& p0,
                        const Eigen::Vector3d& p1,
                        const Eigen::Vector3d& p2)
{
    // cot(angle at p0) where rays are (p1-p0) and (p2-p0)
    const Eigen::Vector3d u = p1 - p0;
    const Eigen::Vector3d v = p2 - p0;
    const double dot = u.dot(v);
    const double cross_norm = u.cross(v).norm(); // == 2*area of triangle
    if (cross_norm <= 1e-16) return 0.0;
    return dot / cross_norm;
}

static void build_cotmatrix_like_matlab(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        SparseRM& L)
{
    // Matches euclidean_orbifolds/cotmatrix.m:
    // - offdiag: +0.5*cot(opposite angle)
    // - diag: negative row sum
    const int nV = static_cast<int>(V.rows());
    const int nF = static_cast<int>(T.rows());

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(nF) * 12);

    Eigen::VectorXd diag = Eigen::VectorXd::Zero(nV);

    auto add_w = [&](int a, int b, double w)
    {
        if (a == b) return;
        if (std::abs(w) <= 0.0) return;
        trips.emplace_back(a, b, w);
        trips.emplace_back(b, a, w);
        diag(a) -= w;
        diag(b) -= w;
    };

    for (int fi = 0; fi < nF; ++fi)
    {
        const int i0 = T(fi, 0);
        const int i1 = T(fi, 1);
        const int i2 = T(fi, 2);

        const Eigen::Vector3d p0 = V.row(i0);
        const Eigen::Vector3d p1 = V.row(i1);
        const Eigen::Vector3d p2 = V.row(i2);

        // MATLAB cotmatrix uses 0.5*cot(angle)
        const double cot0 = 0.5 * cot_angle(p0, p1, p2); // opposite edge (i1,i2)
        const double cot1 = 0.5 * cot_angle(p1, p2, p0); // opposite edge (i2,i0)
        const double cot2 = 0.5 * cot_angle(p2, p0, p1); // opposite edge (i0,i1)

        add_w(i1, i2, cot0);
        add_w(i2, i0, cot1);
        add_w(i0, i1, cot2);
    }

    for (int i = 0; i < nV; ++i)
        trips.emplace_back(i, i, diag(i));

    L.resize(nV, nV);
    L.setFromTriplets(trips.begin(), trips.end());

    // MATLAB clamping:
    //   m = min(min(tril(L,-1)));
    //   if m<0: L(L<0)=clamp; diag=0; diag=-sum(L);
    // Here we clamp negative **off-diagonal** weights.
    double min_offdiag = 0.0;
    for (int i = 0; i < L.rows(); ++i)
    {
        for (SparseRM::InnerIterator it(L, i); it; ++it)
        {
            if (it.col() == it.row()) continue;
            min_offdiag = std::min(min_offdiag, it.value());
        }
    }

    if (min_offdiag < 0.0)
    {
        std::cerr << "[build_cotmatrix_like_matlab] Warning: Mesh is not Delaunay, clamping negative cot weights.\n";
        const double clamp = 1e-2;

        // Clamp off-diagonals
        for (int i = 0; i < L.rows(); ++i)
        {
            for (SparseRM::InnerIterator it(L, i); it; ++it)
            {
                if (it.col() == it.row()) continue;
                if (it.value() < 0.0)
                    it.valueRef() = clamp;
            }
        }

        // Recompute diagonal: diag(i) = -sum_{j!=i} L(i,j)
        for (int i = 0; i < L.rows(); ++i)
        {
            double s = 0.0;
            bool has_diag = false;
            SparseRM::InnerIterator diagIt(L, i);
            for (SparseRM::InnerIterator it(L, i); it; ++it)
            {
                if (it.col() == i)
                {
                    has_diag = true;
                    diagIt = it;
                }
                else
                {
                    s += it.value();
                }
            }

            const double new_diag = -s;
            if (has_diag)
            {
                diagIt.valueRef() = new_diag;
            }
            else
            {
                // Should not happen because we always insert diagonal, but be robust.
                L.coeffRef(i, i) = new_diag;
            }
        }
    }
}

// =======================
// Free boundary cycle (like MATLAB triangulation(...).freeBoundary)
// =======================

static inline std::uint64_t edge_key(int a, int b)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<std::uint32_t>(b);
}

static std::vector<int> ordered_boundary_cycle_directed(const Eigen::MatrixXi& T, int nV, int startVertex)
{
    // Build directed halfedge set from triangles (v0->v1, v1->v2, v2->v0)
    std::unordered_set<std::uint64_t> halfedges;
    halfedges.reserve(static_cast<size_t>(T.rows()) * 3);

    for (int fi = 0; fi < T.rows(); ++fi)
    {
        const int a = T(fi, 0);
        const int b = T(fi, 1);
        const int c = T(fi, 2);
        halfedges.insert(edge_key(a, b));
        halfedges.insert(edge_key(b, c));
        halfedges.insert(edge_key(c, a));
    }

    std::vector<std::vector<int>> out(static_cast<size_t>(nV));
    out.shrink_to_fit();
    out.assign(static_cast<size_t>(nV), {});

    // Boundary directed edges are those whose reverse halfedge is absent.
    for (const auto& k : halfedges)
    {
        const int u = static_cast<int>(k >> 32);
        const int v = static_cast<int>(k & 0xFFFFFFFFu);
        const std::uint64_t rev = edge_key(v, u);
        if (halfedges.find(rev) == halfedges.end())
        {
            if (u >= 0 && u < nV)
                out[static_cast<size_t>(u)].push_back(v);
        }
    }

    int start = startVertex;
    if (start < 0 || start >= nV || out[static_cast<size_t>(start)].empty())
    {
        start = -1;
        for (int i = 0; i < nV; ++i)
        {
            if (!out[static_cast<size_t>(i)].empty()) { start = i; break; }
        }
    }
    if (start < 0) return {};

    // If mesh orientation is consistent and boundary is a single cycle,
    // each boundary vertex should have exactly one outgoing boundary edge.
    // If not, fall back to an undirected walk (handled elsewhere).
    for (int i = 0; i < nV; ++i)
    {
        if (out[static_cast<size_t>(i)].size() > 1)
        {
            // ambiguous
            return {};
        }
    }

    std::vector<int> cycle;
    cycle.reserve(static_cast<size_t>(nV));

    int cur = start;
    for (int guard = 0; guard < nV + 5; ++guard)
    {
        cycle.push_back(cur);
        const auto& nxts = out[static_cast<size_t>(cur)];
        if (nxts.empty()) break;
        const int nxt = nxts[0];
        cur = nxt;
        if (cur == start) break;
    }

    return cycle;
}

static std::vector<int> ordered_boundary_cycle_undirected(const Eigen::MatrixXi& T, int nV, int startVertex)
{
    // Undirected boundary extraction: count undirected edges, keep those with count==1.
    struct EdgeKey { int a, b; };
    struct EdgeHash {
        std::size_t operator()(const EdgeKey& e) const {
            return std::hash<long long>()((static_cast<long long>(e.a) << 32) ^ e.b);
        }
    };
    struct EdgeEq {
        bool operator()(const EdgeKey& x, const EdgeKey& y) const {
            return x.a == y.a && x.b == y.b;
        }
    };

    std::unordered_map<EdgeKey, int, EdgeHash, EdgeEq> cnt;
    cnt.reserve(static_cast<size_t>(T.rows()) * 3);

    auto add = [&](int u, int v)
    {
        if (u > v) std::swap(u, v);
        cnt[EdgeKey{u, v}] += 1;
    };

    for (int fi = 0; fi < T.rows(); ++fi)
    {
        const int a = T(fi, 0);
        const int b = T(fi, 1);
        const int c = T(fi, 2);
        add(a, b);
        add(b, c);
        add(c, a);
    }

    std::vector<std::vector<int>> adj(static_cast<size_t>(nV));
    for (const auto& kv : cnt)
    {
        if (kv.second != 1) continue;
        const int a = kv.first.a;
        const int b = kv.first.b;
        if (a >= 0 && a < nV) adj[static_cast<size_t>(a)].push_back(b);
        if (b >= 0 && b < nV) adj[static_cast<size_t>(b)].push_back(a);
    }

    int start = startVertex;
    if (start < 0 || start >= nV || adj[static_cast<size_t>(start)].empty())
    {
        start = -1;
        for (int i = 0; i < nV; ++i)
        {
            if (!adj[static_cast<size_t>(i)].empty()) { start = i; break; }
        }
    }
    if (start < 0) return {};

    std::vector<int> cycle;
    cycle.reserve(static_cast<size_t>(nV));
    cycle.push_back(start);

    int prev = -1;
    int cur = start;
    for (int guard = 0; guard < nV + 5; ++guard)
    {
        const auto& nbrs = adj[static_cast<size_t>(cur)];
        if (nbrs.empty()) break;
        int nxt = -1;
        if (nbrs.size() == 1)
        {
            nxt = nbrs[0];
        }
        else
        {
            // pick neighbor different from prev
            nxt = (nbrs[0] == prev) ? nbrs[1] : nbrs[0];
        }
        if (nxt == start) break;
        cycle.push_back(nxt);
        prev = cur;
        cur = nxt;
    }

    return cycle;
}

static std::vector<int> ordered_boundary_cycle(const CutMesh& cm, int startVertex)
{
    const int nV = static_cast<int>(cm.V.rows());
    // Try directed first (closest to MATLAB freeBoundary ordering)
    std::vector<int> cyc = ordered_boundary_cycle_directed(cm.T, nV, startVertex);
    if (!cyc.empty()) return cyc;
    // Fallback
    return ordered_boundary_cycle_undirected(cm.T, nV, startVertex);
}

// =======================
// Main API: flatten_sphere
// =======================

void flatten_sphere(MyMesh& mesh,const std::vector<int>& cones,
                    int orbifold_type,bool verbose)
{
    if (cones.size() < 3)
        throw std::runtime_error("flatten_sphere: need at least 3 cone indices.");

    // 1) OpenMesh -> Eigen
    Eigen::MatrixXd V_orig;
    Eigen::MatrixXi T_orig;
    mesh_to_eigen(mesh, V_orig, T_orig);

    if (verbose)
    {
        std::cout << "Mesh loaded for flattening: "
                  << V_orig.rows() << " vertices, "
                  << T_orig.rows() << " faces\n";
    }

    // 2) Orbifold singularities (MATCH MATLAB)
    // MATLAB Flattener: singularities are defined by cone angles at 1st, 3rd,
    // and possibly 4th cone, but in flatten_sphere() it only assigns angles to
    // ind<=2 (i.e., cones[0] and cones[1]).
    std::vector<int> singularities;
    switch (orbifold_type)
    {
        case 1: singularities = {4, 4}; break;      // Square
        case 2: singularities = {3, 3}; break;      // Diamond
        case 3: singularities = {6, 2}; break;      // Triangle
        case 4: singularities = {2, 2, 2}; break;   // Parallelogram (4 cones)
        default:
            throw std::runtime_error("flatten_sphere: orbifold_type must be 1..4");
    }

    if (orbifold_type < 4 && cones.size() != 3)
        throw std::runtime_error("flatten_sphere: orbifold types 1..3 require exactly 3 cones.");
    if (orbifold_type == 4 && cones.size() != 4)
        throw std::runtime_error("flatten_sphere: orbifold type 4 requires exactly 4 cones.");

    // 3) Build cone-tree (MATCH MATLAB)
    const int k = static_cast<int>(cones.size());
    Eigen::MatrixXi treeAdj = Eigen::MatrixXi::Zero(k, k);
    int treeRoot = 0;

    if (k == 3)
    {
        // MATLAB: root = length(inds) (=3), fixedPairs = [root 1; root 2]
        treeRoot = 2;
        treeAdj(2, 0) = treeAdj(0, 2) = 1;
        treeAdj(2, 1) = treeAdj(1, 2) = 1;
    }
    else if (k == 4)
    {
        // MATLAB: root = 1, fixedPairs = [1 3; 3 4; 4 2]
        treeRoot = 0;
        auto add_e = [&](int a, int b)
        {
            treeAdj(a, b) = 1;
            treeAdj(b, a) = 1;
        };
        add_e(0, 2);
        add_e(2, 3);
        add_e(3, 1);
    }

    // 4) Cut the mesh
    if (verbose)
        std::cout << "[flatten_sphere] Cutting mesh along cone tree...\n";

    TreeCutter cutter(V_orig, T_orig, treeAdj, cones, treeRoot);
    cutter.cutTree();
    CutMesh M_cut = cutter.getCutMesh();

    const int nVcut = static_cast<int>(M_cut.V.rows());
    if (verbose)
    {
        std::cout << "[flatten_sphere] After cutting: "
                  << M_cut.V.rows() << " vertices, "
                  << M_cut.T.rows() << " faces, "
                  << M_cut.pathPairs.size() << " seam(s)\n";
    }

    // 5) Build constraints (MATCH MATLAB)
    PosConstraints cons(nVcut);

    // startP = uncutIndsToCutInds{inds(1)}; assert length==1
    int startP = -1;
    if (cones[0] >= 0 && cones[0] < static_cast<int>(M_cut.uncutIndsToCutInds.size()) &&
        !M_cut.uncutIndsToCutInds[cones[0]].empty())
    {
        startP = M_cut.uncutIndsToCutInds[cones[0]][0];
        if (M_cut.uncutIndsToCutInds[cones[0]].size() != 1)
        {
            std::cerr << "[flatten_sphere] Warning: MATLAB expects inds(1) to have exactly one cut copy, "
                      << "but got " << M_cut.uncutIndsToCutInds[cones[0]].size() << "\n";
        }
    }

    std::vector<int> all_binds = ordered_boundary_cycle(M_cut, startP);
    if (all_binds.empty())
        throw std::runtime_error("flatten_sphere: cut mesh has no boundary (cut failed?)");

    // Rotate boundary cycle so it starts at startP (MATLAB: all_binds([ind:end,1:ind-1]))
    if (startP >= 0)
    {
        auto it = std::find(all_binds.begin(), all_binds.end(), startP);
        if (it != all_binds.end())
            std::rotate(all_binds.begin(), it, all_binds.end());
    }

    // pathEnds = unique([pathPairs{i}(1,:) pathPairs{i}(end,:)])
    std::vector<int> pathEnds;
    pathEnds.reserve(M_cut.pathPairs.size() * 4);
    for (const auto& PP : M_cut.pathPairs)
    {
        if (PP.rows() == 0) continue;
        const int r0 = 0;
        const int r1 = PP.rows() - 1;
        pathEnds.push_back(PP(r0, 0));
        pathEnds.push_back(PP(r0, 1));
        pathEnds.push_back(PP(r1, 0));
        pathEnds.push_back(PP(r1, 1));
    }
    std::sort(pathEnds.begin(), pathEnds.end());
    pathEnds.erase(std::unique(pathEnds.begin(), pathEnds.end()), pathEnds.end());

    // p = all_binds( ismember(all_binds, pathEnds) )
    std::unordered_set<int> pathEndsSet;
    pathEndsSet.reserve(pathEnds.size() * 2);
    for (int v : pathEnds) pathEndsSet.insert(v);

    std::vector<int> p;
    p.reserve(pathEnds.size());
    for (int v : all_binds)
    {
        if (pathEndsSet.find(v) != pathEndsSet.end())
            p.push_back(v);
    }

    if (verbose)
        std::cout << "[flatten_sphere] Boundary cycle size=" << all_binds.size()
                  << "  pathEnds=" << pathEnds.size()
                  << "  p(cones on boundary)=" << p.size() << "\n";

    // coords = sqrt(2)*[cos(theta), sin(theta)] where theta = 2*pi*(1:m)/m + pi/4
    const int m = static_cast<int>(p.size());
    if (m < 2)
        throw std::runtime_error("flatten_sphere: boundary has too few cone endpoints.");

    std::vector<Eigen::Vector2d> coords(m);
    for (int i = 0; i < m; ++i)
    {
        const double theta = 2.0 * kPi * (static_cast<double>(i + 1) / static_cast<double>(m)) + kPi / 4.0;
        coords[i] = std::sqrt(2.0) * Eigen::Vector2d(std::cos(theta), std::sin(theta));
    }

    // tcoords (MATCH MATLAB)
    std::vector<Eigen::Vector2d> tcoords;
    if (cones.size() == 4)
    {
        tcoords = {Eigen::Vector2d(0.0, -0.5), Eigen::Vector2d(0.0, 0.5)};
    }
    else if (orbifold_type == 1 && singularities.size() >= 2 &&
             singularities[0] == 4 && singularities[1] == 4)
    {
        tcoords = {Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, 1.0)};
    }
    else
    {
        tcoords = coords; // types II/III
    }

    // angs{v_cut} = singularities(ind) for ind<=2; otherwise empty.
    std::unordered_map<int, int> angs;
    angs.reserve(p.size());

    for (int i = 0; i < m; ++i)
    {
        const int v_cut = p[i];
        if (v_cut < 0 || v_cut >= static_cast<int>(M_cut.cutIndsToUncutInds.size()))
            continue;

        const int v_orig = M_cut.cutIndsToUncutInds[v_cut];
        const auto itCone = std::find(cones.begin(), cones.end(), v_orig);
        if (itCone == cones.end())
            continue;
        const int ind = static_cast<int>(itCone - cones.begin()); // 0..k-1

        if (ind <= 1)
        {
            if (ind < static_cast<int>(tcoords.size()))
                cons.addConstraint(v_cut, 1.0, tcoords[ind]);
            if (ind < static_cast<int>(singularities.size()))
                angs[v_cut] = singularities[ind];
        }

        // MATLAB special-case for 4 cones: if length(inds)==4 && i==2
        if (cones.size() == 4 && i == 1)
        {
            cons.addConstraint(v_cut, 1.0, Eigen::Vector2d(1.0, -0.5));
        }
    }

    // 6) Seam constraints: for each pathPair, addTransConstraints with rotation
    if (verbose)
        std::cout << "[flatten_sphere] Adding seam constraints...\n";

    for (const auto& PP : M_cut.pathPairs)
    {
        if (PP.rows() == 0) continue;

        std::vector<int> path1(PP.rows());
        std::vector<int> path2(PP.rows());
        for (int r = 0; r < PP.rows(); ++r)
        {
            path1[r] = PP(r, 0);
            path2[r] = PP(r, 1);
        }

        int sign = -1;
        if (!path1.empty() && path1.back() == path2.back())
        {
            std::reverse(path1.begin(), path1.end());
            std::reverse(path2.begin(), path2.end());
            sign = 1;
        }

        const int v0 = path1.front();
        int ang = 1;
        auto itAng = angs.find(v0);
        if (itAng != angs.end())
            ang = itAng->second;
        ang *= sign;

        const double theta = 2.0 * kPi / static_cast<double>(ang);
        Eigen::Matrix2d R;
        R << std::cos(theta), -std::sin(theta),
                std::sin(theta),  std::cos(theta);

        cons.addTransConstraints(path1, path2, R);
    }

    // 7) Laplacian: L = kron(L0, I2)
    if (verbose)
        std::cout << "[flatten_sphere] Building cotangent Laplacian...\n";

    SparseRM L0;
    build_cotmatrix_like_matlab(M_cut.V, M_cut.T, L0);

    Eigen::SparseMatrix<double> L(2 * nVcut, 2 * nVcut);
    std::vector<Eigen::Triplet<double>> Ltrips;
    Ltrips.reserve(static_cast<size_t>(L0.nonZeros()) * 2);

    for (int i = 0; i < L0.rows(); ++i)
    {
        for (SparseRM::InnerIterator it(L0, i); it; ++it)
        {
            const int r = it.row();
            const int c = it.col();
            const double v = it.value();
            Ltrips.emplace_back(2 * r,     2 * c,     v);
            Ltrips.emplace_back(2 * r + 1, 2 * c + 1, v);
        }
    }
    L.setFromTriplets(Ltrips.begin(), Ltrips.end());

    // 8) Solve the KKT system (computeFlattening)
    const Eigen::SparseMatrix<double> A = cons.getA();
    const Eigen::VectorXd b = cons.getB();

    if (verbose)
    {
        std::cout << "[flatten_sphere] #constraints = " << cons.numConstraints() << "\n";
        std::cout << "[flatten_sphere] Solving KKT system...\n";
    }

    const Eigen::VectorXd x = computeFlatteningCxx(L, A, b);

    if (verbose)
        std::cout << "[flatten_sphere] KKT solved.\n";

    // 9) Unpack x into flat_V (nVcut x 2)
    Eigen::MatrixXd flat_V(nVcut, 2);
    for (int i = 0; i < nVcut; ++i)
    {
        flat_V(i, 0) = x(2 * i);
        flat_V(i, 1) = x(2 * i + 1);
    }

    // Optional: write a debug cut mesh (colors for seam & cones) for inspection.
    // This matches the earlier debugging workflow (not in MATLAB, but helpful).
    try
    {
        std::vector<char> isSeam(nVcut, 0);
        for (const auto& PP : M_cut.pathPairs)
        {
            for (int r = 0; r < PP.rows(); ++r)
            {
                const int a = PP(r, 0);
                const int b2 = PP(r, 1);
                if (a >= 0 && a < nVcut) isSeam[a] = 1;
                if (b2 >= 0 && b2 < nVcut) isSeam[b2] = 1;
            }
        }

        std::vector<char> isCone(nVcut, 0);
        for (int v_cut : p)
        {
            if (v_cut >= 0 && v_cut < nVcut) isCone[v_cut] = 1;
        }

        MyMesh debugMesh;
        debugMesh.request_vertex_colors();

        std::vector<MyMesh::VertexHandle> vhandles(nVcut);
        for (int i = 0; i < nVcut; ++i)
        {
            vhandles[i] = debugMesh.add_vertex(MyMesh::Point(
                    static_cast<float>(flat_V(i, 0)),
                    static_cast<float>(flat_V(i, 1)),
                    0.0f));
        }

        for (int fi = 0; fi < M_cut.T.rows(); ++fi)
        {
            const int a = M_cut.T(fi, 0);
            const int b2 = M_cut.T(fi, 1);
            const int c = M_cut.T(fi, 2);
            if (a < 0 || a >= nVcut || b2 < 0 || b2 >= nVcut || c < 0 || c >= nVcut)
                continue;
            debugMesh.add_face({vhandles[a], vhandles[b2], vhandles[c]});
        }

        for (int i = 0; i < nVcut; ++i)
        {
            MyMesh::Color col(200, 200, 200);
            if (isSeam[i]) col = MyMesh::Color(0, 0, 255);
            if (isCone[i]) col = MyMesh::Color(255, 0, 0);
            debugMesh.set_color(vhandles[i], col);
        }

        OpenMesh::IO::Options wopt;
        wopt += OpenMesh::IO::Options::VertexColor;
        OpenMesh::IO::write_mesh(debugMesh, "flattened_debug.off", wopt);
    }
    catch (...) {
        // ignore debug output failures
    }

    // 10) Write result back to original mesh
    eigen_to_mesh_flat(mesh, flat_V, M_cut);

    if (verbose)
        std::cout << "Flattening finished. Mesh vertices updated to 2D.\n";
}
