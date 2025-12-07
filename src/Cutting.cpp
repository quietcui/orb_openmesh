#include "Cutting.h"
#include "MeshUtils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <unordered_map>
#include <cstdint>

#include <algorithm>
#include <queue>
#include <limits>
#include <iostream>
#include <vector>
static void mark_boundary_vertices(const std::vector<Eigen::Vector3i>& faces,
                                   std::vector<char>& isBoundary)
{
    // 先找出最大顶点索引，确定数组大小
    int maxV = -1;
    for (const auto& tri : faces)
    {
        maxV = std::max(maxV, std::max(tri[0], std::max(tri[1], tri[2])));
    }
    int nV = maxV + 1;
    isBoundary.assign(nV, 0);

    // 用 (min(a,b), max(a,b)) 当作 key 统计每条边被多少个三角形使用
    std::unordered_map<std::uint64_t, int> edgeCount;
    edgeCount.reserve(faces.size() * 3);

    auto make_key = [](int a, int b) -> std::uint64_t
    {
        if (a > b) std::swap(a, b);
        return (static_cast<std::uint64_t>(a) << 32) |
               static_cast<std::uint32_t>(b);
    };

    for (const auto& tri : faces)
    {
        for (int i = 0; i < 3; ++i)
        {
            int a = tri[i];
            int b = tri[(i + 1) % 3];
            auto key = make_key(a, b);
            edgeCount[key] += 1;
        }
    }

    // 使用次数为 1 的 edge 是边界边，它的两个端点就是边界顶点
    for (const auto& kv : edgeCount)
    {
        if (kv.second == 1)
        {
            std::uint64_t key = kv.first;
            int a = static_cast<int>(key >> 32);
            int b = static_cast<int>(key & 0xffffffffu);
            if (a >= 0 && a < nV) isBoundary[a] = 1;
            if (b >= 0 && b < nV) isBoundary[b] = 1;
        }
    }
}
// ---------------------------------------------------------------------
// Helpers: export / import mesh <-> arrays
// ---------------------------------------------------------------------

static void export_mesh_to_arrays(const MyMesh& mesh,
                                  std::vector<Eigen::Vector3d>& vertices,
                                  std::vector<Eigen::Vector3i>& faces)
{
    const int nV = static_cast<int>(mesh.n_vertices());
    const int nF = static_cast<int>(mesh.n_faces());

    vertices.resize(nV);
    for (auto v : mesh.vertices())
    {
        auto p = mesh.point(v);
        vertices[v.idx()] = Eigen::Vector3d(p[0], p[1], p[2]);
    }

    faces.clear();
    faces.reserve(nF);
    for (auto f : mesh.faces())
    {
        std::vector<int> ids;
        for (auto v : mesh.fv_range(f))
            ids.push_back(v.idx());

        if (ids.size() == 3)
            faces.emplace_back(ids[0], ids[1], ids[2]);
        else
            std::cerr << "[cut_mesh] Warning: non-triangular face skipped.\n";
    }
}

static void build_mesh_from_arrays(const std::vector<Eigen::Vector3d>& vertices,
                                   const std::vector<Eigen::Vector3i>& faces,
                                   MyMesh& mesh)
{
    mesh.clear();

    std::vector<MyMesh::VertexHandle> vhandles(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        const Eigen::Vector3d& p = vertices[i];
        vhandles[i] = mesh.add_vertex(MyMesh::Point(
                static_cast<float>(p.x()),
                static_cast<float>(p.y()),
                static_cast<float>(p.z())));
    }

    for (const auto& tri : faces)
    {
        std::vector<MyMesh::VertexHandle> fh(3);
        fh[0] = vhandles[tri[0]];
        fh[1] = vhandles[tri[1]];
        fh[2] = vhandles[tri[2]];

        if (!mesh.add_face(fh).is_valid())
        {
            std::cerr << "[cut_mesh] Warning: failed to add face.\n";
        }
    }

    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_normals();
}

// ---------------------------------------------------------------------
// Helper: build adjacency from triangles (undirected weighted graph)
// ---------------------------------------------------------------------

struct Neighbor
{
    int   vid;
    double w;
};

static void build_vertex_adjacency(const std::vector<Eigen::Vector3d>& vertices,
                                   const std::vector<Eigen::Vector3i>& faces,
                                   std::vector<std::vector<Neighbor>>& adj)
{
    const int nV = static_cast<int>(vertices.size());
    adj.assign(nV, {});

    auto add_edge = [&](int a, int b)
    {
        if (a == b) return;
        double w = (vertices[a] - vertices[b]).norm();
        adj[a].push_back({b, w});
        adj[b].push_back({a, w});
    };

    for (const auto& tri : faces)
    {
        int v0 = tri[0], v1 = tri[1], v2 = tri[2];
        add_edge(v0, v1);
        add_edge(v0, v2);
        add_edge(v1, v2);
    }
}

// ---------------------------------------------------------------------
// Dijkstra: shortest path from source to target (vertex indices)
// ---------------------------------------------------------------------

static std::vector<int> shortest_path(const std::vector<Eigen::Vector3d>& vertices,
                                      const std::vector<Eigen::Vector3i>& faces,
                                      int source,
                                      int target)
{
    std::cout << "[cut_mesh] Computing shortest path from " << source
              << " to " << target << "...\n";

    const int nV = static_cast<int>(vertices.size());
    std::vector<std::vector<Neighbor>> adj;
    build_vertex_adjacency(vertices, faces, adj);

    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(nV, INF);
    std::vector<int>    prev(nV, -1);

    using Node = std::pair<double,int>; // (dist, vid)
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

    dist[source] = 0.0;
    pq.push({0.0, source});

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        if (u == target) break;

        for (const auto& nb : adj[u])
        {
            double nd = d + nb.w;
            if (nd < dist[nb.vid])
            {
                dist[nb.vid] = nd;
                prev[nb.vid] = u;
                pq.push({nd, nb.vid});
            }
        }
    }

    std::vector<int> path;
    if (prev[target] == -1 && source != target)
    {
        std::cerr << "[cut_mesh] WARNING: no path found between cones.\n";
        path.push_back(source);
        path.push_back(target);
        return path;
    }

    for (int v = target; v != -1; v = prev[v])
        path.push_back(v);

    std::reverse(path.begin(), path.end());
    std::cout << "[cut_mesh] Path length (num vertices) = " << path.size() << "\n";
    return path;
}

// ---------------------------------------------------------------------
// Split mesh along a vertex path p[0..k-1]
// (Simplified version of the idea behind TreeCutter::split_mesh_by_path)
// ---------------------------------------------------------------------

static bool tri_contains_vertex(const Eigen::Vector3i& tri, int v)
{
    return (tri[0] == v || tri[1] == v || tri[2] == v);
}

static bool tri_contains_edge(const Eigen::Vector3i& tri, int a, int b)
{
    int count = 0;
    for (int k = 0; k < 3; ++k)
    {
        int vk = tri[k];
        if (vk == a || vk == b) ++count;
    }
    return count == 2;
}

static bool edge_orientation_positive(const Eigen::Vector3i& tri, int a, int b)
{
    // tri = (v0,v1,v2) oriented (0->1->2->0)
    int ia = -1, ib = -1;
    for (int k = 0; k < 3; ++k)
    {
        if (tri[k] == a) ia = k;
        if (tri[k] == b) ib = k;
    }
    if (ia == -1 || ib == -1)
        return false;

    int next = (ia + 1) % 3;
    return (tri[next] == b);
}

static void split_mesh_along_path(
        std::vector<Eigen::Vector3d>& vertices,
        std::vector<Eigen::Vector3i>& faces,
        const std::vector<int>& path,
        std::vector<int>& cutToUncut,
        std::vector<std::vector<int>>& uncutToCut,
        std::vector<std::pair<int,int>>& pathCorr)
{
    std::cout << "[cut_mesh] Splitting mesh along path with "
              << path.size() << " vertices.\n";

    const int nF = static_cast<int>(faces.size());

    // --- 1) For each edge on the path, classify the two incident faces
    //     into left / right sets according to orientation.
    std::vector<int> leftFaces;
    std::vector<int> rightFaces;

    for (size_t j = 0; j + 1 < path.size(); ++j)
    {
        int a = path[j];
        int b = path[j+1];

        // Find triangles that contain edge (a,b)
        std::vector<int> tris;
        for (int fi = 0; fi < nF; ++fi)
        {
            if (tri_contains_edge(faces[fi], a, b))
                tris.push_back(fi);
        }

        if (tris.size() != 2)
        {
            std::cerr << "[cut_mesh] WARNING: edge (" << a << "," << b
                      << ") adjacent to " << tris.size()
                      << " faces (expected 2). Skipping.\n";
            continue;
        }

        int t0 = tris[0];
        int t1 = tris[1];

        if (edge_orientation_positive(faces[t0], a, b))
        {
            leftFaces.push_back(t0);
            rightFaces.push_back(t1);
        }
        else
        {
            leftFaces.push_back(t1);
            rightFaces.push_back(t0);
        }
    }

    auto unique_ints = [](std::vector<int>& v)
    {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    unique_ints(leftFaces);
    unique_ints(rightFaces);

    // --- 2) Grow left/right sets to all triangles touching path vertices,
    //        following adjacency via at-least-two-common-vertices rule.

    std::vector<int> touching;
    {
        std::vector<char> mark(nF, 0);
        for (size_t j = 1; j + 1 < path.size(); ++j)
        {
            int v = path[j];
            for (int fi = 0; fi < nF; ++fi)
            {
                if (!mark[fi] && tri_contains_vertex(faces[fi], v))
                {
                    mark[fi] = 1;
                    touching.push_back(fi);
                }
            }
        }
    }

    auto remove_from = [&](std::vector<int>& vec, const std::vector<int>& rem)
    {
        std::vector<char> toRemove(nF, 0);
        for (int x : rem) toRemove[x] = 1;
        std::vector<int> out;
        out.reserve(vec.size());
        for (int x : vec) if (!toRemove[x]) out.push_back(x);
        vec.swap(out);
    };

    remove_from(touching, leftFaces);
    remove_from(touching, rightFaces);

    std::vector<char> isLeftFace(nF, 0), isRightFace(nF, 0);
    for (int fi : leftFaces)  isLeftFace[fi]  = 1;
    for (int fi : rightFaces) isRightFace[fi] = 1;

    auto shares_two_vertices = [&](int f1, int f2) -> bool
    {
        const auto& t1 = faces[f1];
        const auto& t2 = faces[f2];
        int count = 0;
        for (int i = 0; i < 3; ++i)
        {
            int v = t1[i];
            if (v == t2[0] || v == t2[1] || v == t2[2])
                ++count;
        }
        return count >= 2;
    };

    for (int iter = 0; iter < 1000; ++iter)
    {
        bool changed = false;
        std::vector<int> newRight;
        std::vector<int> newLeft;

        for (int fi : touching)
        {
            bool nearRight = false;
            bool nearLeft  = false;

            for (int rf : rightFaces)
            {
                if (shares_two_vertices(fi, rf))
                {
                    nearRight = true;
                    break;
                }
            }
            for (int lf : leftFaces)
            {
                if (shares_two_vertices(fi, lf))
                {
                    nearLeft = true;
                    break;
                }
            }

            if (nearRight && !nearLeft)
                newRight.push_back(fi);
            else if (nearLeft && !nearRight)
                newLeft.push_back(fi);
        }

        if (!newRight.empty())
        {
            for (int fi : newRight)
            {
                if (!isRightFace[fi])
                {
                    isRightFace[fi] = 1;
                    rightFaces.push_back(fi);
                    changed = true;
                }
            }
        }
        if (!newLeft.empty())
        {
            for (int fi : newLeft)
            {
                if (!isLeftFace[fi])
                {
                    isLeftFace[fi] = 1;
                    leftFaces.push_back(fi);
                    changed = true;
                }
            }
        }

        if (changed)
        {
            unique_ints(leftFaces);
            unique_ints(rightFaces);
            remove_from(touching, leftFaces);
            remove_from(touching, rightFaces);
        }

        if (!changed || touching.empty())
            break;
    }

    std::cout << "[cut_mesh] Left faces:  " << leftFaces.size()  << "\n";
    std::cout << "[cut_mesh] Right faces: " << rightFaces.size() << "\n";

    // --- 3) Duplicate internal path vertices for LEFT side
    pathCorr.clear();

    for (size_t j = 1; j + 1 < path.size(); ++j)
    {
        int vOrig = path[j];
        Eigen::Vector3d pos = vertices[vOrig];
        int newIdx = static_cast<int>(vertices.size());
        vertices.push_back(pos);

        for (int fi : leftFaces)
        {
            auto& tri = faces[fi];
            for (int k = 0; k < 3; ++k)
            {
                if (tri[k] == vOrig)
                    tri[k] = newIdx;
            }
        }

        cutToUncut.push_back(vOrig);
        uncutToCut[vOrig].push_back(newIdx);

        pathCorr.emplace_back(vOrig, newIdx);
    }

    int vEnd = path.back();
    pathCorr.emplace_back(vEnd, vEnd);

    std::cout << "[cut_mesh] Finished splitting along path.\n";
}

// ---------------------------------------------------------------------
// Public function: cut_mesh_along_cones
// ---------------------------------------------------------------------
void cut_mesh_along_cones(MyMesh& mesh,
                          const std::vector<int>& cones_in,
                          std::vector<int>&       cones_out)
{
    std::cout << "\n========== [cut_mesh_along_cones] Start ==========\n";

    if (cones_in.size() != 3)
    {
        std::cerr << "[cut_mesh_along_cones] Currently only supports 3 cones "
                  << "(Type I). Skipping cutting.\n";
        cones_out = cones_in; // 直接原样返回
        std::cout << "========== [cut_mesh_along_cones] End ==========\n\n";
        return;
    }

    // 1. Export mesh to simple arrays
    std::vector<Eigen::Vector3d> V;
    std::vector<Eigen::Vector3i> F;
    export_mesh_to_arrays(mesh, V, F);

    std::cout << "[cut_mesh_along_cones] Original mesh: "
              << V.size() << " vertices, "
              << F.size() << " faces.\n";

    int c0 = cones_in[0];
    int c1 = cones_in[1];
    int c2 = cones_in[2];

    auto check_index = [&](int idx)
    {
        if (idx < 0 || idx >= static_cast<int>(V.size()))
            std::cerr << "[cut_mesh_along_cones] WARNING: cone index "
                      << idx << " out of range.\n";
    };
    check_index(c0);
    check_index(c1);
    check_index(c2);

    // 2. Compute a single path that goes c0 -> c1 -> c2
    auto path01 = shortest_path(V, F, c0, c1);
    auto path12 = shortest_path(V, F, c1, c2);

    std::vector<int> fullPath = path01;
    if (!path12.empty())
        fullPath.insert(fullPath.end(), path12.begin() + 1, path12.end());

    std::cout << "[cut_mesh_along_cones] Full cut path length = "
              << fullPath.size() << "\n";

    // 3. Initialize mappings uncut <-> cut
    std::vector<int> cutToUncut(V.size());
    std::vector<std::vector<int>> uncutToCut(V.size());
    for (int i = 0; i < static_cast<int>(V.size()); ++i)
    {
        cutToUncut[i] = i;
        uncutToCut[i].push_back(i);
    }

    // 4. Split mesh along that path
    std::vector<std::pair<int,int>> pathCorr;
    split_mesh_along_path(V, F, fullPath,
                          cutToUncut, uncutToCut, pathCorr);

    std::cout << "[cut_mesh_along_cones] After cutting (arrays): "
              << V.size() << " vertices, "
              << F.size() << " faces.\n";

    // 5. 在 arrays 上计算哪些顶点是边界顶点
    std::vector<char> isBoundaryVertex;
    mark_boundary_vertices(F, isBoundaryVertex);

    // 6. 根据 cut 之后的拓扑，为每个原始锥点选择一个「对应的切后顶点」
    cones_out.resize(cones_in.size());
    for (size_t k = 0; k < cones_in.size(); ++k)
    {
        int orig = cones_in[k];
        int chosen = -1;

        if (orig >= 0 && orig < static_cast<int>(uncutToCut.size()))
        {
            // 所有由这个原始顶点衍生出来的 cut 顶点
            const auto& cutIndices = uncutToCut[orig];

            // 优先选「在边界上的那个副本」
            for (int ci : cutIndices)
            {
                if (ci >= 0 &&
                    ci < static_cast<int>(isBoundaryVertex.size()) &&
                    isBoundaryVertex[ci])
                {
                    chosen = ci;
                    break;
                }
            }

            // 如果没有在边界上的副本，就随便选第一个（退而求其次）
            if (chosen == -1 && !cutIndices.empty())
                chosen = cutIndices[0];
        }
        else
        {
            std::cerr << "[cut_mesh_along_cones] Cone index "
                      << orig << " has no cut copies.\n";
        }

        if (chosen == -1)
        {
            std::cerr << "[cut_mesh_along_cones] Fallback: using original index "
                      << orig << " as cone.\n";
            chosen = orig;
        }

        cones_out[k] = chosen;
    }

    std::cout << "[cut_mesh_along_cones] Updated cone indices after cut: ";
    for (int idx : cones_out) std::cout << idx << " ";
    std::cout << "\n";

    // 7. Rebuild OpenMesh from arrays
    build_mesh_from_arrays(V, F, mesh);

    std::cout << "[cut_mesh_along_cones] Rebuilt OpenMesh. "
              << mesh.n_vertices() << " vertices, "
              << mesh.n_faces()    << " faces.\n";
    std::cout << "========== [cut_mesh_along_cones] End ==========\n\n";
}