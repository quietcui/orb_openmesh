// main.cpp
//
// CLI wrapper around flatten_sphere() (src/Flattening.cpp).
//
// NEW in this version:
//   - --auto : automatically pick cones for closed genus-0 meshes (e.g. a head)
//              using farthest-point sampling on approximate geodesic distances.
//   - --cutonly : only run the cutting stage and write the cut mesh to output.
//
// Rationale: for a different mesh, indices like "50 100 130" usually do NOT
// correspond to meaningful points. If two cones are geometrically close,
// orbifold constraints become ill-conditioned and the flattening collapses.
//
// Examples:
//   (1) Manual cones (3-cone orbifolds):
//     ./orb_openmesh input.obj out.off 1 12 345 678
//
//   (2) MATLAB (1-based) cones:
//     ./orb_openmesh input.obj out.off 1 13 346 679 --1based
//
//   (3) Auto cones (recommended for head / scanned meshes):
//     ./orb_openmesh bimba.obj flat.off 1 --auto
//
//   (4) Only cut to disk and export (for debugging seams):
//     ./orb_openmesh bimba.obj cut.off 1 --auto --cutonly

#include "Flattening.h"
#include "MeshTypes.h"

#include <OpenMesh/Core/IO/MeshIO.hh>

#include <Eigen/Core>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static void print_usage(const char* argv0)
{
    std::cerr
            << "Usage:\n"
            << "  " << argv0 << " <input_mesh> <output_mesh> <orbifold_type> [cones...] [flags]\n\n"
            << "orbifold_type:\n"
            << "  1 = Square\n"
            << "  2 = Diamond\n"
            << "  3 = Triangle\n"
            << "  4 = Parallelogram\n\n"
            << "cones:\n"
            << "  For types 1..3: provide exactly 3 cone vertex indices (0-based by default).\n"
            << "  For type 4:    provide exactly 4 cone vertex indices (0-based by default).\n"
            << "  You may also pass a single token like: 50/100/130\n\n"
            << "flags:\n"
            << "  --1based   Treat given cone indices as 1-based (MATLAB), subtract 1.\n"
            << "  --auto     Ignore / omit cones and auto-pick good cones (recommended for head meshes).\n"
            << "  --cutonly  Only run the cutting stage; write the CUT mesh to output and exit.\n"
            << "  --quiet    Less logging.\n";
}

static inline bool is_flag(const std::string& s)
{
    return !s.empty() && s[0] == '-';
}

static std::vector<int> parse_int_list_token(const std::string& token)
{
    // Allow tokens like "50/100/130" or "50,100,130".
    std::vector<int> out;
    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        out.push_back(std::stoi(cur));
        cur.clear();
    };

    for (char ch : token)
    {
        if (ch == '/' || ch == ',' || ch == ';')
        {
            flush();
        }
        else if (!std::isspace(static_cast<unsigned char>(ch)))
        {
            cur.push_back(ch);
        }
    }
    flush();
    return out;
}

static inline Eigen::Vector3d to_vec3d(const MyMesh::Point& p)
{
    return Eigen::Vector3d(static_cast<double>(p[0]),
                           static_cast<double>(p[1]),
                           static_cast<double>(p[2]));
}

static double bbox_diag(const MyMesh& mesh)
{
    if (mesh.n_vertices() == 0) return 0.0;
    Eigen::Vector3d mn(+std::numeric_limits<double>::infinity(),
                       +std::numeric_limits<double>::infinity(),
                       +std::numeric_limits<double>::infinity());
    Eigen::Vector3d mx(-std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity());
    for (auto vh : mesh.vertices())
    {
        Eigen::Vector3d p = to_vec3d(mesh.point(vh));
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    return (mx - mn).norm();
}

static double dist_euclid(const MyMesh& mesh, int a, int b)
{
    const auto pa = to_vec3d(mesh.point(mesh.vertex_handle(a)));
    const auto pb = to_vec3d(mesh.point(mesh.vertex_handle(b)));
    return (pa - pb).norm();
}

static std::vector<double> dijkstra_geodesic(const MyMesh& mesh, int source)
{
    const int nV = static_cast<int>(mesh.n_vertices());
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(static_cast<size_t>(nV), INF);

    using Node = std::pair<double, int>; // (dist, vid)
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

    dist[source] = 0.0;
    pq.push({0.0, source});

    while (!pq.empty())
    {
        const auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) continue;

        const auto uh = mesh.vertex_handle(u);
        const Eigen::Vector3d pu = to_vec3d(mesh.point(uh));

        for (auto vv_it = mesh.cvv_iter(uh); vv_it.is_valid(); ++vv_it)
        {
            const int v = vv_it->idx();
            const Eigen::Vector3d pv = to_vec3d(mesh.point(*vv_it));
            const double w = (pu - pv).norm();
            const double nd = d + w;
            if (nd < dist[v])
            {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    return dist;
}

static int argmax_finite(const std::vector<double>& dist)
{
    int best = -1;
    double bestVal = -1.0;
    for (int i = 0; i < static_cast<int>(dist.size()); ++i)
    {
        const double d = dist[static_cast<size_t>(i)];
        if (!std::isfinite(d)) continue;
        if (d > bestVal)
        {
            bestVal = d;
            best = i;
        }
    }
    return best;
}

static std::vector<int> auto_pick_cones_fps(const MyMesh& mesh, int k, bool verbose)
{
    const int nV = static_cast<int>(mesh.n_vertices());
    if (nV == 0) return {};
    if (k <= 0) return {};

    // Seed: farthest from centroid (Euclidean). (Cheap and robust.)
    Eigen::Vector3d centroid(0, 0, 0);
    for (auto vh : mesh.vertices()) centroid += to_vec3d(mesh.point(vh));
    centroid /= static_cast<double>(nV);

    int seed = 0;
    double bestS = -1.0;
    for (auto vh : mesh.vertices())
    {
        const int i = vh.idx();
        const double d2 = (to_vec3d(mesh.point(vh)) - centroid).squaredNorm();
        if (d2 > bestS) { bestS = d2; seed = i; }
    }

    if (verbose)
        std::cout << "[auto] seed vertex = " << seed << "\n";

    // Farthest-point sampling using approximate geodesic distances (Dijkstra).
    std::vector<int> cones;
    cones.reserve(static_cast<size_t>(k));

    // First cone: farthest from seed.
    auto dist_seed = dijkstra_geodesic(mesh, seed);
    int c0 = argmax_finite(dist_seed);
    if (c0 < 0) c0 = seed;
    cones.push_back(c0);

    // Maintain min-distance-to-set array.
    std::vector<double> mindist = dijkstra_geodesic(mesh, c0);

    while (static_cast<int>(cones.size()) < k)
    {
        // Next cone = vertex maximizing mindist.
        int nxt = argmax_finite(mindist);
        if (nxt < 0) break;
        // Avoid duplicates (should not happen, but be safe).
        if (std::find(cones.begin(), cones.end(), nxt) != cones.end())
            break;
        cones.push_back(nxt);

        // Update mindist with distances from the new cone.
        auto dist_new = dijkstra_geodesic(mesh, nxt);
        for (int i = 0; i < nV; ++i)
        {
            const double dn = dist_new[static_cast<size_t>(i)];
            double& md = mindist[static_cast<size_t>(i)];
            if (!std::isfinite(md)) md = dn;
            else if (std::isfinite(dn)) md = std::min(md, dn);
        }
    }

    // For 3-cone orbifolds (types 1..3), we want the first two cones to be
    // the farthest pair, because the MATLAB code assigns singularities to
    // cones[0] and cones[1]. The third is the root.
    if (k == 3 && cones.size() == 3)
    {
        int a = cones[0], b = cones[1], c = cones[2];
        const double dab = dist_euclid(mesh, a, b);
        const double dac = dist_euclid(mesh, a, c);
        const double dbc = dist_euclid(mesh, b, c);
        if (dac >= dab && dac >= dbc)      cones = {a, c, b};
        else if (dbc >= dab && dbc >= dac) cones = {b, c, a};
    }

    if (verbose)
    {
        std::cout << "[auto] cones = ";
        for (int ci : cones) std::cout << ci << " ";
        std::cout << "\n";
    }

    return cones;
}

static void warn_if_cones_too_close(const MyMesh& mesh,
                                    const std::vector<int>& cones,
                                    bool verbose)
{
    if (!verbose || cones.size() < 2) return;
    const double diag = bbox_diag(mesh);
    if (diag <= 0) return;

    double minD = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < cones.size(); ++i)
        for (size_t j = i + 1; j < cones.size(); ++j)
            minD = std::min(minD, dist_euclid(mesh, cones[i], cones[j]));

    // Heuristic threshold: cones closer than 5% of bbox diagonal are usually bad.
    const double thr = 0.05 * diag;
    if (minD < thr)
    {
        std::cerr
                << "[warning] Some cones are very close (minDist=" << minD
                << ", bboxDiag=" << diag << ").\n"
                << "          This often causes severe collapse / crowding.\n"
                << "          Consider using --auto to pick better cones.\n";
    }
}

// Convert OpenMesh -> Eigen (V,T) (triangle-only)
static void mesh_to_eigen(const MyMesh& mesh, Eigen::MatrixXd& V, Eigen::MatrixXi& T)
{
    const int nV = static_cast<int>(mesh.n_vertices());
    const int nF = static_cast<int>(mesh.n_faces());
    V.resize(nV, 3);
    T.resize(nF, 3);

    for (auto vh : mesh.vertices())
    {
        const int i = vh.idx();
        auto p = mesh.point(vh);
        V(i, 0) = p[0];
        V(i, 1) = p[1];
        V(i, 2) = p[2];
    }

    int fi = 0;
    for (auto fh : mesh.faces())
    {
        int k = 0;
        for (auto fv_it = mesh.cfv_iter(fh); fv_it.is_valid(); ++fv_it)
        {
            if (k < 3) T(fi, k) = fv_it->idx();
            ++k;
        }
        ++fi;
    }
}

// Build the same cone-tree structure as Flattening.cpp (MATLAB-compatible)
static void build_cone_tree(int k, Eigen::MatrixXi& treeAdj, int& treeRoot)
{
    treeAdj = Eigen::MatrixXi::Zero(k, k);
    treeRoot = 0;
    if (k == 3)
    {
        // MATLAB root=3 (1-based) => C++ root=2.
        treeRoot = 2;
        treeAdj(2, 0) = treeAdj(0, 2) = 1;
        treeAdj(2, 1) = treeAdj(1, 2) = 1;
    }
    else if (k == 4)
    {
        treeRoot = 0;
        auto add_e = [&](int a, int b) {
            treeAdj(a, b) = 1;
            treeAdj(b, a) = 1;
        };
        // fixedPairs = [1 3;3 4;4 2] in MATLAB
        add_e(0, 2);
        add_e(2, 3);
        add_e(3, 1);
    }
}

// Cut-only pipeline: run TreeCutter and export the cut mesh (3D)
static bool cut_only_export(const MyMesh& mesh,
                            const std::vector<int>& cones,
                            int orbifold_type,
                            const std::string& out_path,
                            bool verbose)
{
    (void)orbifold_type; // tree depends on #cones, not type, in our current implementation

    Eigen::MatrixXd V;
    Eigen::MatrixXi T;
    mesh_to_eigen(mesh, V, T);

    const int k = static_cast<int>(cones.size());
    Eigen::MatrixXi treeAdj;
    int treeRoot = 0;
    build_cone_tree(k, treeAdj, treeRoot);

    if (verbose)
        std::cout << "[cutonly] Cutting mesh along cone-tree...\n";

    TreeCutter cutter(V, T, treeAdj, cones, treeRoot);
    cutter.cutTree();
    CutMesh cm = cutter.getCutMesh();

    if (verbose)
    {
        std::cout << "[cutonly] Cut mesh: "
                  << cm.V.rows() << " vertices, "
                  << cm.T.rows() << " faces, "
                  << cm.pathPairs.size() << " seam(s)\n";
    }

    // Convert cut mesh to OpenMesh and write.
    MyMesh out;
    std::vector<MyMesh::VertexHandle> vhs(cm.V.rows());
    for (int i = 0; i < cm.V.rows(); ++i)
    {
        vhs[i] = out.add_vertex(MyMesh::Point(
                static_cast<float>(cm.V(i, 0)),
                static_cast<float>(cm.V(i, 1)),
                static_cast<float>(cm.V(i, 2))));
    }
    for (int fi = 0; fi < cm.T.rows(); ++fi)
    {
        const int a = cm.T(fi, 0);
        const int b = cm.T(fi, 1);
        const int c = cm.T(fi, 2);
        if (a < 0 || b < 0 || c < 0 ||
            a >= cm.V.rows() || b >= cm.V.rows() || c >= cm.V.rows())
            continue;
        out.add_face({vhs[a], vhs[b], vhs[c]});
    }

    if (!OpenMesh::IO::write_mesh(out, out_path))
    {
        std::cerr << "[cutonly] Error: cannot write cut mesh: " << out_path << "\n";
        return false;
    }

    if (verbose)
        std::cout << "[cutonly] Wrote cut mesh to: " << out_path << "\n";
    return true;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        print_usage(argv[0]);
        return 1;
    }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    const int orbifold_type    = std::stoi(argv[3]);

    bool oneBased = false;
    bool verbose  = true;
    bool autoCones = false;
    bool cutOnly  = false;

    // Extract cone indices from argv[4..] (skip flags).
    std::vector<int> cones;
    for (int i = 4; i < argc; ++i)
    {
        const std::string s = argv[i];
        if (s == "--1based") { oneBased = true; continue; }
        if (s == "--quiet")  { verbose = false; continue; }
        if (s == "--auto")   { autoCones = true; continue; }
        if (s == "--cutonly") { cutOnly = true; continue; }
        if (is_flag(s))
        {
            std::cerr << "Warning: unknown flag: " << s << "\n";
            continue;
        }

        // Cone token.
        const auto vals = parse_int_list_token(s);
        cones.insert(cones.end(), vals.begin(), vals.end());
    }

    const int needed = (orbifold_type == 4) ? 4 : 3;

    // If cones are omitted, require --auto.
    if (!autoCones && static_cast<int>(cones.size()) != needed)
    {
        std::cerr << "Error: orbifold_type=" << orbifold_type
                  << " requires " << needed << " cone indices (got " << cones.size() << ").\n"
                  << "       Either provide the indices, or run with --auto.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Read mesh.
    MyMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, in_path))
    {
        std::cerr << "Error: cannot read mesh: " << in_path << "\n";
        return 1;
    }

    if (oneBased && !autoCones)
    {
        for (int& c : cones) c -= 1;
    }

    // Auto-pick cones if requested.
    if (autoCones)
    {
        if (verbose)
            std::cout << "[auto] Selecting " << needed << " cones for this mesh...\n";
        cones = auto_pick_cones_fps(mesh, needed, verbose);
        if (static_cast<int>(cones.size()) != needed)
        {
            std::cerr << "Error: auto cone selection failed (got " << cones.size() << ").\n";
            return 1;
        }
    }

    // Basic sanity: warn if cones are too close.
    warn_if_cones_too_close(mesh, cones, verbose);

    // Option: cut-only.
    if (cutOnly)
    {
        if (!cut_only_export(mesh, cones, orbifold_type, out_path, verbose))
            return 1;
        return 0;
    }

    // Full pipeline: cut + flatten.
    try
    {
        flatten_sphere(mesh, cones, orbifold_type, verbose);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Flattening failed: " << e.what() << "\n";
        return 1;
    }

    if (!OpenMesh::IO::write_mesh(mesh, out_path))
    {
        std::cerr << "Error: cannot write mesh: " << out_path << "\n";
        return 1;
    }

    if (verbose)
    {
        std::cout << "Wrote mesh to: " << out_path << "\n";
        std::cout << "Cones used (0-based): ";
        for (int c : cones) std::cout << c << " ";
        std::cout << "\n";
    }

    return 0;
}
