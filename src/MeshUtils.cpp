#include "MeshUtils.h"
#include <algorithm>
#include <set>
#include <queue>
#include <map>
#include <limits>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <cmath>
#include <utility>

// ====================================================================
// 实用函数 (I/O 和 Dijkstra) - 保持不变
// ====================================================================

double edge_length(const Eigen::MatrixXd& V, int i, int j) {
    return (V.row(i) - V.row(j)).norm();
}

std::vector<int> get_vertex_neighbors(const Eigen::MatrixXi& T, int vertex_index) {
    std::set<int> neighbors;
    int target_index = vertex_index;

    for (int i = 0; i < T.rows(); ++i) {
        bool contains_v = false;
        for (int j = 0; j < 3; ++j) {
            if (T(i, j) == target_index) {
                contains_v = true;
                break;
            }
        }

        if (contains_v) {
            for (int j = 0; j < 3; ++j) {
                int neighbor = T(i, j);
                if (neighbor != target_index) {
                    neighbors.insert(neighbor);
                }
            }
        }
    }
    return std::vector<int>(neighbors.begin(), neighbors.end());
}

std::vector<int> find_shortest_path(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        int start_v,
        int end_v
) {
    using P = std::pair<double, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    std::map<int, double> distances;
    std::map<int, int> previous;

    for (int i = 0; i < V.rows(); ++i) {
        distances[i] = std::numeric_limits<double>::infinity();
    }
    distances[start_v] = 0.0;
    pq.push({0.0, start_v});

    while (!pq.empty()) {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > distances[u]) continue;
        if (u == end_v) break;

        std::vector<int> neighbors = get_vertex_neighbors(T, u);

        for (int v : neighbors) {
            double weight = edge_length(V, u, v);
            double new_dist = distances[u] + weight;

            if (new_dist < distances[v]) {
                distances[v] = new_dist;
                previous[v] = u;
                pq.push({new_dist, v});
            }
        }
    }

    std::vector<int> path;
    int current = end_v;
    while (previous.count(current)) {
        path.push_back(current);
        current = previous[current];
    }

    if (current == start_v) {
        path.push_back(start_v);
        std::reverse(path.begin(), path.end());
    } else {
        return {};
    }

    return path;
}

// ====================================================================
// 几何辅助函数：识别网格的边界顶点 - 保持不变
// ====================================================================
std::set<int> find_boundary_vertices(const Eigen::MatrixXi& T, int num_vertices) {
    std::map<std::pair<int, int>, int> edge_counts;

    for (int f = 0; f < T.rows(); ++f) {
        for (int i = 0; i < 3; ++i) {
            int v1 = T(f, i);
            int v2 = T(f, (i + 1) % 3);

            if (v1 > v2) std::swap(v1, v2);
            edge_counts[{v1, v2}]++;
        }
    }

    std::set<int> boundary_vertices;

    for (int f = 0; f < T.rows(); ++f) {
        for (int i = 0; i < 3; ++i) {
            int v_curr = T(f, i);
            int v_next = T(f, (i + 1) % 3);

            int v1 = v_curr;
            int v2 = v_next;
            if (v1 > v2) std::swap(v1, v2);

            if (edge_counts[{v1, v2}] == 1) {
                boundary_vertices.insert(v_curr);
                boundary_vertices.insert(v_next);
            }
        }
    }

    return boundary_vertices;
}


// ====================================================================
// 几何辅助函数：计算三角形内角 - 保持不变
// ====================================================================

double angle_at_vertex(const Eigen::MatrixXd& V, int v1_idx, int v2_idx, int v3_idx) {
    Eigen::Vector3d P1 = V.row(v1_idx).transpose();
    Eigen::Vector3d P2 = V.row(v2_idx).transpose();
    Eigen::Vector3d P3 = V.row(v3_idx).transpose();

    Eigen::Vector3d V21 = P2 - P1;
    Eigen::Vector3d V31 = P3 - P1;

    double dot = V21.dot(V31);
    double len21 = V21.norm();
    double len31 = V31.norm();

    if (len21 == 0.0 || len31 == 0.0) return M_PI / 3.0;

    double cos_angle = dot / (len21 * len31);

    if (cos_angle > 1.0) cos_angle = 1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;

    return std::acos(cos_angle);
}

// ====================================================================
// Laplacian 矩阵构建 (新增负权重钳制)
// ====================================================================

void compute_cotangent_laplacian(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        Eigen::SparseMatrix<double>& L
) {
    int N = V.rows();
    int Nf = T.rows();

    typedef Eigen::Triplet<double> T_triplet;
    std::vector<T_triplet> tripletList;
    tripletList.reserve(Nf * 9);

    for (int f = 0; f < Nf; ++f) {
        int v0 = T(f, 0);
        int v1 = T(f, 1);
        int v2 = T(f, 2);

        int indices[] = {v0, v1, v2};

        for (int i = 0; i < 3; ++i) {
            int v_curr = indices[i];
            int v_next = indices[(i + 1) % 3];
            int v_opp = indices[(i + 2) % 3];

            double alpha = angle_at_vertex(V, v_opp, v_curr, v_next);

            double cot_alpha = std::tan(M_PI / 2.0 - alpha);

            // 🌟 关键修复：钳制 Cotangent 权重 🌟
            // 避免负权重导致非正定矩阵，这是 LDLT 分解失败的常见原因。
            double clamped_cot_alpha = std::max(0.0, cot_alpha);

            double weight = clamped_cot_alpha * 0.5;

            // L(i, j) = -w_ij
            tripletList.push_back(T_triplet(v_curr, v_next, -weight));
            tripletList.push_back(T_triplet(v_next, v_curr, -weight));

            // L(i, i) += w_ij
            tripletList.push_back(T_triplet(v_curr, v_curr, weight));
            tripletList.push_back(T_triplet(v_next, v_next, weight));
        }
    }

    L.resize(N, N);
    L.setFromTriplets(tripletList.begin(), tripletList.end());
    L.makeCompressed();

    std::cout << "DEBUG: Cotangent Laplacian successfully computed and assembled." << std::endl;
}