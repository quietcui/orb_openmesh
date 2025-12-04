#include "MeshUtils.h"
#include <queue>
#include <map>
#include <iostream>
#include <algorithm> // for std::reverse
#include <limits> // for std::numeric_limits

// =========================================================
// 1. Dijkstra Shortest Path (OpenMesh Implementation)
// =========================================================
std::vector<MyMesh::VertexHandle> find_shortest_path(
        MyMesh& mesh,
        MyMesh::VertexHandle start_v,
        MyMesh::VertexHandle end_v
) {
    // Use dynamic properties for distance and predecessor tracking
    OpenMesh::VPropHandleT<double> dist_prop;
    OpenMesh::VPropHandleT<MyMesh::VertexHandle> prev_prop;
    mesh.add_property(dist_prop);
    mesh.add_property(prev_prop);

    // Initialization
    for (auto v_it : mesh.vertices()) {
        mesh.property(dist_prop, v_it) = std::numeric_limits<double>::infinity();
    }
    mesh.property(dist_prop, start_v) = 0.0;

    // Priority Queue: {distance, vertex handle}
    using P = std::pair<double, MyMesh::VertexHandle>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    pq.push({0.0, start_v});

    while(!pq.empty()) {
        double d = pq.top().first;
        MyMesh::VertexHandle u = pq.top().second;
        pq.pop();

        if (d > mesh.property(dist_prop, u)) continue;
        if (u == end_v) break;

        // Iterate over neighbors using Vertex-Vertex Iterator
        for (auto vv_it = mesh.vv_iter(u); vv_it.is_valid(); ++vv_it) {
            MyMesh::VertexHandle v = *vv_it;
            double len = (mesh.point(u) - mesh.point(v)).norm();

            if (mesh.property(dist_prop, u) + len < mesh.property(dist_prop, v)) {
                mesh.property(dist_prop, v) = mesh.property(dist_prop, u) + len;
                mesh.property(prev_prop, v) = u;
                pq.push({mesh.property(dist_prop, v), v});
            }
        }
    }

    // Backtrack path
    std::vector<MyMesh::VertexHandle> path;
    MyMesh::VertexHandle curr = end_v;

    if (mesh.property(dist_prop, end_v) == std::numeric_limits<double>::infinity()) {
        // Cleanup properties
        mesh.remove_property(dist_prop);
        mesh.remove_property(prev_prop);
        return {};
    }

    // Reconstruction
    while (curr.is_valid() && curr != start_v) {
        path.push_back(curr);
        curr = mesh.property(prev_prop, curr);
    }
    if (curr == start_v) path.push_back(start_v);

    std::reverse(path.begin(), path.end());

    // Cleanup properties
    mesh.remove_property(dist_prop);
    mesh.remove_property(prev_prop);

    return path;
}

// =========================================================
// 2. Cotangent Laplacian Matrix (OpenMesh Implementation)
// **FIXED: Ensure N x N matrix structure regardless of boundary status.**
// =========================================================
void compute_cotangent_laplacian(MyMesh& mesh, Eigen::SparseMatrix<double>& L) {
    int n_verts = mesh.n_vertices();
    L.resize(n_verts, n_verts);
    std::vector<Eigen::Triplet<double>> triplets;

    // Iterate over ALL vertices to ensure an N x N matrix structure.
    for (auto v_it : mesh.vertices()) {
        double weight_sum = 0.0;
        int i = v_it.idx();

        // Only calculate weights and off-diagonal entries for INTERNAL vertices
        // Boundary vertices will have a diagonal weight_sum of 0, enforcing the Dirichlet condition implicitly.
        if (!mesh.is_boundary(v_it)) {
            // Iterate over outgoing halfedges
            for (auto voh_it = mesh.voh_iter(v_it); voh_it.is_valid(); ++voh_it) {
                MyMesh::HalfedgeHandle he = *voh_it;
                MyMesh::VertexHandle v_neighbor = mesh.to_vertex_handle(he);
                int j = v_neighbor.idx();

                double weight = 0.0;

                // Calculate Cotangent weights (Alpha + Beta)
                // 1. Alpha Angle (from the face 'left' of he)
                if (!mesh.is_boundary(he)) {
                    MyMesh::HalfedgeHandle he_next = mesh.next_halfedge_handle(he);
                    MyMesh::VertexHandle v_alpha = mesh.to_vertex_handle(he_next);

                    Eigen::Vector3d p1 = to_eigen(mesh.point(v_it));
                    Eigen::Vector3d p2 = to_eigen(mesh.point(v_neighbor));
                    Eigen::Vector3d p3 = to_eigen(mesh.point(v_alpha));

                    Eigen::Vector3d u = p1 - p3;
                    Eigen::Vector3d v = p2 - p3;
                    double cot_alpha = u.dot(v) / (u.cross(v).norm());
                    weight += std::max(0.0, cot_alpha);
                }

                // 2. Beta Angle (from the face 'right' of he, using he_opp)
                MyMesh::HalfedgeHandle he_opp = mesh.opposite_halfedge_handle(he);
                if (!mesh.is_boundary(he_opp)) {
                    MyMesh::HalfedgeHandle he_opp_next = mesh.next_halfedge_handle(he_opp);
                    MyMesh::VertexHandle v_beta = mesh.to_vertex_handle(he_opp_next);

                    Eigen::Vector3d p1 = to_eigen(mesh.point(v_it));
                    Eigen::Vector3d p2 = to_eigen(mesh.point(v_neighbor));
                    Eigen::Vector3d p3 = to_eigen(mesh.point(v_beta));

                    Eigen::Vector3d u = p1 - p3;
                    Eigen::Vector3d v = p2 - p3;
                    double cot_beta = u.dot(v) / (u.cross(v).norm());
                    weight += std::max(0.0, cot_beta);
                }

                weight *= 0.5;

                // Off-diagonal element L[i, j]
                triplets.push_back(Eigen::Triplet<double>(i, j, -weight));
                weight_sum += weight;
            }
        } // End if (!mesh.is_boundary(v_it))

        // Diagonal element L[i, i] MUST be added for ALL vertices.
        triplets.push_back(Eigen::Triplet<double>(i, i, weight_sum));
    }

    L.setFromTriplets(triplets.begin(), triplets.end());
}

// =========================================================
// 3. Extract Ordered Boundary Loop (OpenMesh Implementation)
// =========================================================
bool get_boundary_loop(MyMesh& mesh, std::vector<MyMesh::VertexHandle>& boundary_loop) {
    boundary_loop.clear();

    // 1. Find the first boundary halfedge
    MyMesh::HalfedgeHandle he_start;
    bool found = false;

    for (auto he_it : mesh.halfedges()) {
        if (mesh.is_boundary(he_it)) {
            he_start = he_it;
            found = true;
            break;
        }
    }

    if (!found) return false; // Closed mesh, no boundary

    // 2. Loop along boundary halfedges
    MyMesh::HalfedgeHandle he_curr = he_start;

    size_t max_iter = mesh.n_vertices() * 2;
    size_t count = 0;

    do {
        // The vertex *from* which the halfedge starts is the boundary vertex
        boundary_loop.push_back(mesh.from_vertex_handle(he_curr));

        // Move to the next boundary halfedge
        he_curr = mesh.next_halfedge_handle(he_curr);

        count++;
        if (count > max_iter) {
            std::cerr << "Error: Boundary tracing stuck in infinite loop." << std::endl;
            return false;
        }
    } while (he_curr != he_start);

    return true;
}