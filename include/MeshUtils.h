#ifndef MESH_UTILS_H
#define MESH_UTILS_H

#include "MeshTypes.h"
#include <Eigen/Sparse>
#include <vector>
#include <cmath> // For M_PI

// 1. Dijkstra Shortest Path
// Finds the shortest path between two vertices in a mesh.
std::vector<MyMesh::VertexHandle> find_shortest_path(
        MyMesh& mesh,
        MyMesh::VertexHandle start_v,
        MyMesh::VertexHandle end_v
);

// 2. Build Cotangent Laplacian Matrix
// Computes the Cotangent Laplacian matrix for the mesh.
void compute_cotangent_laplacian(
        MyMesh& mesh,
        Eigen::SparseMatrix<double>& L
);

// 3. Extract Ordered Boundary Loop
// Extracts the vertices forming the outer boundary loop in order.
bool get_boundary_loop(
        MyMesh& mesh,
        std::vector<MyMesh::VertexHandle>& boundary_loop
);

#endif // MESH_UTILS_H