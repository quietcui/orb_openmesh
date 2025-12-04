#include "Flattening.h"
#include "MeshUtils.h"
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>

// ====================================================================
// Helper: Generate Cut Mesh (OpenMesh Version)
// Splits vertices along the shortest paths to create the cut mesh boundary.
// ====================================================================
void generate_cut_mesh_openmesh(
        MyMesh& original_mesh,
        const std::vector<std::vector<MyMesh::VertexHandle>>& paths,
        MyMesh& cut_mesh,
        std::map<MyMesh::VertexHandle, MyMesh::VertexHandle>& cut_to_orig_map
) {
    std::cout << "\n--- Sub-phase: Generating Cut Mesh Structure (OpenMesh) ---" << std::endl;

    cut_mesh.clear();
    cut_to_orig_map.clear();

    OpenMesh::VPropHandleT<bool> is_on_cut;
    original_mesh.add_property(is_on_cut);
    for (auto v : original_mesh.vertices()) {
        original_mesh.property(is_on_cut, v) = false;
    }

    // Mark interior path vertices for splitting (excluding cones, which are endpoints)
    for(const auto& path : paths) {
        // Skip first and last elements (cones)
        for(size_t i = 1; i < path.size() - 1; ++i) {
            original_mesh.property(is_on_cut, path[i]) = true;
        }
    }

    // Collect all path vertices (including cones)
    std::set<MyMesh::VertexHandle> all_path_vertices;
    for(const auto& path : paths) {
        for(auto v : path) all_path_vertices.insert(v);
    }

    std::map<MyMesh::VertexHandle, std::vector<MyMesh::VertexHandle>> orig_to_cut_v_map;

    // Create new vertices in the cut mesh
    for (auto v_it : original_mesh.vertices()) {
        int num_copies = original_mesh.property(is_on_cut, v_it) ? 2 : 1;

        // Ensure path endpoints (cones) are not duplicated if the marking logic missed them
        if(all_path_vertices.count(v_it) && !original_mesh.property(is_on_cut, v_it)) {
            num_copies = 1;
        }

        for (int k = 0; k < num_copies; ++k) {
            MyMesh::VertexHandle new_v = cut_mesh.add_vertex(original_mesh.point(v_it));
            orig_to_cut_v_map[v_it].push_back(new_v);
            cut_to_orig_map[new_v] = v_it;
        }
    }

    // Create new faces in the cut mesh
    for (auto f_it : original_mesh.faces()) {
        std::vector<MyMesh::VertexHandle> face_v_handles;
        std::vector<MyMesh::VertexHandle> orig_face_v;

        for (auto fv_it = original_mesh.fv_iter(f_it); fv_it.is_valid(); ++fv_it) {
            orig_face_v.push_back(*fv_it);
        }

        // Determine which copy of the split vertices to use for this face
        int on_path_count = 0;
        for(auto v : orig_face_v) {
            if(all_path_vertices.count(v)) on_path_count++;
        }

        // Simple heuristic: if two path vertices are on the face, use the second copy.
        bool use_second_copy = (on_path_count >= 2);

        for (auto v_orig : orig_face_v) {
            const auto& copies = orig_to_cut_v_map[v_orig];

            if (copies.size() == 1) {
                face_v_handles.push_back(copies[0]);
            } else {
                // Determine which copy of the split vertex to assign to the new face
                face_v_handles.push_back(use_second_copy ? copies[1] : copies[0]);
            }
        }

        // Only add the face if it's valid (i.e., not a degenerate face after copying)
        if (face_v_handles.size() >= 3) {
            cut_mesh.add_face(face_v_handles);
        }
    }

    original_mesh.remove_property(is_on_cut);
    std::cout << "Cut Mesh generated. New Vertices: " << cut_mesh.n_vertices() << std::endl;
}

// ====================================================================
// Core Function: Flatten Sphere (Fixed Partitioning Logic)
// ====================================================================
Eigen::MatrixXd flatten_sphere_openmesh(
        MyMesh& mesh,
        const std::vector<int>& cones_indices
) {
    // 1. Compute cut paths
    std::cout << "\n--- Phase 1: Computing Cut Paths (Dijkstra) ---" << std::endl;
    std::vector<MyMesh::VertexHandle> cones;
    for(int idx : cones_indices) {
        if (idx >= 0 && idx < (int)mesh.n_vertices()) {
            cones.push_back(mesh.vertex_handle(idx));
        }
    }

    if (cones.size() < 3) {
        std::cerr << "Error: Need at least 3 cone vertices." << std::endl;
        return Eigen::MatrixXd(0, 0);
    }

    std::vector<std::vector<MyMesh::VertexHandle>> all_paths;
    all_paths.push_back(find_shortest_path(mesh, cones[0], cones[1]));
    all_paths.push_back(find_shortest_path(mesh, cones[1], cones[2]));
    all_paths.push_back(find_shortest_path(mesh, cones[2], cones[0]));

    if (all_paths[0].empty() || all_paths[1].empty() || all_paths[2].empty()) {
        std::cerr << "Error: Could not compute all shortest paths." << std::endl;
        return Eigen::MatrixXd(0, 0);
    }

    // 2. Generate CutMesh
    MyMesh cut_mesh;
    std::map<MyMesh::VertexHandle, MyMesh::VertexHandle> cut_to_orig_map;
    generate_cut_mesh_openmesh(mesh, all_paths, cut_mesh, cut_to_orig_map);

    // 3. Build Laplacian
    std::cout << "\n--- Phase 3: Building and Solving Linear System ---" << std::endl;
    Eigen::SparseMatrix<double> L_cut;
    compute_cotangent_laplacian(cut_mesh, L_cut);

    // 4. Extract Boundary and setup dimensions
    std::vector<MyMesh::VertexHandle> boundary_loop;
    if (!get_boundary_loop(cut_mesh, boundary_loop)) {
        std::cerr << "Error: No boundary detected in the cut mesh. Cutting might have failed." << std::endl;
        return Eigen::MatrixXd(0, 0);
    }

    const int N_BOUNDARY_LOOP = boundary_loop.size(); // e.g., 8 (may contain duplicates)
    int n_total = cut_mesh.n_vertices(); // e.g., 504

    std::cout << "Boundary Loop extracted successfully, size: " << N_BOUNDARY_LOOP << " vertices." << std::endl;
    std::cout << "Total Vertices: " << n_total << std::endl;

    if (N_BOUNDARY_LOOP == 0 || n_total <= N_BOUNDARY_LOOP) {
        std::cerr << "Consistency Error: Fixed or Free vertex count is zero or negative." << std::endl;
        return Eigen::MatrixXd(0, 0);
    }

    // 5. Setup Index Mapping (Refined to use UNIQUE fixed vertices)
    std::vector<int> map_to_free(n_total, -1);
    std::vector<int> map_to_fixed(n_total, -1);
    std::vector<int> free_indices; // Stores the original index of free vertices

    // 5a. Determine the unique set of fixed vertices and map them to [0, N_UNIQUE_FIXED-1]
    std::vector<MyMesh::VertexHandle> unique_fixed_boundary;
    std::set<MyMesh::VertexHandle> fixed_set;

    // Iterate through boundary_loop to maintain order for arc-length, but only add unique handles to the set
    for(const auto& vh : boundary_loop) {
        // Use the VertexHandle itself for comparison, which relies on the internal index
        if(fixed_set.find(vh) == fixed_set.end()) {
            fixed_set.insert(vh);
            unique_fixed_boundary.push_back(vh);
        }
    }

    const int N_UNIQUE_FIXED = unique_fixed_boundary.size(); // e.g., 6

    // Map the unique fixed vertices to their new sequential index
    for(int i=0; i<N_UNIQUE_FIXED; ++i) {
        map_to_fixed[unique_fixed_boundary[i].idx()] = i; // Map to index 0 to N_UNIQUE_FIXED-1
    }

    // 5b. Determine free vertices and map them to [0, N_FREE-1]
    int free_counter = 0;
    for(int i=0; i<n_total; ++i) {
        // Check if the vertex with original index 'i' is NOT in the unique fixed set
        if(map_to_fixed[i] == -1) {
            map_to_free[i] = free_counter++; // Map to index 0 to N_FREE-1
            free_indices.push_back(i);
        }
    }

    // Use the actual count for the system size.
    const int N_FREE = free_counter; // e.g., 498

    // Check for potential issues (now using unique counts for robustness)
    int expected_n_free = n_total - N_UNIQUE_FIXED;
    if (N_FREE != expected_n_free) {
        // This should not happen if map_to_fixed is correctly populated from all boundary vertices
        std::cerr << "Warning: Internal consistency check failed after unique fixed count. Expected Free: " << expected_n_free
                  << ", Actual Counted Free: " << N_FREE << ". Proceeding with N_FREE=" << N_FREE << " and N_UNIQUE_FIXED=" << N_UNIQUE_FIXED << "." << std::endl;
    }

    std::cout << "Unique Fixed Vertices: " << N_UNIQUE_FIXED << ", Free Vertices: " << N_FREE << std::endl;

    // 6. Set Boundary Constraints (Arc-length parameterization)
    // X_fixed will be built using the ordered list of UNIQUE boundary vertices (unique_fixed_boundary)

    // Recalculate total length using the full (possibly duplicate) boundary loop to ensure
    // the arc-length parameterization spans the entire cut boundary length.
    std::vector<double> edge_lengths;
    double total_length = 0.0;
    for (size_t i = 0; i < N_BOUNDARY_LOOP; ++i) {
        MyMesh::VertexHandle v_curr = boundary_loop[i];
        MyMesh::VertexHandle v_next = boundary_loop[(i + 1) % N_BOUNDARY_LOOP];
        double len = (cut_mesh.point(v_curr) - cut_mesh.point(v_next)).norm();
        edge_lengths.push_back(len);
        total_length += len;
    }

    // Now, create the fixed coordinates X_fixed based on the UNIQUE fixed vertices
    Eigen::MatrixXd X_fixed(N_UNIQUE_FIXED, 2);
    std::map<MyMesh::VertexHandle, double> boundary_param;
    double current_len = 0.0;

    // Calculate arc-length parameter for every point in the full boundary loop
    for (size_t i = 0; i < N_BOUNDARY_LOOP; ++i) {
        MyMesh::VertexHandle v_curr = boundary_loop[i];
        double ratio = (total_length > 1e-9) ? (current_len / total_length) : 0.0;

        // Only set the parameter if this is the first time we see this unique vertex
        // This ensures a stable position for the vertex in the unique set.
        if (boundary_param.find(v_curr) == boundary_param.end()) {
            boundary_param[v_curr] = ratio;
        }

        current_len += edge_lengths[i];
    }

    // Assign the calculated parameter to the X_fixed matrix using the unique ordering
    for (int i = 0; i < N_UNIQUE_FIXED; ++i) {
        MyMesh::VertexHandle vh = unique_fixed_boundary[i];
        double angle = 2.0 * M_PI * boundary_param[vh];
        X_fixed(i, 0) = std::cos(angle);
        X_fixed(i, 1) = std::sin(angle);
    }

    // 7. Partition Matrix L_cut and Solve
    std::vector<Eigen::Triplet<double>> L_inner_triplets;
    std::vector<Eigen::Triplet<double>> L_ib_triplets;

    for (int k = 0; k < L_cut.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L_cut, k); it; ++it) {
            int row = it.row(); // Original row index (0 to n_total-1)
            int col = k; // Original col index (0 to n_total-1) - **FIXED: Use k for column index**
            double val = it.value();

            // We are only interested in rows corresponding to FREE vertices (L_inner and L_ib part)
            if (map_to_free[row] != -1) {
                int r_idx = map_to_free[row]; // New row index (0 to N_FREE-1)

                // Partition columns
                if (map_to_free[col] != -1) { // Column is also free -> L_inner
                    L_inner_triplets.push_back(Eigen::Triplet<double>(r_idx, map_to_free[col], val));
                } else if (map_to_fixed[col] != -1) { // Column is fixed -> L_ib
                    // c_idx now refers to the index in the UNIQUE fixed set (0 to N_UNIQUE_FIXED-1)
                    int c_idx = map_to_fixed[col];
                    L_ib_triplets.push_back(Eigen::Triplet<double>(r_idx, c_idx, val));
                }
            }
        }
    }

    // Define matrices using the corrected partitioned sizes
    Eigen::SparseMatrix<double> L_inner(N_FREE, N_FREE);
    Eigen::SparseMatrix<double> L_ib(N_FREE, N_UNIQUE_FIXED); // e.g., 498x6

    L_inner.setFromTriplets(L_inner_triplets.begin(), L_inner_triplets.end());
    L_ib.setFromTriplets(L_ib_triplets.begin(), L_ib_triplets.end());

    // Regularization
    for(int i=0; i<N_FREE; ++i) L_inner.coeffRef(i, i) += 1e-12;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(L_inner);
    if(solver.info() != Eigen::Success) {
        std::cerr << "Solver failed to decompose L_inner. Check for near-singularity." << std::endl;
        return Eigen::MatrixXd(0,0);
    }

    // Solve L_inner * X_inner = -L_ib * X_fixed
    Eigen::MatrixXd rhs = -L_ib * X_fixed;
    Eigen::MatrixXd X_inner = solver.solve(rhs);

    // 8. Assemble and Update CutMesh Coordinates
    Eigen::MatrixXd V_flat(n_total, 2);
    // Insert free vertices
    for(int i=0; i<N_FREE; ++i) V_flat.row(free_indices[i]) = X_inner.row(i);
    // Insert fixed vertices
    for(int i=0; i<N_UNIQUE_FIXED; ++i) V_flat.row(unique_fixed_boundary[i].idx()) = X_fixed.row(i);

    // Update the cut_mesh points
    for (int i = 0; i < n_total; ++i) {
        MyMesh::Point p;
        p[0] = V_flat(i, 0);
        p[1] = V_flat(i, 1);
        p[2] = 0.0; // Flattening to the XY plane
        cut_mesh.set_point(cut_mesh.vertex_handle(i), p);
    }

    mesh = cut_mesh;

    return V_flat;
}