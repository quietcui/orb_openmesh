#include "Flattening.h"
#include "MeshUtils.h"
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <tuple>

// 注意: MeshUtils.h 中应该包含 compute_cotangent_laplacian 和 find_boundary_vertices 的声明

// ====================================================================
// 辅助函数：生成切割网格结构
// ====================================================================

// 根据最短路径 (paths) 在网格 V 和 T 上生成切割后的网格 (CutMesh) 结构。
// 它通过在切割边上的内部顶点创建顶点副本 (copies) 来实现。
CutMesh generate_cut_mesh(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        const std::vector<std::vector<int>>& paths
) {
    std::cout << "\n--- Sub-Stage: Generating Cut Mesh Structure ---" << std::endl;

    // 识别切割路径上的内部顶点（不包括端点）
    std::set<int> vertices_on_cut;
    for (const auto& path : paths) {
        // 从第二个顶点开始，到倒数第二个顶点结束，因为端点是锥点，保持不复制
        for (size_t i = 1; i < path.size() - 1; ++i) {
            vertices_on_cut.insert(path[i]);
        }
    }

    std::vector<Eigen::Vector3d> new_V_coords;
    std::vector<int> cut_to_uncut;
    // 存储原始索引到新索引的映射
    std::vector<std::vector<int>> uncut_to_cut_map(V.rows());
    int current_cut_index = 0;

    // 1. 创建新顶点列表 V'
    for (int i = 0; i < V.rows(); ++i) {
        // 切割边上的内部顶点复制两次，其他顶点复制一次
        int num_copies = (vertices_on_cut.count(i) > 0) ? 2 : 1;

        for (int k = 0; k < num_copies; ++k) {
            new_V_coords.push_back(V.row(i).transpose());
            cut_to_uncut.push_back(i);
            uncut_to_cut_map[i].push_back(current_cut_index);
            current_cut_index++;
        }
    }

    // 2. 创建新面片拓扑 T'
    Eigen::MatrixXi new_T(T.rows(), T.cols());

    // 快速查找属于切割路径的顶点集合
    std::set<int> path_vertices_set;
    if (!paths.empty()) {
        for (const auto& path : paths) {
            path_vertices_set.insert(path.begin(), path.end());
        }
    }

    for (int f = 0; f < T.rows(); ++f) {
        int v0_uncut = T(f, 0);
        int v1_uncut = T(f, 1);
        int v2_uncut = T(f, 2);

        // 默认使用第一个副本的索引
        int v0_cut = uncut_to_cut_map[v0_uncut][0];
        int v1_cut = uncut_to_cut_map[v1_uncut][0];
        int v2_cut = uncut_to_cut_map[v2_uncut][0];

        // 检查该面片是否跨越切割路径（即它是否有两条边在切割路径上）
        int path_count = (path_vertices_set.count(v0_uncut) > 0) +
                         (path_vertices_set.count(v1_uncut) > 0) +
                         (path_vertices_set.count(v2_uncut) > 0);

        // 只有当面片跨越切割边时，才需要使用第二个副本。
        if (path_count >= 2) {
            // 尝试切换到第二个副本
            if (vertices_on_cut.count(v0_uncut) && uncut_to_cut_map[v0_uncut].size() >= 2) {
                v0_cut = uncut_to_cut_map[v0_uncut][1];
            }
            if (vertices_on_cut.count(v1_uncut) && uncut_to_cut_map[v1_uncut].size() >= 2) {
                v1_cut = uncut_to_cut_map[v1_uncut][1];
            }
            if (vertices_on_cut.count(v2_uncut) && uncut_to_cut_map[v2_uncut].size() >= 2) {
                v2_cut = uncut_to_cut_map[v2_uncut][1];
            }
        }

        new_T(f, 0) = v0_cut;
        new_T(f, 1) = v1_cut;
        new_T(f, 2) = v2_cut;
    }


    // 3. 构建路径对 (Path Pairs) 用于 CutMesh 结构
    std::vector<std::vector<std::pair<int, int>>> pathPairs_out;
    for (const auto& path : paths) {
        std::vector<std::pair<int, int>> current_pair;
        for (int uncut_idx : path) {
            const auto& cut_indices = uncut_to_cut_map[uncut_idx];

            if (cut_indices.size() >= 2) {
                // 内部点有两个副本
                current_pair.push_back({cut_indices[0], cut_indices[1]});
            } else {
                // 边界锥点只有一个副本
                int index = cut_indices[0];
                current_pair.push_back({index, index});
            }
        }
        pathPairs_out.push_back(current_pair);
    }

    // 4. 组装结果
    CutMesh result;
    result.V.resize(new_V_coords.size(), 3);
    for (size_t i = 0; i < new_V_coords.size(); ++i) {
        result.V.row(i) = new_V_coords[i].transpose();
    }
    result.T = new_T;
    result.cutIndsToUncutInds = cut_to_uncut;
    result.uncutIndsToCutInds = uncut_to_cut_map;
    result.pathPairs = pathPairs_out;

    std::cout << "New vertices (V'): " << result.V.rows() << " (+" << result.V.rows() - V.rows() << " copies)" << std::endl;
    return result;
}

// ====================================================================
// 核心函数：平展球体 (包含正则化和归一化)
// ====================================================================

// 执行共形平展算法 (Orbifold Flattening)
Eigen::MatrixXd flatten_sphere(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& T,
        const std::vector<int>& cones,
        int orbifold_type,
        CutMesh& cutMesh
) {
    // Stage 1: 计算切割路径 (使用 MeshUtils 中的 find_shortest_path)
    std::cout << "\n--- Stage 1: Calculating Cut Graph (Shortest Paths) ---" << std::endl;

    // 将锥点索引从 1-based 转换为 0-based
    std::vector<int> c_cones_0based = cones;
    for(int& c : c_cones_0based) c = c - 1;

    std::vector<std::vector<int>> all_paths;
    if (c_cones_0based.size() >= 3) {
        // 假设切割路径连接前三个锥点 (1->2, 2->3, 3->1)
        all_paths.push_back(find_shortest_path(V, T, c_cones_0based[0], c_cones_0based[1]));
        all_paths.push_back(find_shortest_path(V, T, c_cones_0based[1], c_cones_0based[2]));
        all_paths.push_back(find_shortest_path(V, T, c_cones_0based[2], c_cones_0based[0]));

        std::cout << "Path (1->2) length: " << all_paths[0].size() << " vertices." << std::endl;
    } else {
        std::cerr << "Error: Not enough cones to form a cut (need at least 3)." << std::endl;
        return Eigen::MatrixXd::Zero(0, 2);
    }

    // Stage 2: 生成 CutMesh 对象
    cutMesh = generate_cut_mesh(V, T, all_paths);

    // Stage 3: 构建和求解稀疏线性系统
    std::cout << "\n--- Stage 3: Building and Solving Linear System ---" << std::endl;

    int N_cut = cutMesh.V.rows();
    Eigen::SparseMatrix<double> L_cut;

    // 1. 构建 Laplacian 矩阵 (使用 V' 和 T')
    compute_cotangent_laplacian(cutMesh.V, cutMesh.T, L_cut);
    std::cout << "DEBUG: Cotangent Laplacian successfully computed and assembled." << std::endl;

    // 2. 识别边界顶点 (使用 MeshUtils 中的 find_boundary_vertices)
    std::set<int> boundary_set = find_boundary_vertices(cutMesh.T, N_cut);

    Eigen::MatrixXd V_flat_result;
    V_flat_result.resize(N_cut, 2);

    if (boundary_set.empty()) {
        std::cerr << "Error: No boundary vertices found! Cannot constrain the system." << std::endl;
        return Eigen::MatrixXd::Zero(N_cut, 2);
    }

    // =======================================================
    // 🌟 边界顶点排序 (Boundary Traversal) - 查找最长组件 🌟
    // =======================================================

    // 目标：找到最长的闭合边界环 (longest_loop)，用于弧长参数化。

    std::vector<int> longest_loop;

    // --- 边界追踪逻辑 (寻找最长闭合环) ---
    {
        // 1. 构建边界边连接图
        std::map<int, std::set<int>> boundary_adj_map;
        for (int f = 0; f < cutMesh.T.rows(); ++f) {
            int v[3] = {cutMesh.T(f, 0), cutMesh.T(f, 1), cutMesh.T(f, 2)};
            for (int i = 0; i < 3; ++i) {
                int v1 = v[i];
                int v2 = v[(i + 1) % 3];
                // 确保 v1-v2 是边界边 (即 v1 和 v2 都是边界点)
                if (boundary_set.count(v1) && boundary_set.count(v2)) {
                    boundary_adj_map[v1].insert(v2);
                    boundary_adj_map[v2].insert(v1);
                }
            }
        }

        std::set<int> remaining_boundary = boundary_set;

        // 2. 遍历所有未访问的边界点，查找所有闭合环
        while (!remaining_boundary.empty()) {
            int start_v = *remaining_boundary.begin();
            int current_v = start_v;
            int prev_v = -1;

            std::vector<int> current_loop;
            bool loop_closed = false;

            while (true) {
                if (!boundary_adj_map.count(current_v)) {
                    // 孤立边界点或死胡同
                    break;
                }

                std::vector<int> valid_next_neighbors;
                for (int neighbor : boundary_adj_map[current_v]) {
                    if (neighbor != prev_v) {
                        valid_next_neighbors.push_back(neighbor);
                    }
                }

                int next_v = -1;

                if (valid_next_neighbors.size() > 1) {
                    // 非流形警告保持不变，因为这是网格的固有问题
                    std::cerr << "Topology Warning: Non-manifold boundary junction at vertex " << current_v
                              << ". Found " << valid_next_neighbors.size() << " next neighbors. Forcing path choice (first available)." << std::endl;
                    next_v = valid_next_neighbors[0];
                } else if (valid_next_neighbors.size() == 1) {
                    next_v = valid_next_neighbors[0];
                } else {
                    // Dead end
                    break;
                }

                if (next_v == start_v) {
                    loop_closed = true;
                    break;
                }

                if (std::find(current_loop.begin(), current_loop.end(), next_v) != current_loop.end()) {
                    // 循环检测警告保持不变
                    std::cerr << "Topology Warning: Cycle detected at next vertex " << next_v << " before closing to start "
                              << start_v << ". Aborting component trace." << std::endl;
                    break;
                }

                current_loop.push_back(current_v);
                prev_v = current_v;
                current_v = next_v;
            }

            // 3. 后处理和最长环选择
            if (loop_closed && current_loop.size() >= 3) {
                current_loop.push_back(start_v);

                if (current_loop.size() > longest_loop.size()) {
                    longest_loop = current_loop;
                }
            }

            // 4. 清理已尝试的顶点，即使追踪失败，也从 remaining_boundary 中移除
            std::set<int> visited_in_this_trace;
            visited_in_this_trace.insert(start_v);
            visited_in_this_trace.insert(current_loop.begin(), current_loop.end());

            for (int v_idx : visited_in_this_trace) {
                remaining_boundary.erase(v_idx);
            }
        }
    }
    // --- 边界追踪结束 ---

    // REVERTED: 仅使用最长闭合环上的点作为固定约束。
    std::vector<int> fixed_indices;
    // longest_loop 包含重复的起始点，所以要减去 1
    if (longest_loop.size() >= 3) {
        fixed_indices.assign(longest_loop.begin(), longest_loop.end() - 1);
    }

    int N_fixed_expected = boundary_set.size();
    int N_fixed = fixed_indices.size();

    // 调试输出
    std::cout << "DEBUG: Boundary set size: " << N_fixed_expected << std::endl;
    if (N_fixed == 0) {
        std::cerr << "Error: No closed boundary loop found with >= 3 vertices. Cannot fix boundary." << std::endl;
        return Eigen::MatrixXd::Zero(N_cut, 2);
    } else {
        std::cout << "DEBUG: Boundary trace completed. Found " << longest_loop.size() << " vertices in the longest loop." << std::endl;
        if (N_fixed < N_fixed_expected) {
            std::cerr << "Topology Warning: Cut mesh boundary consists of multiple disconnected components. Only "
                      << N_fixed << " vertices from the longest closed loop are used for constraints." << std::endl;
        }
    }

    int N_free = N_cut - N_fixed;

    if (N_free <= 0 || N_fixed == 0) {
        std::cerr << "Error: Invalid number of free or fixed vertices after tracing. Fixed: " << N_fixed << ", Free: " << N_free << std::endl;
        return Eigen::MatrixXd::Zero(N_cut, 2);
    }

    // 3. 构建内部索引映射和分离矩阵
    std::map<int, int> cut_to_free_map;
    std::map<int, int> cut_to_fixed_map;

    std::vector<int> free_indices;
    int current_free_idx = 0;

    // 将 fixed_indices (最长环上的点) 映射到固定索引
    for(int i = 0; i < N_fixed; ++i) {
        cut_to_fixed_map[fixed_indices[i]] = i;
    }

    // 映射自由索引
    for (int i = 0; i < N_cut; ++i) {
        if (!cut_to_fixed_map.count(i)) {
            free_indices.push_back(i);
            cut_to_free_map[i] = current_free_idx++;
        }
    }

    // 4. 构建 L_inner 和 L_I_B 矩阵 (子矩阵) - 使用 Triplet 列表
    std::vector<Eigen::Triplet<double>> L_inner_triplets;
    std::vector<Eigen::Triplet<double>> L_I_B_triplets;

    for (int k = 0; k < L_cut.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L_cut, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            double value = it.value();

            bool row_is_free = !cut_to_fixed_map.count(row);

            if (row_is_free) {
                int free_row = cut_to_free_map[row];

                bool col_is_free = !cut_to_fixed_map.count(col);

                if (col_is_free) {
                    int free_col = cut_to_free_map[col];
                    // 属于 L_inner (自由行, 自由列)
                    L_inner_triplets.push_back(Eigen::Triplet<double>(free_row, free_col, value));
                } else { // col is fixed (boundary)
                    int fixed_col = cut_to_fixed_map[col];
                    // 属于 L_I_B (自由行, 固定列)
                    L_I_B_triplets.push_back(Eigen::Triplet<double>(free_row, fixed_col, value));
                }
            }
            // 如果 row 是固定行 (边界点), 则它不属于需要求解的系统部分，跳过。
        }
    }

    // 填充稀疏矩阵
    Eigen::SparseMatrix<double> L_inner(N_free, N_free);
    Eigen::SparseMatrix<double> L_I_B(N_free, N_fixed);

    L_inner.setFromTriplets(L_inner_triplets.begin(), L_inner_triplets.end());
    L_I_B.setFromTriplets(L_I_B_triplets.begin(), L_I_B_triplets.end());

    std::cout << "DEBUG: L_inner (" << N_free << "x" << N_free << ") and L_I_B (" << N_free << "x" << N_fixed << ") assembled." << std::endl;

    // 5. 添加正则化项 (epsilon)
    const double epsilon = 1e-6; // 增加稳定性
    for (int i = 0; i < N_free; ++i) {
        L_inner.coeffRef(i, i) += epsilon;
    }

    L_inner.makeCompressed();
    L_I_B.makeCompressed();


    // 6. 设置边界点 X_boundary 的坐标 (固定在单位圆上)
    Eigen::MatrixXd X_fixed(N_fixed, 2);

    // 6a. 计算固定边界的总 3D 长度和累积长度
    std::vector<double> edge_lengths;
    double total_length = 0.0;

    // 遍历 fixed_indices (最长闭环上的点)
    for (int i = 0; i < N_fixed; ++i) {
        int current_idx = fixed_indices[i];
        // 模 N_fixed 确保循环连接
        int next_idx = fixed_indices[(i + 1) % N_fixed];

        Eigen::Vector3d p_current = cutMesh.V.row(current_idx);
        Eigen::Vector3d p_next = cutMesh.V.row(next_idx);

        double length = (p_next - p_current).norm();
        edge_lengths.push_back(length);
        total_length += length;
    }

    // 6b. 设置单位圆上的坐标 (弧长参数化)
    double current_cumulative_length = 0.0;
    for (int i = 0; i < N_fixed; ++i) {
        double normalized_arc_length = (total_length > 1e-9) ?
                                       current_cumulative_length / total_length :
                                       (double)i / N_fixed;

        double angle = 2.0 * M_PI * normalized_arc_length;

        // X_fixed 的行索引 i 对应 fixed_indices[i] 这个 cut 顶点
        X_fixed.row(i) = Eigen::Vector2d(std::cos(angle), std::sin(angle));

        if (i < edge_lengths.size()) {
            current_cumulative_length += edge_lengths[i];
        }
    }


    // 7. 构建右侧向量 B
    Eigen::MatrixXd B = -L_I_B * X_fixed;

    // 8. 求解稀疏系统
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    try {
        solver.compute(L_inner);
        if(solver.info() != Eigen::Success) {
            std::cerr << "Solver decomposition failed for L_inner! Check singularity/constraints." << std::endl;
            return V_flat_result;
        }

        Eigen::MatrixXd X_inner = solver.solve(B);

        // 9. 重建最终结果 V_flat
        for (int i = 0; i < N_free; ++i) {
            V_flat_result.row(free_indices[i]) = X_inner.row(i);
        }
        for (int i = 0; i < N_fixed; ++i) {
            V_flat_result.row(fixed_indices[i]) = X_fixed.row(i); // 注意这里使用 fixed_indices
        }

        // =======================================================
        // 10. 归一化和居中 V_flat (增强可视化效果)
        // =======================================================

        // 1. 计算质心 (Centroid)
        Eigen::Vector2d centroid = V_flat_result.colwise().mean();

        // 2. 居中
        V_flat_result.rowwise() -= centroid.transpose();

        // 3. 归一化 (缩放到最大半径为 1)
        double max_radius_sq = 0.0;
        for (int i = 0; i < V_flat_result.rows(); ++i) {
            max_radius_sq = std::max(max_radius_sq, V_flat_result.row(i).squaredNorm());
        }
        double scale = 1.0 / std::sqrt(max_radius_sq);
        V_flat_result *= scale;

        std::cout << "Solver decomposition successful. System solved." << std::endl;
        std::cout << "DEBUG: V_flat successfully centered and scaled to unit radius." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Solver execution failed: " << e.what() << std::endl;
        return V_flat_result;
    }

    return V_flat_result;
}