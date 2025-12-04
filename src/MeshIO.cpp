#include "MeshIO.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip> // 用于 setprecision

// ====================================================================
// OBJ 网格加载函数
// ====================================================================

// 这是您程序中缺失的 load_mesh 的实现代码！
bool load_mesh(
        const std::string& filename,
        Eigen::MatrixXd& V,
        Eigen::MatrixXi& T,
        std::vector<int>& cones
) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return false;
    }

    std::string line;
    int num_v = 0;
    int num_f = 0;
    std::vector<Eigen::Vector3d> v_list;
    std::vector<Eigen::Vector3i> t_list;
    bool reading_cones = false;
    int line_count = 0;

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;
        line_count++;

        if (token == "OFF" || token.empty() || token[0] == '#') {
            continue;
        }

        if (num_v == 0) {
            // 读取顶点数和面数
            ss.str("");
            ss.clear();
            ss << line;
            ss >> num_v >> num_f;
            v_list.reserve(num_v);
            t_list.reserve(num_f);
            continue;
        }

        if (v_list.size() < (size_t)num_v) {
            // 读取顶点坐标
            double x, y, z;
            ss.str("");
            ss.clear();
            ss << line;
            if (!(ss >> x >> y >> z)) {
                continue;
            }
            v_list.push_back(Eigen::Vector3d(x, y, z));
        } else if (t_list.size() < (size_t)num_f) {
            // 读取面片
            int count, i1, i2, i3;
            ss.str("");
            ss.clear();
            ss << line;
            if (ss >> count && count == 3) {
                if (!(ss >> i1 >> i2 >> i3)) {
                    continue;
                }
                // OFF/OBJ 格式通常是 1-based，Eigen 是 0-based
                t_list.push_back(Eigen::Vector3i(i1, i2, i3));
            }
        } else if (line_count > num_v + num_f + 1) {
            // 假设后面的行是锥点索引 (1-based)
            int c_idx;
            ss.str("");
            ss.clear();
            ss << line;
            while (ss >> c_idx) {
                cones.push_back(c_idx);
            }
        }
    }

    if (v_list.size() != (size_t)num_v || t_list.size() != (size_t)num_f) {
        std::cerr << "Error: Read " << v_list.size() << "/" << num_v
                  << " vertices and " << t_list.size() << "/" << num_f << " faces. Incomplete file." << std::endl;
        return false;
    }

    V.resize(num_v, 3);
    for (int i = 0; i < num_v; ++i) {
        V.row(i) = v_list[i].transpose();
    }

    T.resize(num_f, 3);
    for (int i = 0; i < num_f; ++i) {
        T.row(i) = t_list[i].transpose();
    }

    // 如果没有加载到锥点，提供一个默认值以确保程序运行
    if (cones.size() < 3) {
        std::cout << "Warning: No cones found in file. Using default cones (1, 100, 200)." << std::endl;
        cones = {1, 100, 200};
    }

    return true;
}

// ====================================================================
// OBJ 网格保存函数 (保持不变)
// ====================================================================
bool write_mesh_obj(
        const std::string& filename,
        const Eigen::MatrixXd& V_flat,
        const Eigen::MatrixXi& T_cut
) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    outfile << "# Mesh flattened to 2D coordinates." << std::endl;
    outfile << "# Vertices: " << V_flat.rows() << ", Faces: " << T_cut.rows() << std::endl;

    // 1. 写入顶点 (v)
    for (int i = 0; i < V_flat.rows(); ++i) {
        outfile << "v "
                << std::fixed << std::setprecision(8) << V_flat(i, 0) << " "
                << std::fixed << std::setprecision(8) << V_flat(i, 1) << " "
                << std::fixed << std::setprecision(8) << 0.0 << "\n";
    }

    // 2. 写入面片 (f)
    // OBJ 格式使用 1-based 索引，所以需要 T_cut(i, j) + 1
    for (int i = 0; i < T_cut.rows(); ++i) {
        outfile << "f "
                << T_cut(i, 0) + 1 << " "
                << T_cut(i, 1) + 1 << " "
                << T_cut(i, 2) + 1 << "\n";
    }

    outfile.close();
    std::cout << "Successfully wrote flattened mesh to: " << filename << std::endl;
    return true;
}