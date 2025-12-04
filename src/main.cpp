#include "MeshIO.h"
#include "Flattening.h"
#include "CutMesh.h"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>

// 假设的输入文件路径和参数
const std::string INPUT_MESH_PATH = "../data/sphere.off";
const int ORBIFOLD_TYPE = 4; // 假设用于四面体锥点结构 (t4)

int main(int argc, char* argv[]) {

    // 1. 变量初始化
    Eigen::MatrixXd V_initial; // 初始顶点坐标 (N x 3)
    Eigen::MatrixXi T_initial; // 初始三角形拓扑 (F x 3)
    std::vector<int> cones;    // 锥点索引 (1-based)
    CutMesh cutMesh;           // 用于存储切割后的网格 V', T', 映射等

    // 2. 加载网格和锥点数据
    std::cout << "--- Stage 0: Loading Mesh ---" << std::endl;
    if (!load_mesh(INPUT_MESH_PATH, V_initial, T_initial, cones)) {
        std::cerr << "Fatal Error: Failed to load input mesh from " << INPUT_MESH_PATH << std::endl;
        return 1;
    }

    std::cout << "Successfully loaded " << V_initial.rows() << " vertices and "
              << T_initial.rows() << " faces." << std::endl;
    std::cout << "Input V size: " << V_initial.rows() << " rows x " << V_initial.cols() << " cols." << std::endl;

    if (cones.size() < 3) {
        std::cerr << "Error: Not enough cone points found (need at least 3)." << std::endl;
        return 1;
    }

    // 3. 执行平展算法
    std::cout << "\n--- Starting Orbifold Flattening ---" << std::endl;

    Eigen::MatrixXd V_flat_result; // 结果 V_flat (N' x 2)

    try {
        V_flat_result = flatten_sphere(V_initial, T_initial, cones, ORBIFOLD_TYPE, cutMesh);
    } catch (const std::exception& e) {
        std::cerr << "Flattening failed due to exception: " << e.what() << std::endl;
        return 1;
    }

    // 4. 检查结果并保存
    if (V_flat_result.rows() == 0 || cutMesh.T.rows() == 0) {
        std::cerr << "\nFatal Error: Flattening resulted in empty or invalid output." << std::endl;
        return 1;
    }

    std::cout << "\n--- Stage 4: Saving Result ---" << std::endl;
    std::cout << "Flattening complete. V_flat size: " << V_flat_result.rows() << " rows x "
              << V_flat_result.cols() << " cols." << std::endl;

    // 使用切割后的拓扑 T' (cutMesh.T) 和平展坐标 V_flat_result 进行保存
    std::string output_filename = "flattened_result.obj";
    if (write_mesh_obj(output_filename, V_flat_result, cutMesh.T)) {
        std::cout << "Orbifold flattening completed successfully and result saved to "
                  << output_filename << "." << std::endl;
    } else {
        std::cerr << "Error: Failed to save the flattened mesh." << std::endl;
        return 1;
    }

    return 0;
}