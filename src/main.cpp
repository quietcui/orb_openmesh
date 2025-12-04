#include "Flattening.h"
#include "MeshTypes.h"
#include <iostream>
#include <vector>

int main() {
    MyMesh mesh;
    // NOTE: 请确保您的项目根目录下有一个名为 'data' 的文件夹，其中包含 'sphere.off' 文件。
    std::string input_file = "../data/sphere.off";

    // 1. Read the mesh
    OpenMesh::IO::Options opt;
    if (!OpenMesh::IO::read_mesh(mesh, input_file, opt)) {
        std::cerr << "Error: Could not read file: " << input_file << std::endl;
        return 1;
    }
    std::cout << "Mesh read successfully: " << mesh.n_vertices() << " vertices, " << mesh.n_faces() << " faces." << std::endl;

    // 2. Set cone indices (假设输入文件是 1-based 索引, 所以在 C++ 中需要 -1)
    std::vector<int> cones = {50-1, 100-1, 130-1};
    std::cout << "Using cone indices: " << cones[0] << ", " << cones[1] << ", " << cones[2] << std::endl;

    // 3. Execute flattening
    Eigen::MatrixXd res = flatten_sphere_openmesh(mesh, cones);

    if (res.rows() == 0) {
        std::cerr << "Flattening failed. See error messages above." << std::endl;
        return 1;
    }

    // 4. Save the result
    std::string output_file = "flattened_result.obj";
    if (OpenMesh::IO::write_mesh(mesh, output_file)) {
        std::cout << "Result successfully saved to: " << output_file << std::endl;
    } else {
        std::cerr << "Error: Failed to write mesh to file." << std::endl;
        return 1;
    }

    return 0;
}