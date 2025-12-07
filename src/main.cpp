#include "MeshTypes.h"
#include "Flattening.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    std::string input_path  = "../data/sphere.off";
    std::string output_path = "flattened_result.obj";

    if (argc >= 2) input_path  = argv[1];
    if (argc >= 3) output_path = argv[2];

    MyMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, input_path)) {
        std::cerr << "Cannot read mesh: " << input_path << "\n";
        return 1;
    }

    std::cout << "Mesh loaded successfully: "
              << mesh.n_vertices() << " vertices, "
              << mesh.n_faces()    << " faces\n";

    // ⚠️ 这里记得把 Matlab 的 1-based 索引改成 C++ 的 0-based 索引
    // 假设 Matlab 里用的是 inds = [50 100 130]
    std::vector<int> cones = {50 - 1, 100 - 1, 130 - 1}; // 0-based

    int orbifold_type = 1; // Type I

    // 调用我们刚刚实现的 flatten_sphere
    flatten_sphere(mesh, cones, orbifold_type, true);

    // 写出结果
    if (!OpenMesh::IO::write_mesh(mesh, output_path)) {
        std::cerr << "Cannot write mesh: " << output_path << "\n";
        return 1;
    }

    std::cout << "[OBJWriter] : write file\n";
    std::cout << "Flattening finished. Output written to: "
              << output_path << "\n";
    return 0;
}
