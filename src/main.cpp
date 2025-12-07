#include "MeshTypes.h"
#include "Flattening.h"
#include "Cutting.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    std::string input_path  = "../data/sphere.off";
    std::string output_path = "flattened_result.obj";

    if (argc >= 2)
        input_path = argv[1];
    if (argc >= 3)
        output_path = argv[2];

    MyMesh mesh;

    try
    {
        if (!OpenMesh::IO::read_mesh(mesh, input_path))
        {
            std::cerr << "Failed to read mesh file: " << input_path << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception while reading mesh: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Mesh loaded successfully: "
              << mesh.n_vertices() << " vertices, "
              << mesh.n_faces()    << " faces\n";

    // 1-based 的 Matlab index -> 0-based
    std::vector<int> cones_in = {50 - 1, 100 - 1, 130 - 1};  // TODO: 换成你真实的锥点
    std::vector<int> cones_after_cut;

    // 1) 先切缝，并更新锥点索引
    cut_mesh_along_cones(mesh, cones_in, cones_after_cut);

    // 2) 再用更新后的锥点做 orbifold flatten
    flatten_sphere(mesh, cones_after_cut, 1);

    try
    {
        if (!OpenMesh::IO::write_mesh(mesh, output_path))
        {
            std::cerr << "Failed to write mesh to: " << output_path << "\n";
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception while writing mesh: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Flattening finished. Output written to: "
              << output_path << "\n";
    return 0;
}
