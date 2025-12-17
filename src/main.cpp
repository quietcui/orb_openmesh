// main.cpp
//
// Minimal CLI wrapper around flatten_sphere() (see src/Flattening.cpp).
// The goal is to make it easy to reproduce the author's MATLAB experiments:
// - Provide input mesh
// - Provide orbifold type (1..4)
// - Provide cone vertex indices
//
// Example (3-cone orbifolds):
//   ./orb_openmesh input.off output.off 1 12 345 678
// If you are copying indices from MATLAB (1-based), add --1based:
//   ./orb_openmesh input.off output.off 1 13 346 679 --1based

#include "Flattening.h"
#include "MeshTypes.h"

#include <OpenMesh/Core/IO/MeshIO.hh>

#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char* argv0)
{
    std::cerr
            << "Usage:\n"
            << "  " << argv0 << " <input_mesh> <output_mesh> <orbifold_type> <cones...> [--1based] [--quiet]\n\n"
            << "orbifold_type:\n"
            << "  1 = Square\n"
            << "  2 = Diamond\n"
            << "  3 = Triangle\n"
            << "  4 = Parallelogram\n\n"
            << "cones:\n"
            << "  For types 1..3: provide exactly 3 cone vertex indices.\n"
            << "  For type 4:    provide exactly 4 cone vertex indices.\n\n"
            << "Flags:\n"
            << "  --1based  Treat cone indices as 1-based (MATLAB), subtract 1.\n"
            << "  --quiet   Less logging.\n";
}

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        print_usage(argv[0]);
        return 1;
    }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    const int orbifold_type    = std::stoi(argv[3]);

    bool oneBased = false;
    bool verbose = true;

    // Parse trailing flags (after required args).
    for (int i = 4; i < argc; ++i)
    {
        const std::string s = argv[i];
        if (s == "--1based") oneBased = true;
        if (s == "--quiet") verbose = false;
    }

    // Extract cone indices from argv[4..] until a flag.
    std::vector<int> cones;
    for (int i = 4; i < argc; ++i)
    {
        const std::string s = argv[i];
        if (!s.empty() && s[0] == '-')
            break;
        cones.push_back(std::stoi(s));
    }

    const int needed = (orbifold_type == 4) ? 4 : 3;
    if (static_cast<int>(cones.size()) != needed)
    {
        std::cerr << "Error: orbifold_type=" << orbifold_type
                  << " requires " << needed << " cone indices, got " << cones.size() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    if (oneBased)
    {
        for (int& c : cones) c -= 1;
    }

    MyMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, in_path))
    {
        std::cerr << "Error: cannot read mesh: " << in_path << "\n";
        return 1;
    }

    try
    {
        flatten_sphere(mesh, cones, orbifold_type, verbose);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Flattening failed: " << e.what() << "\n";
        return 1;
    }

    if (!OpenMesh::IO::write_mesh(mesh, out_path))
    {
        std::cerr << "Error: cannot write mesh: " << out_path << "\n";
        return 1;
    }

    if (verbose)
        std::cout << "Wrote flattened mesh to: " << out_path << "\n";

    return 0;
}
