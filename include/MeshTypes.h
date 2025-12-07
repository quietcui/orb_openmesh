#pragma once

// OpenMesh 头文件
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

// 你要操作的网格类型：三角网格
using MyMesh = OpenMesh::TriMesh_ArrayKernelT<>;
