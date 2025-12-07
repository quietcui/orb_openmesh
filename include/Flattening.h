// Flattening.h
#pragma once

#include "MeshTypes.h"         // 你工程里定义 MyMesh 的地方
#include "CutMesh.h"
#include "TreeCutter.h"
#include "PosConstraints.h"
#include "FlatteningSolver.h"

#include <vector>

// orbifold_type: 1..4 (目前重点支持 1~3)
// cones: 0-based 顶点索引（建议你在 main 里把 Matlab 的 1-based 转成 0-based 再传进来）
void flatten_sphere(
        MyMesh& mesh,
        const std::vector<int>& cones,
        int orbifold_type,
        bool verbose = true);
