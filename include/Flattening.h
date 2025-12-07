#pragma once

#include "MeshTypes.h"

#include <Eigen/Core>
#include <vector>

// orbifold_type 先预留参数，目前实现的是 3 个锥点的 Type-I 圆盘
// cones 是顶点索引（0-based）——和你 mesh 的 vertex handle 对应
void flatten_sphere(
        MyMesh& mesh,
        const std::vector<int>& cones,
        int orbifold_type = 1);
