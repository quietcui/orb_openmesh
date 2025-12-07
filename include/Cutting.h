#pragma once

#include "MeshTypes.h"
#include <vector>

// cones_in  :  原始网格上的锥点索引（0-based，与输入 mesh 一致）
// cones_out :  切缝之后，对应在「切后网格」上的锥点索引（0-based），
//              会尽量选边界上的那一个副本。
void cut_mesh_along_cones(MyMesh& mesh,
                          const std::vector<int>& cones_in,
                          std::vector<int>&       cones_out);
