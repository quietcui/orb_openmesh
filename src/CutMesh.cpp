#include "CutMesh.h"

CutMesh::CutMesh(
        const Eigen::MatrixXd& V_in,
        const Eigen::MatrixXi& T_in,
        const std::vector<std::vector<std::pair<int, int>>>& pathPairs_in,
        const std::vector<int>& cutIndsToUncutInds_in,
        const std::vector<std::vector<int>>& uncutIndsToCutInds_in
) :
        V(V_in),
        T(T_in),
        pathPairs(pathPairs_in),
        cutIndsToUncutInds(cutIndsToUncutInds_in),
        uncutIndsToCutInds(uncutIndsToCutInds_in)
{
}