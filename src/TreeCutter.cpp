//
// Created by cgl on 2025/12/7.
//
// TreeCutter.cpp
#include "TreeCutter.h"
#include <queue>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <iostream>

TreeCutter::TreeCutter(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXi& T,
                       const Eigen::MatrixXi& treeAdj,
                       const std::vector<int>& treeIndices,
                       int root)
        : V_(V)
        , T_(T)
        , treeStructure_(treeAdj)
        , treeIndices_(treeIndices)
        , treeRoot_(root)
        , alreadyCut_(false)
{
    int nV = static_cast<int>(V_.rows());
    cutIndsToUncutInds_.resize(nV);
    uncutIndsToCutInds_.resize(nV);
    for (int i = 0; i < nV; ++i)
    {
        cutIndsToUncutInds_[i] = i;
        uncutIndsToCutInds_[i].push_back(i);
    }

    directTree();
}

void TreeCutter::directTree()
{
    const int k = static_cast<int>(treeStructure_.rows());
    Eigen::MatrixXi directed = Eigen::MatrixXi::Zero(k, k);
    Eigen::MatrixXi tmp = treeStructure_;

    std::vector<int> q;
    q.push_back(treeRoot_);

    while (!q.empty())
    {
        int root = q.front();
        q.erase(q.begin());

        std::vector<int> sons;
        for (int j = 0; j < k; ++j)
        {
            if (tmp(root, j) != 0 || tmp(j, root) != 0)
                sons.push_back(j);
        }

        for (int s : sons)
        {
            directed(root, s) = 1;
            tmp(root, s) = 0;
            tmp(s, root) = 0;
            q.push_back(s);
        }
    }

    treeStructure_ = directed;
}

void TreeCutter::cutTree()
{
    if (alreadyCut_)
        throw std::runtime_error("TreeCutter::cutTree can only be called once.");
    alreadyCut_ = true;
    cutTreeRecurse(treeRoot_);
}

void TreeCutter::cutTreeRecurse(int rootNode)
{
    const int k = static_cast<int>(treeStructure_.rows());
    std::vector<int> sons;
    for (int j = 0; j < k; ++j)
    {
        if (treeStructure_(rootNode, j) != 0)
            sons.push_back(j);
    }
    if (sons.empty())
        return;

    int sourceInd = treeIndices_[rootNode];
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2>> starPathPairs;

    for (int son : sons)
    {
        int targetInd = treeIndices_[son];
        std::vector<int> newPath = shortestPath(sourceInd, targetInd);
        Eigen::Matrix<int, Eigen::Dynamic, 2> pathCorr = splitMeshByPath(newPath);
        starPathPairs.push_back(pathCorr);
    }

    splitCenterNode(treeIndices_[rootNode], starPathPairs);

    for (int son : sons)
        cutTreeRecurse(son);
}

void TreeCutter::findBoundaryVertices(std::vector<char>& isBoundary) const
{
    int nV = static_cast<int>(V_.rows());
    isBoundary.assign(nV, 0);

    struct EdgeKey
    {
        int a, b;
        bool operator==(const EdgeKey& other) const { return a == other.a && b == other.b; }
    };

    struct EdgeKeyHash
    {
        std::size_t operator()(const EdgeKey& e) const
        {
            return std::hash<int>()(e.a * 73856093 ^ e.b * 19349663);
        }
    };

    std::unordered_map<EdgeKey, int, EdgeKeyHash> edgeCount;

    auto add_edge = [&](int x, int y)
    {
        if (x > y) std::swap(x, y);
        EdgeKey key{x, y};
        edgeCount[key] += 1;
    };

    int nF = static_cast<int>(T_.rows());
    for (int fi = 0; fi < nF; ++fi)
    {
        int v0 = T_(fi, 0);
        int v1 = T_(fi, 1);
        int v2 = T_(fi, 2);
        add_edge(v0, v1);
        add_edge(v0, v2);
        add_edge(v1, v2);
    }

    for (const auto& kv : edgeCount)
    {
        if (kv.second == 1)
        {
            int a = kv.first.a;
            int b = kv.first.b;
            if (a >= 0 && a < nV) isBoundary[a] = 1;
            if (b >= 0 && b < nV) isBoundary[b] = 1;
        }
    }
}

std::vector<int> TreeCutter::shortestPath(int source, int target) const
{
    const int nV = static_cast<int>(V_.rows());
    std::vector<std::vector<std::pair<int,double>>> adj(nV);

    auto add_edge = [&](int a, int b)
    {
        if (a == b) return;
        double w = (V_.row(a) - V_.row(b)).norm();
        adj[a].push_back({b, w});
        adj[b].push_back({a, w});
    };

    int nF = static_cast<int>(T_.rows());
    for (int fi = 0; fi < nF; ++fi)
    {
        int v0 = T_(fi, 0);
        int v1 = T_(fi, 1);
        int v2 = T_(fi, 2);
        add_edge(v0, v1);
        add_edge(v0, v2);
        add_edge(v1, v2);
    }

    std::vector<char> isBoundary;
    findBoundaryVertices(isBoundary);
    for (int v = 0; v < nV; ++v)
    {
        if (v == source || v == target) continue;
        if (isBoundary[v]) adj[v].clear();
    }

    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(nV, INF);
    std::vector<int> prev(nV, -1);

    using Node = std::pair<double,int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

    dist[source] = 0.0;
    pq.push({0.0, source});

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        if (u == target) break;

        for (auto [v, w] : adj[u])
        {
            if (w <= 0) continue;
            double nd = d + w;
            if (nd < dist[v])
            {
                dist[v] = nd;
                prev[v] = u;
                pq.push({nd, v});
            }
        }
    }

    std::vector<int> path;
    if (prev[target] == -1 && source != target)
    {
        path.push_back(source);
        path.push_back(target);
        return path;
    }

    for (int v = target; v != -1; v = prev[v])
        path.push_back(v);

    std::reverse(path.begin(), path.end());
    return path;
}

Eigen::Matrix<int, Eigen::Dynamic, 2>
TreeCutter::splitMeshByPath(const std::vector<int>& path)
{
    int nF = static_cast<int>(T_.rows());
    std::vector<int> left, right;

    auto triContainsEdge = [&](int fi, int a, int b) -> bool
    {
        int c0 = T_(fi,0), c1 = T_(fi,1), c2 = T_(fi,2);
        int cnt = 0;
        if (c0 == a || c0 == b) ++cnt;
        if (c1 == a || c1 == b) ++cnt;
        if (c2 == a || c2 == b) ++cnt;
        return cnt == 2;
    };

    auto edgeOrientationPositive = [&](int fi, int a, int b) -> bool
    {
        int tri[3] = {T_(fi,0), T_(fi,1), T_(fi,2)};
        int ia = -1, ib = -1;
        for (int k = 0; k < 3; ++k)
        {
            if (tri[k] == a) ia = k;
            if (tri[k] == b) ib = k;
        }
        if (ia < 0 || ib < 0) return false;
        int next = (ia + 1) % 3;
        return tri[next] == b;
    };

    for (size_t j = 0; j + 1 < path.size(); ++j)
    {
        int a = path[j];
        int b = path[j+1];

        std::vector<int> tris;
        for (int fi = 0; fi < nF; ++fi)
        {
            if (triContainsEdge(fi, a, b))
                tris.push_back(fi);
        }
        if (tris.size() != 2) continue;

        int t0 = tris[0], t1 = tris[1];
        if (edgeOrientationPositive(t0, a, b))
        {
            left.push_back(t0);
            right.push_back(t1);
        }
        else
        {
            left.push_back(t1);
            right.push_back(t0);
        }
    }

    auto uniq = [](std::vector<int>& v)
    {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    uniq(left);
    uniq(right);

    std::vector<int> touching;
    {
        std::vector<char> used(nF, 0);
        for (size_t jj = 1; jj + 1 < path.size(); ++jj)
        {
            int v = path[jj];
            for (int fi = 0; fi < nF; ++fi)
            {
                if (used[fi]) continue;
                int c0 = T_(fi,0), c1 = T_(fi,1), c2 = T_(fi,2);
                if (c0 == v || c1 == v || c2 == v)
                {
                    used[fi] = 1;
                    touching.push_back(fi);
                }
            }
        }
    }

    auto remove_from = [&](std::vector<int>& base, const std::vector<int>& rem)
    {
        std::vector<char> mark(nF, 0);
        for (int x : rem) mark[x] = 1;
        std::vector<int> out;
        out.reserve(base.size());
        for (int x : base) if (!mark[x]) out.push_back(x);
        base.swap(out);
    };

    remove_from(touching, left);
    remove_from(touching, right);

    auto shareTwo = [&](int f1, int f2) -> bool
    {
        int a0 = T_(f1,0), a1 = T_(f1,1), a2 = T_(f1,2);
        int b0 = T_(f2,0), b1 = T_(f2,1), b2 = T_(f2,2);
        int cnt = 0;
        if (a0==b0 || a0==b1 || a0==b2) ++cnt;
        if (a1==b0 || a1==b1 || a1==b2) ++cnt;
        if (a2==b0 || a2==b1 || a2==b2) ++cnt;
        return cnt >= 2;
    };

    for (int iter = 0; iter < 1000; ++iter)
    {
        bool changed = false;
        std::vector<int> newRight, newLeft;

        for (int fi : touching)
        {
            bool nearR = false, nearL = false;
            for (int rf : right)
            {
                if (shareTwo(fi, rf)) { nearR = true; break; }
            }
            for (int lf : left)
            {
                if (shareTwo(fi, lf)) { nearL = true; break; }
            }
            if (nearR && !nearL) newRight.push_back(fi);
            else if (nearL && !nearR) newLeft.push_back(fi);
        }

        if (!newRight.empty())
        {
            right.insert(right.end(), newRight.begin(), newRight.end());
            changed = true;
        }
        if (!newLeft.empty())
        {
            left.insert(left.end(), newLeft.begin(), newLeft.end());
            changed = true;
        }

        uniq(left);
        uniq(right);
        remove_from(touching, left);
        remove_from(touching, right);

        if (!changed || touching.empty()) break;
    }

    std::vector<std::pair<int,int>> pathCorr;
    for (size_t j = 1; j + 1 < path.size(); ++j)
    {
        int vOrig = path[j];
        Eigen::RowVector3d p = V_.row(vOrig);
        int newIdx = static_cast<int>(V_.rows());
        V_.conservativeResize(newIdx+1, 3);
        V_.row(newIdx) = p;

        for (int fi : left)
        {
            for (int k = 0; k < 3; ++k)
            {
                if (T_(fi, k) == vOrig)
                    T_(fi, k) = newIdx;
            }
        }

        cutIndsToUncutInds_.push_back(vOrig);
        uncutIndsToCutInds_[vOrig].push_back(newIdx);

        pathCorr.emplace_back(vOrig, newIdx);
    }

    int vEnd = path.back();
    pathCorr.emplace_back(vEnd, vEnd);

    Eigen::Matrix<int, Eigen::Dynamic, 2> M(pathCorr.size(), 2);
    for (int i = 0; i < static_cast<int>(pathCorr.size()); ++i)
    {
        M(i,0) = pathCorr[i].first;
        M(i,1) = pathCorr[i].second;
    }
    return M;
}

void TreeCutter::splitCenterNode(
        int centerVertex,
        std::vector<Eigen::Matrix<int, Eigen::Dynamic, 2>>& starPathPairs)
{
    // ------------------------------------------------------------
    // 1) collect all triangles incident to "centerVertex"
    // ------------------------------------------------------------
    const int nF = static_cast<int>(T_.rows());
    std::vector<int> remaining;
    remaining.reserve(nF);
    for (int fi = 0; fi < nF; ++fi)
    {
        int v0 = T_(fi,0);
        int v1 = T_(fi,1);
        int v2 = T_(fi,2);
        if (v0 == centerVertex || v1 == centerVertex || v2 == centerVertex)
            remaining.push_back(fi);
    }
    if (remaining.empty())
    {
        // No triangles around this center (should not happen, but be robust).
        for (const auto& pp : starPathPairs)
            pathPairs_.push_back(pp);
        return;
    }

    // ------------------------------------------------------------
    // 2) split the one-ring of centerVertex into groups of adjacent
    //    triangles, where adjacency is via any non-center vertex.
    // ------------------------------------------------------------
    std::vector<std::vector<int>> groups;
    groups.reserve(4); // usually small

    auto triSharesNonCenterVertex = [&](int f1, int f2) -> bool
    {
        int a0 = T_(f1,0), a1 = T_(f1,1), a2 = T_(f1,2);
        int b0 = T_(f2,0), b1 = T_(f2,1), b2 = T_(f2,2);
        int vertsA[3] = {a0,a1,a2};
        int vertsB[3] = {b0,b1,b2};
        for (int ia = 0; ia < 3; ++ia)
        {
            int va = vertsA[ia];
            if (va == centerVertex) continue;
            for (int ib = 0; ib < 3; ++ib)
            {
                int vb = vertsB[ib];
                if (vb == centerVertex) continue;
                if (va == vb)
                    return true;
            }
        }
        return false;
    };

    while (!remaining.empty())
    {
        std::vector<int> group;
        group.push_back(remaining.back());
        remaining.pop_back();

        bool changed = true;
        while (changed)
        {
            changed = false;
            std::vector<int> rest;
            rest.reserve(remaining.size());
            for (int fi : remaining)
            {
                bool belongs = false;
                for (int gj : group)
                {
                    if (triSharesNonCenterVertex(fi, gj))
                    {
                        belongs = true;
                        break;
                    }
                }
                if (belongs)
                {
                    group.push_back(fi);
                    changed = true;
                }
                else
                {
                    rest.push_back(fi);
                }
            }
            remaining.swap(rest);
        }

        std::sort(group.begin(), group.end());
        group.erase(std::unique(group.begin(), group.end()), group.end());
        groups.push_back(group);
    }

    // ------------------------------------------------------------
    // 3) for each group, duplicate the center vertex (except the first)
    // ------------------------------------------------------------
    std::vector<int> groupCenters;
    groupCenters.reserve(groups.size());

    int currentNV = static_cast<int>(V_.rows());
    for (size_t gi = 0; gi < groups.size(); ++gi)
    {
        int centerInd;
        if (gi == 0)
        {
            centerInd = centerVertex;
        }
        else
        {
            V_.conservativeResize(currentNV + 1, V_.cols());
            V_.row(currentNV) = V_.row(centerVertex);
            centerInd = currentNV;
            ++currentNV;

            cutIndsToUncutInds_.push_back(centerVertex);
            if (centerVertex >= 0 &&
                centerVertex < static_cast<int>(uncutIndsToCutInds_.size()))
            {
                uncutIndsToCutInds_[centerVertex].push_back(centerInd);
            }
        }

        for (int fi : groups[gi])
        {
            for (int k = 0; k < 3; ++k)
            {
                if (T_(fi,k) == centerVertex)
                    T_(fi,k) = centerInd;
            }
        }

        groupCenters.push_back(centerInd);
    }

    // ------------------------------------------------------------
    // 4) build per-group vertex sets
    // ------------------------------------------------------------
    std::vector<std::unordered_set<int>> groupVerts(groups.size());
    for (size_t gi = 0; gi < groups.size(); ++gi)
    {
        auto& vs = groupVerts[gi];
        for (int fi : groups[gi])
        {
            int v0 = T_(fi,0);
            int v1 = T_(fi,1);
            int v2 = T_(fi,2);
            vs.insert(v0);
            vs.insert(v1);
            vs.insert(v2);
        }
    }

    // ------------------------------------------------------------
    // 5) update the local starPathPairs — add one row with centers
    // ------------------------------------------------------------
    for (auto& pair : starPathPairs)
    {
        if (pair.rows() == 0) continue;

        int centers[2] = { -1, -1 };

        for (int col = 0; col < 2; ++col)
        {
            for (size_t gi = 0; gi < groups.size(); ++gi)
            {
                const auto& vs = groupVerts[gi];
                bool found = false;
                for (int r = 0; r < pair.rows(); ++r)
                {
                    int v = pair(r, col);
                    if (vs.find(v) != vs.end())
                    {
                        centers[col] = groupCenters[gi];
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }

        if (centers[0] == -1 || centers[1] == -1)
        {
            std::cerr << "[TreeCutter::splitCenterNode] Warning: could not "
                      << "assign centers to starPathPair.\n";
            continue;
        }

        Eigen::Matrix<int, Eigen::Dynamic, 2> newPair(pair.rows() + 1, 2);
        newPair.row(0) << centers[0], centers[1];
        newPair.block(1,0,pair.rows(),2) = pair;
        pair.swap(newPair);
    }

    // ------------------------------------------------------------
    // 6) update already existing global pathPairs_
    // ------------------------------------------------------------
    for (auto& pair : pathPairs_)
    {
        if (pair.rows() == 0) continue;

        int centers[2] = { -1, -1 };
        for (int col = 0; col < 2; ++col)
        {
            for (size_t gi = 0; gi < groups.size(); ++gi)
            {
                const auto& vs = groupVerts[gi];
                bool found = false;
                for (int r = 0; r < pair.rows() - 1; ++r) // exclude last row
                {
                    int v = pair(r, col);
                    if (vs.find(v) != vs.end())
                    {
                        centers[col] = groupCenters[gi];
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
        }

        if ((centers[0] == -1) != (centers[1] == -1))
        {
            std::cerr << "[TreeCutter::splitCenterNode] Inconsistent centers "
                      << "assignment for existing pathPairs_.\n";
        }

        if (centers[0] != -1 && centers[1] != -1)
        {
            int lastRow = pair.rows() - 1;
            pair(lastRow,0) = centers[0];
            pair(lastRow,1) = centers[1];
        }
    }

    // ------------------------------------------------------------
    // 7) append all starPathPairs to pathPairs_
    // ------------------------------------------------------------
    for (const auto& pp : starPathPairs)
        pathPairs_.push_back(pp);
}

CutMesh TreeCutter::getCutMesh() const
{
    return CutMesh(V_, T_, pathPairs_, cutIndsToUncutInds_, uncutIndsToCutInds_);
}
