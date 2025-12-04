// src/Mesh.h
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <array>
#include <cmath>

#include <Eigen/Core>

// 一个简单的三角网格结构
class TriangleMesh {
public:
    // 顶点坐标
    std::vector<Eigen::Vector3d> vertices;
    // 每个面三个顶点索引（0-based）
    std::vector<Eigen::Vector3i> faces;

    // 每个顶点的一圈邻接点
    std::vector<std::vector<int>> vertexNeighbors;

    // 每条边的 cotan 权重：key = (min(i,j), max(i,j)) 打包成 uint64_t
    std::unordered_map<std::uint64_t, double> edgeCotWeights;

    bool loadOBJ(const std::string& path, std::string& errMsg) {
        std::ifstream in(path);
        if (!in) {
            errMsg = "无法打开 OBJ 文件: " + path;
            return false;
        }
        vertices.clear();
        faces.clear();

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string tag;
            iss >> tag;
            if (tag == "v") {
                double x, y, z;
                if (!(iss >> x >> y >> z)) continue;
                vertices.emplace_back(x, y, z);
            } else if (tag == "f") {
                // 只支持三角形
                std::array<int, 3> idx{};
                for (int k = 0; k < 3; ++k) {
                    std::string token;
                    if (!(iss >> token)) {
                        errMsg = "OBJ 面不是三角形或格式不正确";
                        return false;
                    }
                    // token 可能是 "v", "v/vt", "v/vt/vn"
                    int vIdx = 0;
                    std::stringstream tss(token);
                    std::string part;
                    if (std::getline(tss, part, '/')) {
                        vIdx = std::stoi(part);
                    } else {
                        vIdx = std::stoi(token);
                    }
                    // OBJ 顶点索引从 1 开始
                    idx[k] = vIdx - 1;
                }
                faces.emplace_back(idx[0], idx[1], idx[2]);
            }
        }
        if (vertices.empty() || faces.empty()) {
            errMsg = "OBJ 文件中顶点或面为空";
            return false;
        }
        buildAdjacencyAndWeights();
        return true;
    }

    // 导出带 UV 的 OBJ（每个顶点一个 uv，face 使用 v/vt 形式）
    bool writeOBJWithUV(const std::string& path,
                        const std::vector<Eigen::Vector2d>& uv,
                        std::string& errMsg) const {
        if (uv.size() != vertices.size()) {
            errMsg = "UV 数量与顶点数量不一致";
            return false;
        }
        std::ofstream out(path);
        if (!out) {
            errMsg = "无法写入 OBJ 文件: " + path;
            return false;
        }

        out << "# Orbifold Tutte parameterization result\n";

        for (const auto& v : vertices) {
            out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
        }
        for (const auto& t : uv) {
            out << "vt " << t.x() << " " << t.y() << "\n";
        }
        // 顶点索引和纹理索引都假设一一对应
        for (const auto& f : faces) {
            int i0 = f.x() + 1;
            int i1 = f.y() + 1;
            int i2 = f.z() + 1;
            out << "f "
                << i0 << "/" << i0 << " "
                << i1 << "/" << i1 << " "
                << i2 << "/" << i2 << "\n";
        }
        return true;
    }

private:
    static std::uint64_t edgeKey(int i, int j) {
        if (i > j) std::swap(i, j);
        return (static_cast<std::uint64_t>(i) << 32) |
               static_cast<std::uint64_t>(j);
    }

    void buildAdjacencyAndWeights() {
        const int n = static_cast<int>(vertices.size());
        vertexNeighbors.assign(n, {});
        edgeCotWeights.clear();

        // 为了避免重复邻接，先用临时的邻接集合（用 map<bool> 也行）
        std::vector<std::unordered_map<int, bool>> nbrSet(n);

        // 遍历每个面，构造 cotan 权重
        for (const auto& f : faces) {
            int i0 = f[0];
            int i1 = f[1];
            int i2 = f[2];

            const Eigen::Vector3d& p0 = vertices[i0];
            const Eigen::Vector3d& p1 = vertices[i1];
            const Eigen::Vector3d& p2 = vertices[i2];

            // 角点分别为 p0, p1, p2，对应对边分别是 (i1,i2), (i2,i0), (i0,i1)
            auto accumulateEdge = [&](int vi, int vj, int vk,
                                      const Eigen::Vector3d& pi,
                                      const Eigen::Vector3d& pj,
                                      const Eigen::Vector3d& pk) {

                Eigen::Vector3f u = (pj - pi).cast<float>();
                Eigen::Vector3f v = (pk - pi).cast<float>();

                float area2 = u.cross(v).norm();
                if (area2 < 1e-12f) return;

                float dot = u.dot(v);
                float cot = dot / area2;

                double w = 0.5 * cot;
                std::uint64_t key = edgeKey(vj, vk);
                edgeCotWeights[key] += w;
            };


            accumulateEdge(i0, i1, i2, p0, p1, p2);
            accumulateEdge(i1, i2, i0, p1, p2, p0);
            accumulateEdge(i2, i0, i1, p2, p0, p1);

            // 构建一圈邻接
            auto addNbr = [&](int a, int b) {
                if (a == b) return;
                nbrSet[a][b] = true;
                nbrSet[b][a] = true;
            };
            addNbr(i0, i1);
            addNbr(i1, i2);
            addNbr(i2, i0);
        }

        vertexNeighbors.clear();
        vertexNeighbors.resize(n);
        for (int i = 0; i < n; ++i) {
            vertexNeighbors[i].reserve(nbrSet[i].size());
            for (auto& kv : nbrSet[i]) {
                vertexNeighbors[i].push_back(kv.first);
            }
        }
    }
};
