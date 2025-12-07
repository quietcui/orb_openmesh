#include "MeshUtils.h"

#include <Eigen/Sparse>
#include <cmath>
#include <Eigen/Geometry>   // ★ 为了 Vector3d::cross 等几何函数
// -------------------- 收集一圈边界顶点 --------------------

void collect_boundary_loop(const MyMesh& mesh,
                           std::vector<MyMesh::VertexHandle>& boundary)
{
    boundary.clear();

    // 找第一个边界半边
    MyMesh::HalfedgeHandle he_start;

    for (auto he : mesh.halfedges())
    {
        if (mesh.is_boundary(he))
        {
            he_start = he;
            break;
        }
    }

    if (!he_start.is_valid())
    {
        // 没有边界（闭合球），说明切缝还没做，这种情况 flatten_sphere 里会处理
        return;
    }

    // 按半边循环走一圈
    MyMesh::HalfedgeHandle he = he_start;
    do
    {
        MyMesh::VertexHandle vh = mesh.to_vertex_handle(he);
        boundary.push_back(vh);

        // 沿着边界往前走
        MyMesh::HalfedgeHandle next_he = mesh.next_halfedge_handle(he);
        while (!mesh.is_boundary(next_he))
        {
            // 如果 next 不是边界，一般是遇到了内部半边，跳到对边继续
            next_he = mesh.opposite_halfedge_handle(next_he);
            next_he = mesh.next_halfedge_handle(next_he);
        }

        he = next_he;

    } while (he != he_start);
}

// -------------------- 标准 cotan Laplacian --------------------

void compute_cotangent_laplacian(const MyMesh& mesh,
                                 Eigen::SparseMatrix<double>& L)
{
    const int n = static_cast<int>(mesh.n_vertices());

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(n * 7);

    std::vector<double> diag(n, 0.0);

    auto cot_angle_at = [](const Eigen::Vector3d& a,
                           const Eigen::Vector3d& b,
                           const Eigen::Vector3d& c) -> double
    {
        // 计算三角形 (a,b,c) 在 b 点的 cot(angle)
        Eigen::Vector3d u = a - b;
        Eigen::Vector3d v = c - b;
        Eigen::Vector3d cross = u.cross(v);
        double denom = cross.norm();
        if (denom < 1e-12)
            return 0.0;
        return u.dot(v) / denom;
    };

    for (auto eh : mesh.edges())
    {
        if (mesh.is_boundary(eh))
            continue;

        // edge 有两个半边
        auto he0 = mesh.halfedge_handle(eh, 0);
        auto he1 = mesh.halfedge_handle(eh, 1);

        MyMesh::VertexHandle vi = mesh.to_vertex_handle(he0);
        MyMesh::VertexHandle vj = mesh.to_vertex_handle(he1);

        int i = vi.idx();
        int j = vj.idx();

        // 两个三角形各自的第三个点（k 和 l）
        MyMesh::HalfedgeHandle he0_next = mesh.next_halfedge_handle(he0);
        MyMesh::HalfedgeHandle he1_next = mesh.next_halfedge_handle(he1);

        MyMesh::VertexHandle vk = mesh.to_vertex_handle(he0_next);
        MyMesh::VertexHandle vl = mesh.to_vertex_handle(he1_next);

        Eigen::Vector3d pi = to_eigen(mesh.point(vi));
        Eigen::Vector3d pj = to_eigen(mesh.point(vj));
        Eigen::Vector3d pk = to_eigen(mesh.point(vk));
        Eigen::Vector3d pl = to_eigen(mesh.point(vl));

        // 对应两个三角形在对边的角的 cot 值
        double w = 0.0;
        // 三角形 (i,k,j) 在 k 点的角
        w += cot_angle_at(pi, pk, pj);
        // 三角形 (i,l,j) 在 l 点的角
        w += cot_angle_at(pi, pl, pj);

        w *= 0.5; // 标准 cotan 权重

        if (std::abs(w) < 1e-16)
            continue;

        diag[i] += w;
        diag[j] += w;

        trips.emplace_back(i, j, -w);
        trips.emplace_back(j, i, -w);
    }

    // 对角线
    for (int i = 0; i < n; ++i)
    {
        trips.emplace_back(i, i, diag[i]);
    }

    L.resize(n, n);
    L.setFromTriplets(trips.begin(), trips.end());
    L.makeCompressed();
}
