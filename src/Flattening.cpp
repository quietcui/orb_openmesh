#include "Flattening.h"
#include "MeshUtils.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <unordered_map>

// -------------------- Orbifold-type boundary (3 cone points) --------------------
// Idea:
//  - boundaryLoop: one ordered loop of boundary vertices
//  - cones: 0-based vertex indices of the 3 cone points
//  - place them on the unit circle at 0°, 120°, 240°
//  - split the boundary loop into 3 segments and linearly interpolate angles.

static void assign_orbifold_boundary_typeI(
        const MyMesh& mesh,
        const std::vector<MyMesh::VertexHandle>& boundaryLoop,
        const std::vector<int>& cones,
        Eigen::MatrixXd& X_fixed)
{
    const int nB = static_cast<int>(boundaryLoop.size());
    X_fixed.setZero(nB, 2);

    if (nB == 0)
    {
        std::cerr << "[assign_orbifold_boundary_typeI] boundary loop is empty\n";
        return;
    }

    if (cones.size() != 3)
    {
        std::cerr << "[assign_orbifold_boundary_typeI] cones.size() != 3. "
                  << "This implementation currently only supports 3 cone points.\n"
                  << "Falling back to a simple uniform circle boundary.\n";

        for (int i = 0; i < nB; ++i)
        {
            double t = static_cast<double>(i) / static_cast<double>(nB);
            double angle = 2.0 * M_PI * t;
            X_fixed(i, 0) = std::cos(angle);
            X_fixed(i, 1) = std::sin(angle);
        }
        return;
    }

    auto wrap = [nB](int i)
    {
        if (i < 0) i += nB;
        if (i >= nB) i -= nB;
        return i;
    };

    // 1. Find each cone point's position on the boundary loop.
    std::vector<int> conePosOnBoundary;
    conePosOnBoundary.reserve(3);

    for (int c = 0; c < 3; ++c)
    {
        int cone_vid = cones[c];
        int pos = -1;
        for (int bi = 0; bi < nB; ++bi)
        {
            if (boundaryLoop[bi].idx() == cone_vid)
            {
                pos = bi;
                break;
            }
        }
        if (pos < 0)
        {
            std::cerr << "[assign_orbifold_boundary_typeI] Cone vertex " << cone_vid
                      << " is not on the boundary loop. This often means the cone "
                         "vertex was duplicated during cutting.\n"
                      << "We will snap it to the nearest boundary vertex.\n";

            // Snap to nearest boundary vertex in 3D.
            double best_dist = 1e100;
            int best_b = 0;
            const MyMesh::Point& pc = mesh.point(MyMesh::VertexHandle(cone_vid));
            for (int bi = 0; bi < nB; ++bi)
            {
                const MyMesh::Point& pb = mesh.point(boundaryLoop[bi]);
                MyMesh::Point diff = pc - pb;
                double d0 = diff.sqrnorm();
                if (d0 < best_dist)
                {
                    best_dist = d0;
                    best_b = bi;
                }
            }
            pos = best_b;
        }
        conePosOnBoundary.push_back(pos);
    }

    // 2. Sort cone positions cyclically to have a consistent order.
    int start_idx = 0;
    int smallest_pos = conePosOnBoundary[0];
    for (int k = 1; k < 3; ++k)
    {
        if (conePosOnBoundary[k] < smallest_pos)
        {
            smallest_pos = conePosOnBoundary[k];
            start_idx = k;
        }
    }

    std::vector<int> conePos(3);
    std::vector<double> coneAngles(3);
    for (int k = 0; k < 3; ++k)
    {
        conePos[k]    = conePosOnBoundary[(start_idx + k) % 3];
        coneAngles[k] = 2.0 * M_PI * (static_cast<double>(k) / 3.0); // 0°, 120°, 240°
    }

    // 3. Interpolate along each segment between consecutive cone points.
    for (int seg = 0; seg < 3; ++seg)
    {
        int pos_start = conePos[seg];
        int pos_end   = conePos[(seg + 1) % 3];

        double angle_start = coneAngles[seg];
        double angle_end   = coneAngles[(seg + 1) % 3];

        if (angle_end < angle_start)
            angle_end += 2.0 * M_PI;

        std::vector<int> segment;
        int cur = pos_start;
        segment.push_back(cur);
        while (cur != pos_end)
        {
            cur = wrap(cur + 1);
            segment.push_back(cur);
        }

        const int len = static_cast<int>(segment.size());
        for (int local = 0; local < len; ++local)
        {
            double t = (len == 1)
                       ? 0.0
                       : static_cast<double>(local) / static_cast<double>(len - 1);

            double angle = (1.0 - t) * angle_start + t * angle_end;
            int bi = segment[local];

            X_fixed(bi, 0) = std::cos(angle);
            X_fixed(bi, 1) = std::sin(angle);
        }
    }
}

// --------------------------- Main flattening function ---------------------------
void flatten_sphere(
        MyMesh& mesh,
        const std::vector<int>& cones,
        int orbifold_type)
{
    std::cout << "\n========== [flatten_sphere] Start ==========\n";

    const int n = static_cast<int>(mesh.n_vertices());
    std::cout << "[Info] Mesh has " << n << " vertices.\n";

    if (n == 0)
    {
        std::cerr << "[Error] Mesh is empty.\n";
        return;
    }

    // 1. Build cotan Laplacian
    std::cout << "[Step] Building cotangent Laplacian...\n";
    Eigen::SparseMatrix<double> L;
    compute_cotangent_laplacian(mesh, L);
    std::cout << "[Info] Laplacian built. Non-zeros = " << L.nonZeros() << "\n";

    // 2. Collect boundary vertices
    std::cout << "[Step] Collecting boundary loop...\n";
    std::vector<MyMesh::VertexHandle> boundaryLoop;
    collect_boundary_loop(mesh, boundaryLoop);

    if (boundaryLoop.empty())
    {
        std::cerr << "[Warning] Mesh has NO boundary. It is still a closed surface.\n";
        std::cerr << "[Warning] Cutting step has NOT been implemented or called.\n";
        std::cerr << "[Warning] Flattening cannot proceed.\n";
        std::cout << "========== [flatten_sphere] End ==========\n";
        return;
    }

    const int nB = static_cast<int>(boundaryLoop.size());
    const int nI = n - nB;

    std::cout << "[Info] Boundary vertices = " << nB << "\n";
    std::cout << "[Info] Interior vertices = " << nI << "\n";

    // 3. Build index mapping
    std::cout << "[Step] Building interior/boundary index mapping...\n";

    std::vector<int> isBoundary(n, 0);
    std::vector<int> bIndex(n, -1);
    std::vector<int> iIndex(n, -1);

    for (int bi = 0; bi < nB; ++bi)
    {
        int vid = boundaryLoop[bi].idx();
        isBoundary[vid] = 1;
        bIndex[vid] = bi;
    }

    int curI = 0;
    for (int v = 0; v < n; ++v)
    {
        if (!isBoundary[v])
            iIndex[v] = curI++;
    }

    std::cout << "[Info] Finished index mapping.\n";

    // 4. Split Laplacian
    std::cout << "[Step] Splitting Laplacian into blocks L_ii and L_ib...\n";

    Eigen::SparseMatrix<double> L_ii(nI, nI);
    Eigen::SparseMatrix<double> L_ib(nI, nB);
    std::vector<Eigen::Triplet<double>> trips_ii;
    std::vector<Eigen::Triplet<double>> trips_ib;

    for (int k = 0; k < L.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
        {
            int row = it.row();
            int col = it.col();
            double val = it.value();

            if (!isBoundary[row])
            {
                int r = iIndex[row];

                if (!isBoundary[col])
                    trips_ii.emplace_back(r, iIndex[col], val);
                else
                    trips_ib.emplace_back(r, bIndex[col], val);
            }
        }
    }

    L_ii.setFromTriplets(trips_ii.begin(), trips_ii.end());
    L_ib.setFromTriplets(trips_ib.begin(), trips_ib.end());

    std::cout << "[Info] L_ii nnz = " << L_ii.nonZeros()
              << ", L_ib nnz = " << L_ib.nonZeros() << "\n";

    // 5. Boundary constraints
    std::cout << "[Step] Assigning orbifold boundary constraints...\n";
    Eigen::MatrixXd X_b(nB, 2);
    assign_orbifold_boundary_typeI(mesh, boundaryLoop, cones, X_b);

    // 6. RHS
    std::cout << "[Step] Computing RHS...\n";
    Eigen::MatrixXd rhs = -L_ib * X_b;

    // 7. Solve system
    std::cout << "[Step] Solving linear system (LDLT)...\n";
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(L_ii);

    Eigen::MatrixXd X_i;

    if (solver.info() == Eigen::Success)
    {
        std::cout << "[Info] LDLT factorization successful.\n";
        X_i.resize(nI, 2);
        X_i.col(0) = solver.solve(rhs.col(0));
        X_i.col(1) = solver.solve(rhs.col(1));
    }
    else
    {
        std::cerr << "[Warning] LDLT factorization FAILED. Trying SparseLU...\n";

        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
        lu.analyzePattern(L_ii);
        lu.factorize(L_ii);

        if (lu.info() != Eigen::Success)
        {
            std::cerr << "[Error] SparseLU failed as well. Cannot flatten mesh.\n";
            std::cout << "========== [flatten_sphere] End ==========\n";
            return;
        }
        else
        {
            std::cout << "[Info] SparseLU succeeded.\n";
        }

        X_i.resize(nI, 2);
        X_i.col(0) = lu.solve(rhs.col(0));
        X_i.col(1) = lu.solve(rhs.col(1));
    }

    // 8. Write back to mesh
    std::cout << "[Step] Writing UV coordinates back to mesh...\n";

    for (int v = 0; v < n; ++v)
    {
        double x, y;
        if (isBoundary[v])
        {
            int bi = bIndex[v];
            x = X_b(bi, 0);
            y = X_b(bi, 1);
        }
        else
        {
            int ii = iIndex[v];
            x = X_i(ii, 0);
            y = X_i(ii, 1);
        }

        mesh.set_point(MyMesh::VertexHandle(v),
                       MyMesh::Point((float)x, (float)y, 0.0f));
    }

    std::cout << "[flatten_sphere] Completed successfully.\n";
    std::cout << "========== [flatten_sphere] End ==========\n\n";
}
