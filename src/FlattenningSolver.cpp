#include "FlatteningSolver.h"

// 注意：这里改成 SparseCholesky
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <stdexcept>
#include <vector>
#include <iostream>

Eigen::VectorXd computeFlatteningCxx(
        const Eigen::SparseMatrix<double>& L,
        const Eigen::MatrixXd&             A,
        const Eigen::VectorXd&             b)
{
    const int n_vars = static_cast<int>(L.rows());
    if (L.cols() != n_vars)
        throw std::runtime_error("computeFlatteningCxx: L must be square");

    const int n_eq = static_cast<int>(A.rows());
    if (A.cols() != n_vars)
        throw std::runtime_error("computeFlatteningCxx: A.cols() != L.rows()");
    if (b.size() != n_eq)
        throw std::runtime_error("computeFlatteningCxx: b.size() != A.rows()");

    const int N = n_vars + n_eq;

    // ------------------------------------
    // 1) 先尝试：和 Matlab 一样的 KKT 系统
    //      M = [ L  A^T;
    //            A   0 ]
    // ------------------------------------
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(L.nonZeros() * 2 + n_vars * n_eq * 2);

    // 左上角 L
    for (int k = 0; k < L.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
        {
            int i = it.row();
            int j = it.col();
            double v = it.value();
            triplets.emplace_back(i, j, v);
        }
    }

    // 右上角 A^T
    for (int r = 0; r < n_eq; ++r)
    {
        for (int c = 0; c < n_vars; ++c)
        {
            double v = A(r, c);
            if (v == 0.0) continue;
            int row = c;
            int col = n_vars + r;
            triplets.emplace_back(row, col, v);
        }
    }

    // 左下角 A
    for (int r = 0; r < n_eq; ++r)
    {
        for (int c = 0; c < n_vars; ++c)
        {
            double v = A(r, c);
            if (v == 0.0) continue;
            int row = n_vars + r;
            int col = c;
            triplets.emplace_back(row, col, v);
        }
    }

    Eigen::SparseMatrix<double> M(N, N);
    M.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N);
    rhs.segment(n_vars, n_eq) = b;

    Eigen::VectorXd x_lambda;
    bool kkt_ok = true;

    {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(M);
        solver.factorize(M);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "[computeFlatteningCxx] KKT factorization failed, "
                      << "fallback to soft-constraint solver.\n";
            kkt_ok = false;
        }
        else
        {
            x_lambda = solver.solve(rhs);
            if (solver.info() != Eigen::Success)
            {
                std::cerr << "[computeFlatteningCxx] KKT solve failed, "
                          << "fallback to soft-constraint solver.\n";
                kkt_ok = false;
            }
            else
            {
                Eigen::VectorXd res = M * x_lambda - rhs;
                double err = res.lpNorm<Eigen::Infinity>();
                if (err > 1e-5)
                {
                    std::cerr << "[computeFlatteningCxx] KKT residual = "
                              << err << " > 1e-5, "
                              << "fallback to soft-constraint solver.\n";
                    kkt_ok = false;
                }
            }
        }
    }

    if (kkt_ok)
    {
        return x_lambda.head(n_vars);
    }

    // ------------------------------------
    // 2) 备用方案：软约束 (L + α A^T A)x = α A^T b
    // ------------------------------------
    const double alpha = 1e5;

    // 把 A 变成稀疏
    Eigen::SparseMatrix<double> As(A.rows(), A.cols());
    {
        std::vector<Eigen::Triplet<double>> Atrips;
        Atrips.reserve(A.rows() * A.cols());
        for (int r = 0; r < A.rows(); ++r)
        {
            for (int c = 0; c < A.cols(); ++c)
            {
                double v = A(r, c);
                if (v != 0.0)
                    Atrips.emplace_back(r, c, v);
            }
        }
        As.setFromTriplets(Atrips.begin(), Atrips.end());
    }

    Eigen::SparseMatrix<double> ATA = As.transpose() * As;

    Eigen::SparseMatrix<double> H = L;
    {
        std::vector<Eigen::Triplet<double>> Htrips;
        Htrips.reserve(H.nonZeros() + ATA.nonZeros());
        for (int k = 0; k < H.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it)
                Htrips.emplace_back(it.row(), it.col(), it.value());
        }
        for (int k = 0; k < ATA.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(ATA, k); it; ++it)
                Htrips.emplace_back(it.row(), it.col(), alpha * it.value());
        }
        H.setFromTriplets(Htrips.begin(), Htrips.end());
    }

    Eigen::VectorXd rhs2 = alpha * As.transpose() * b;

    // 注意：SimplicialLDLT 定义在 <Eigen/SparseCholesky> 里
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver2;
    solver2.compute(H);
    if (solver2.info() != Eigen::Success)
    {
        std::cerr << "[computeFlatteningCxx] Soft-constraint factorization failed.\n";
        return Eigen::VectorXd::Zero(n_vars);
    }

    Eigen::VectorXd x = solver2.solve(rhs2);
    if (solver2.info() != Eigen::Success)
    {
        std::cerr << "[computeFlatteningCxx] Soft-constraint solve failed.\n";
        return Eigen::VectorXd::Zero(n_vars);
    }

    return x;
}
