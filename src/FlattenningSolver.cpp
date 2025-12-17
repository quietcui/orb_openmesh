#include "FlatteningSolver.h"

#include <Eigen/SparseLU>
#include <stdexcept>
#include <vector>
#include <iostream>

// A close translation of MATLAB computeFlattening.m:
//   M = [L A'; A 0]
//   rhs = [0; b]
//   x_lambda = M \ rhs
//   x = x_lambda(1:n_vars)

Eigen::VectorXd computeFlatteningCxx(
        const Eigen::SparseMatrix<double>& L,
        const Eigen::SparseMatrix<double>& A,
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

    // Build sparse KKT matrix M
    //   [ L  A^T ]
    //   [ A   0  ]
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(L.nonZeros()) + 2ULL * static_cast<size_t>(A.nonZeros()));

    // Top-left: L
    for (int k = 0; k < L.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            trips.emplace_back(it.row(), it.col(), it.value());
    }

    // Top-right: A^T and bottom-left: A
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
        {
            const int r = it.row();
            const int c = it.col();
            const double v = it.value();
            trips.emplace_back(c, n_vars + r, v);       // A^T
            trips.emplace_back(n_vars + r, c, v);       // A
        }
    }

    Eigen::SparseMatrix<double> M(N, N);
    M.setFromTriplets(trips.begin(), trips.end());

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N);
    rhs.segment(n_vars, n_eq) = b;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(M);
    solver.factorize(M);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("computeFlatteningCxx: KKT factorization failed");

    const Eigen::VectorXd x_lambda = solver.solve(rhs);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("computeFlatteningCxx: KKT solve failed");

    // MATLAB prints warning if residual > 1e-6
    const Eigen::VectorXd res = M * x_lambda - rhs;
    const double e = res.cwiseAbs().maxCoeff();
    if (e > 1e-6)
        std::cerr << "[computeFlatteningCxx] Warning: linear solve residual = " << e << " (>1e-6)\n";

    return x_lambda.head(n_vars);
}
