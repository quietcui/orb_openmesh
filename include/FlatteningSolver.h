// FlatteningSolver.h
#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

// L:  n_vars x n_vars 稀疏 Laplacian
// A:  n_eq   x n_vars 稠密
// b:  n_eq   x 1
// 返回: x (n_vars x 1)
Eigen::VectorXd computeFlatteningCxx(
        const Eigen::SparseMatrix<double>& L,
        const Eigen::MatrixXd&             A,
        const Eigen::VectorXd&             b);
