// PosConstraints.h
#pragma once

#include <Eigen/Core>
#include <vector>

// 对应 Matlab 的 PosConstraints，变量顺序：
// [x0x,x0y,x1x,x1y,...,x_{n-1}x,x_{n-1}y]^T
class PosConstraints
{
public:
    explicit PosConstraints(int nvars); // nvars = 顶点数

    // addConstraint(ind, w, rhs), 约束:
    // w * x_ind_x = rhs(0)
    // w * x_ind_y = rhs(1)
    void addConstraint(int ind, double w, const Eigen::Vector2d& rhs);

    // addLineConstraint(ind, n, offset), 约束:
    // <x_ind, n> = offset
    void addLineConstraint(int ind, const Eigen::Vector2d& n, double offset);

    // addTransConstraints(sinds, tinds, T):
    // 对所有 i>=1:
    //   T*x_si - T*x_s1 - x_ti + x_t1 = 0
    void addTransConstraints(const std::vector<int>& sinds,
                             const std::vector<int>& tinds,
                             const Eigen::Matrix2d& T);

    Eigen::MatrixXd getA() const;  // m x 2n
    Eigen::VectorXd getB() const;  // m

    int numConstraints() const { return static_cast<int>(rows_.size()); }

private:
    int nvars_;
    int ncols_;
    std::vector<Eigen::RowVectorXd> rows_;
    std::vector<double> b_;
};
