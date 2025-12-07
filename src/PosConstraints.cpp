//
// Created by cgl on 2025/12/7.
//
// PosConstraints.cpp
#include "PosConstraints.h"
#include <stdexcept>

PosConstraints::PosConstraints(int nvars)
        : nvars_(nvars)
        , ncols_(2 * nvars)
{
    if (nvars_ <= 0)
        throw std::runtime_error("PosConstraints: nvars must be positive");
}

void PosConstraints::addConstraint(int ind, double w, const Eigen::Vector2d& rhs)
{
    if (ind < 0 || ind >= nvars_)
        throw std::runtime_error("PosConstraints::addConstraint: index out of range");

    Eigen::RowVectorXd rowX = Eigen::RowVectorXd::Zero(ncols_);
    Eigen::RowVectorXd rowY = Eigen::RowVectorXd::Zero(ncols_);

    int col_x = 2 * ind;
    int col_y = 2 * ind + 1;

    rowX(col_x) = w;
    rowY(col_y) = w;

    rows_.push_back(rowX);
    b_.push_back(rhs(0));

    rows_.push_back(rowY);
    b_.push_back(rhs(1));
}

void PosConstraints::addLineConstraint(int ind,
                                       const Eigen::Vector2d& n,
                                       double offset)
{
    if (ind < 0 || ind >= nvars_)
        throw std::runtime_error("PosConstraints::addLineConstraint: index out of range");

    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(ncols_);
    int col_x = 2 * ind;
    int col_y = 2 * ind + 1;

    row(col_x) = n(0);
    row(col_y) = n(1);

    rows_.push_back(row);
    b_.push_back(offset);
}

void PosConstraints::addTransConstraints(const std::vector<int>& sinds,
                                         const std::vector<int>& tinds,
                                         const Eigen::Matrix2d& T)
{
    if (sinds.size() != tinds.size())
        throw std::runtime_error("PosConstraints::addTransConstraints: size mismatch");
    if (sinds.size() < 2)
        return;

    const int len = static_cast<int>(sinds.size());
    int s1 = sinds[0];
    int t1 = tinds[0];

    if (s1 < 0 || s1 >= nvars_ || t1 < 0 || t1 >= nvars_)
        throw std::runtime_error("PosConstraints::addTransConstraints: index out of range");

    for (int idx = 1; idx < len; ++idx)
    {
        int si = sinds[idx];
        int ti = tinds[idx];

        if (si < 0 || si >= nvars_ || ti < 0 || ti >= nvars_)
            throw std::runtime_error("PosConstraints::addTransConstraints: index out of range");

        for (int comp = 0; comp < 2; ++comp) // 0=x, 1=y
        {
            Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(ncols_);

            int col_si_x = 2 * si;
            int col_si_y = 2 * si + 1;
            row(col_si_x) += T(comp, 0);
            row(col_si_y) += T(comp, 1);

            int col_s1_x = 2 * s1;
            int col_s1_y = 2 * s1 + 1;
            row(col_s1_x) -= T(comp, 0);
            row(col_s1_y) -= T(comp, 1);

            int col_ti = 2 * ti + comp;
            row(col_ti) -= 1.0;

            int col_t1 = 2 * t1 + comp;
            row(col_t1) += 1.0;

            rows_.push_back(row);
            b_.push_back(0.0);
        }
    }
}

Eigen::MatrixXd PosConstraints::getA() const
{
    const int m = static_cast<int>(rows_.size());
    Eigen::MatrixXd A(m, ncols_);
    for (int i = 0; i < m; ++i)
        A.row(i) = rows_[i];
    return A;
}

Eigen::VectorXd PosConstraints::getB() const
{
    const int m = static_cast<int>(b_.size());
    Eigen::VectorXd b(m);
    for (int i = 0; i < m; ++i)
        b(i) = b_[i];
    return b;
}
