// PosConstraints.cpp
//
// Sparse constraint accumulator mirroring the MATLAB implementation in
// euclidean_orbifolds/PosConstraints.m.
//
// Variable ordering:
//   [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}]^T

#include "PosConstraints.h"

#include <stdexcept>

PosConstraints::PosConstraints(int nvars)
        : nvars_(nvars)
        , ncols_(2 * nvars)
{
    if (nvars_ <= 0)
        throw std::runtime_error("PosConstraints: nvars must be positive");

    trips_.reserve(128);//存稀疏矩阵 A 的非零项（三元组 row/col/value）
    b_.reserve(128);//存右侧向量 b 的每一行
}

void PosConstraints::addConstraint(int ind, double w, const Eigen::Vector2d& rhs)
{
    if (ind < 0 || ind >= nvars_)
        throw std::runtime_error("PosConstraints::addConstraint: index out of range");

    const int col_x = 2 * ind;
    const int col_y = 2 * ind + 1;

    // Row for x
    trips_.emplace_back(nrows_, col_x, w);
    b_.push_back(rhs(0));
    ++nrows_;

    // Row for y
    trips_.emplace_back(nrows_, col_y, w);
    b_.push_back(rhs(1));
    ++nrows_;
}

void PosConstraints::addLineConstraint(int ind,
                                       const Eigen::Vector2d& n,
                                       double offset)
{
    if (ind < 0 || ind >= nvars_)
        throw std::runtime_error("PosConstraints::addLineConstraint: index out of range");

    const int col_x = 2 * ind;
    const int col_y = 2 * ind + 1;

    trips_.emplace_back(nrows_, col_x, n(0));
    trips_.emplace_back(nrows_, col_y, n(1));
    b_.push_back(offset);
    ++nrows_;
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

    const int s1 = sinds[0];
    const int t1 = tinds[0];
    if (s1 < 0 || s1 >= nvars_ || t1 < 0 || t1 >= nvars_)
        throw std::runtime_error("PosConstraints::addTransConstraints: index out of range");

    // MATLAB:
    // for ind = 2:length(sinds)
    //   si=sinds(ind); ti=tinds(ind);
    //   for vert_ind = 1:2
    //     A(end+1, si*2-1:si*2) = T(vert_ind,:);
    //     A(end,   s1*2-1:s1*2) = A(end, s1*2-1:s1*2) - T(vert_ind,:);
    //     A(end,   ti*2+vert_ind-2) = -1;
    //     A(end,   t1*2+vert_ind-2) =  1;
    //     b(end+1) = 0;
    //   end
    // end

    for (int idx = 1; idx < len; ++idx)
    {
        const int si = sinds[idx];
        const int ti = tinds[idx];
        if (si < 0 || si >= nvars_ || ti < 0 || ti >= nvars_)
            throw std::runtime_error("PosConstraints::addTransConstraints: index out of range");

        for (int comp = 0; comp < 2; ++comp) // 0=x, 1=y
        {
            const int row = nrows_;

            // + T(comp,:) * x_si
            trips_.emplace_back(row, 2 * si,     T(comp, 0));
            trips_.emplace_back(row, 2 * si + 1, T(comp, 1));

            // - T(comp,:) * x_s1
            trips_.emplace_back(row, 2 * s1,     -T(comp, 0));
            trips_.emplace_back(row, 2 * s1 + 1, -T(comp, 1));

            // - x_ti_comp + x_t1_comp
            trips_.emplace_back(row, 2 * ti + comp, -1.0);
            trips_.emplace_back(row, 2 * t1 + comp,  1.0);

            b_.push_back(0.0);
            ++nrows_;
        }
    }
}

Eigen::SparseMatrix<double> PosConstraints::getA() const
{
    Eigen::SparseMatrix<double> A(nrows_, ncols_);
    A.setFromTriplets(trips_.begin(), trips_.end());
    return A;
}

Eigen::VectorXd PosConstraints::getB() const
{
    Eigen::VectorXd b(nrows_);
    for (int i = 0; i < nrows_; ++i)
        b(i) = b_[static_cast<size_t>(i)];
    return b;
}
