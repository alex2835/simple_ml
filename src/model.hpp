#pragma once
#include "layer.hpp"

// Column-wise softmax: turns raw scores Z [10 x B] into probabilities.
// Subtracting the max of each column first keeps exp() from overflowing.
inline Mat softmax(const Mat &Z)
{
    Mat P = (Z.rowwise() - Z.colwise().maxCoeff()).array().exp();
    P.array().rowwise() /= P.colwise().sum().array();
    return P;
}

// Average cross-entropy over the batch. Y is one-hot [10 x B].
inline float cross_entropy(const Mat &P, const Mat &Y)
{
    return -((P.array() + 1e-8f).log() * Y.array()).sum() / float(P.cols());
}

// Fraction of columns where the highest score matches the label.
inline float accuracy(const Mat &Z, const std::vector<uint8_t> &labels)
{
    int correct = 0;
    for (int i = 0; i < Z.cols(); ++i)
    {
        Eigen::Index pred;
        Z.col(i).maxCoeff(&pred);
        correct += (pred == labels[i]);
    }
    return float(correct) / float(Z.cols());
}
