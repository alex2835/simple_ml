#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>
using Mat = Eigen::MatrixXf;
using Vec = Eigen::VectorXf;

enum class Activation
{
    Linear,
    ReLU,
    Sigmoid,
    Tanh
};

inline std::function<Mat(const Mat&)> get_activation(Activation act)
{
    switch (act)
    {
        case Activation::Linear:  return [](const Mat &x) { return x; };
        case Activation::ReLU:    return [](const Mat &x) { return x.cwiseMax(0.0f); };
        case Activation::Sigmoid: return [](const Mat &x) { return 1.0f / (1.0f + (-x.array()).exp()); };
        case Activation::Tanh:    return [](const Mat &x) { return x.array().tanh(); };
        default: throw std::invalid_argument("Unknown activation function");
    }
}

// NOTE: the argument is the activation OUTPUT a = f(z), not z itself —
// sigmoid' = a(1-a), tanh' = 1-a², ReLU' = [a > 0].
inline std::function<Mat(const Mat&)> get_derivative(Activation act)
{
    switch (act)
    {
        case Activation::Linear:  return [](const Mat &x) { return Mat::Ones(x.rows(), x.cols()); };
        case Activation::ReLU:    return [](const Mat &x) { return (x.array() > 0.0f).cast<float>(); };
        case Activation::Sigmoid: return [](const Mat &x) { return x.array() * (1.0f - x.array()); };
        case Activation::Tanh:    return [](const Mat &x) { return 1.0f - x.array().square(); };
        default: throw std::invalid_argument("Unknown activation function");
    }
}

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
