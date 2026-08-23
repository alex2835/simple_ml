#pragma once
#include "helpers.hpp"

struct DenseLayer
{
    Mat W, dW; // [m x n]
    Vec b, db; // [m]
    Mat X, A;     // cached input and output, [n x B] and [m x B]
    Activation activation;

    DenseLayer(int n, int m, Activation act)
        : W(Mat::Random(m, n) * std::sqrt(2.0f / n)),
          dW(Mat::Zero(m, n)),
          b(Vec::Zero(m)),
          db(Vec::Zero(m)),
          activation(act)
    {}

    Mat forward(const Mat &x)
    {
        X = x;
        Mat Z = (W * X).colwise() + b;
        const auto activaction_func = get_activation(activation);
        A = activaction_func(Z);  // [m x B]
        return A;
    }

    Mat backward(const Mat &dY)
    {
        const auto act_deriv = get_derivative(activation);
        Mat dZ = dY.cwiseProduct(act_deriv(A)); // chain rule through the activation
        dW = dZ * X.transpose();
        db = dZ.rowwise().sum();
        return W.transpose() * dZ;
    }

    // l2 is the weight-decay strength: every step shrinks each weight
    // toward zero in proportion to its own size (gradient of l2/2 * sum(W^2))
    void step(float lr, float l2 = 0.0f)
    {
        W -= lr * (dW + l2 * W);
        b -= lr * db;
    }
};