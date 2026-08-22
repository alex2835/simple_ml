#pragma once
#include <Eigen/Dense>
using Mat = Eigen::MatrixXf;
using Vec = Eigen::VectorXf;

struct Dense
{
    Mat W, dW; // [m x n]
    Vec b, db; // [m]
    Mat X;     // cached input, [n x B]

    Dense(int n, int m)
        : W(Mat::Random(m, n) * std::sqrt(2.0f / n)),
          dW(Mat::Zero(m, n)),
          b(Vec::Zero(m)),
          db(Vec::Zero(m))
    {}

    Mat forward(const Mat &x)
    {
        X = x;
        return (W * X).colwise() + b; // [m x B]
    }

    Mat backward(const Mat &dY)
    {
        dW = dY * X.transpose();   // [m x B][B x n] -> [m x n]
        db = dY.rowwise().sum();   // [m]
        return W.transpose() * dY; // [n x m][m x B] -> [n x B]
    }

    // l2 is the weight-decay strength: every step shrinks each weight
    // toward zero in proportion to its own size (gradient of l2/2 * sum(W^2))
    void step(float lr, float l2 = 0.0f)
    {
        W -= lr * (dW + l2 * W);
        b -= lr * db;
    }
};