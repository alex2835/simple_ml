#pragma once
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "layer.hpp"
#include "model.hpp"

// Pack images into a [784 x N] matrix, one image per column, pixels scaled to [0, 1].
inline Mat to_matrix(const std::vector<std::vector<uint8_t>> &images)
{
    Mat X(images[0].size(), images.size());
    for (size_t i = 0; i < images.size(); ++i)
        for (size_t p = 0; p < images[i].size(); ++p)
            X(p, i) = images[i][p] / 255.0f;
    return X;
}

// Labels -> one-hot matrix [10 x N].
inline Mat to_one_hot(const std::vector<uint8_t> &labels, int classes)
{
    Mat Y = Mat::Zero(classes, labels.size());
    for (size_t i = 0; i < labels.size(); ++i)
        Y(labels[i], i) = 1.0f;
    return Y;
}

struct TrainConfig
{
    std::string name;
    int epochs = 10;
    int batch_size = 128;
    float lr = 0.1f;
    float l2 = 0.0f; // L2 / weight-decay strength, 0 = off
};

// Mini-batch SGD on softmax + cross-entropy. Prints progress per epoch
// and returns the trained layer.
inline Dense train(const Mat &X_train, const Mat &Y_train,
                   const Mat &X_test, const std::vector<uint8_t> &test_labels,
                   const TrainConfig &cfg)
{
    std::cout << "=== " << cfg.name << " (l2 = " << cfg.l2 << ") ===\n";

    Dense model(int(X_train.rows()), int(Y_train.rows()));

    std::mt19937 rng(42);
    std::vector<int> order(X_train.cols());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch)
    {
        std::shuffle(order.begin(), order.end(), rng);

        float loss_sum = 0;
        int   batches  = 0;
        for (int start = 0; start + cfg.batch_size <= int(order.size()); start += cfg.batch_size)
        {
            Mat X(X_train.rows(), cfg.batch_size);
            Mat Y(Y_train.rows(), cfg.batch_size);
            for (int i = 0; i < cfg.batch_size; ++i)
            {
                X.col(i) = X_train.col(order[start + i]);
                Y.col(i) = Y_train.col(order[start + i]);
            }
            
            Mat P = softmax(model.forward(X));
            loss_sum += cross_entropy(P, Y);
            ++batches;

            // For softmax + cross-entropy, dL/dZ = P - Y (averaged over the batch)
            model.backward((P - Y) / float(cfg.batch_size));
            model.step(cfg.lr, cfg.l2);
        }

        std::cout << "epoch " << epoch
                  << "  train loss " << loss_sum / batches
                  << "  test accuracy "
                  << accuracy(model.forward(X_test), test_labels) << "\n";
    }

    return model;
}
