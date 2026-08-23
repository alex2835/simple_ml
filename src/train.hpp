#pragma once
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "dense_layer.hpp"
#include "model.hpp"

struct TrainConfig
{
    std::string name;
    // Layer stack; the last layer must be Activation::Linear because the
    // train loop folds softmax's derivative into dZ = P - Y itself.
    std::vector<LayerConfig> layers;
    int epochs = 10;
    int batch_size = 128;
    float lr = 0.1f;
    float l2 = 0.0f; // L2 / weight-decay strength, 0 = off
};

// Mini-batch SGD on softmax + cross-entropy. Prints progress per epoch
// and returns the trained model.
inline PerceptronModel train(const Mat &X_train,
                   const Mat &Y_train,
                   const Mat &X_test,
                   const std::vector<uint8_t> &test_labels,
                   const TrainConfig &cfg)
{
    std::cout << "=== " << cfg.name
              << " (" << cfg.layers.size() << " layers, l2 = " << cfg.l2 << ") ===\n";

    PerceptronModel model(cfg.layers);

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
