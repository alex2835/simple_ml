#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "layer.hpp"
#include "model.hpp"
#include "visualize.hpp"
#include "mnist/mnist_reader.hpp"

// Pack images into a [784 x N] matrix, one image per column, pixels scaled to [0, 1].
Mat to_matrix(const std::vector<std::vector<uint8_t>> &images)
{
    Mat X(images[0].size(), images.size());
    for (size_t i = 0; i < images.size(); ++i)
        for (size_t p = 0; p < images[i].size(); ++p)
            X(p, i) = images[i][p] / 255.0f;
    return X;
}

// Labels -> one-hot matrix [10 x N].
Mat to_one_hot(const std::vector<uint8_t> &labels, int classes)
{
    Mat Y = Mat::Zero(classes, labels.size());
    for (size_t i = 0; i < labels.size(); ++i)
        Y(labels[i], i) = 1.0f;
    return Y;
}

int main()
{
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    Mat X_train = to_matrix(dataset.training_images); // [784 x 60000]
    Mat Y_train = to_one_hot(dataset.training_labels, 10);
    Mat X_test  = to_matrix(dataset.test_images);     // [784 x 10000]

    Dense model(784, 10);

    std::cout << "before training: test accuracy "
              << accuracy(model.forward(X_test), dataset.test_labels) << "\n";

    const int   batch_size = 128;
    const float lr         = 0.1f;
    const int   epochs     = 10;

    std::mt19937 rng(42);
    std::vector<int> order(X_train.cols());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        std::shuffle(order.begin(), order.end(), rng);

        float loss_sum = 0;
        int   batches  = 0;
        for (int start = 0; start + batch_size <= int(order.size()); start += batch_size)
        {
            Mat X(X_train.rows(), batch_size);
            Mat Y(Y_train.rows(), batch_size);
            for (int i = 0; i < batch_size; ++i)
            {
                X.col(i) = X_train.col(order[start + i]);
                Y.col(i) = Y_train.col(order[start + i]);
            }

            Mat P = softmax(model.forward(X));
            loss_sum += cross_entropy(P, Y);
            ++batches;

            // For softmax + cross-entropy, dL/dZ = P - Y (averaged over the batch)
            model.backward((P - Y) / float(batch_size));
            model.step(lr);
        }

        std::cout << "epoch " << epoch
                  << "  train loss " << loss_sum / batches
                  << "  test accuracy "
                  << accuracy(model.forward(X_test), dataset.test_labels) << "\n";
    }

    save_weight_images(model, 28, 28, "weights");
    std::cout << "saved weight images to weights/\n";

    return 0;
}
