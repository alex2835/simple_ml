#include <iostream>
#include "train.hpp"
#include "visualize.hpp"
#include "mnist/mnist_reader.hpp"

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

int main()
{
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    Mat X_train = to_matrix(dataset.training_images); // [784 x 60000]
    Mat Y_train = to_one_hot(dataset.training_labels, 10);
    Mat X_test  = to_matrix(dataset.test_images);     // [784 x 10000]

    TrainConfig experiments[] = {
        {.name   = "linear",
         .layers = {{Activation::Linear, 784, 10}}},

        {.name   = "linear_l2",
         .layers = {{Activation::Linear, 784, 10}},
         .l2     = 1e-3f},

        {.name   = "mlp_relu_128",
         .layers = {{Activation::ReLU, 784, 128},
                    {Activation::Linear, 128, 10}}},
    };

    for (const TrainConfig &cfg : experiments)
    {
        PerceptronModel model = train(X_train, Y_train, X_test, dataset.test_labels, cfg);

        // first layer's weight rows are 784 wide, so they reshape to 28x28 images
        std::string dir = "pictures/" + cfg.name;
        save_weight_images(model.layer(0), 28, 28, dir);
        std::cout << "saved weight images to " << dir << "/\n\n";
    }

    return 0;
}
