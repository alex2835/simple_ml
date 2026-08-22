#include <iostream>
#include "train.hpp"
#include "visualize.hpp"
#include "mnist/mnist_reader.hpp"

int main()
{
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    Mat X_train = to_matrix(dataset.training_images); // [784 x 60000]
    Mat Y_train = to_one_hot(dataset.training_labels, 10);
    Mat X_test  = to_matrix(dataset.test_images);     // [784 x 10000]

    TrainConfig experiments[] = {
        {.name = "baseline", .l2 = 0.0f},
        {.name = "l2_1e-3",  .l2 = 1e-3f},
        {.name = "l2_1e-2",  .l2 = 1e-2f},
    };

    for (const TrainConfig &cfg : experiments)
    {
        Dense model = train(X_train, Y_train, X_test, dataset.test_labels, cfg);

        std::string dir = "pictures/" + cfg.name;
        save_weight_images(model, 28, 28, dir);
        std::cout << "saved weight images to " << dir << "/\n\n";
    }

    return 0;
}
