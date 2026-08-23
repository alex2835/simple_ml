#pragma once
#include "dense_layer.hpp"
#include <ranges>

struct LayerConfig
{
    Activation activation;
    int in;
    int out;
};

class PerceptronModel
{
    std::vector<DenseLayer> layers;

public:
    PerceptronModel(const std::vector<LayerConfig>& layer_configs)
    {
        for (const LayerConfig &cfg : layer_configs)
            layers.emplace_back(cfg.in, cfg.out, cfg.activation);
    }

    Mat forward(const Mat &x)
    {
        Mat out = x;
        for (DenseLayer &layer : layers)
            out = layer.forward(out);
        return out;
    }

    void backward(const Mat &dY)
    {
        Mat dL = dY;
        for (auto& layer : layers | std::views::reverse)
            dL = layer.backward(dL);
    }

    void step(float lr, float l2 = 0.0f)
    {
        for (DenseLayer &layer : layers)
            layer.step(lr, l2);
    }

    const DenseLayer &layer(size_t i) const { return layers[i]; }
    size_t layer_count() const { return layers.size(); }
};
