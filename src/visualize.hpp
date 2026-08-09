#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include "layer.hpp"

// Save each row of W as a grayscale PGM image [rows x cols].
// Row c of W holds the 784 weights that multiply the image to produce the
// score for class c — reshaped back to 28x28 it shows what the class
// "looks for": white = pixels that push the score up, black = pixels that
// push it down.
inline void save_weight_images(const Dense &layer, int rows, int cols, const std::string &dir)
{
    std::filesystem::create_directories(dir);
    for (int c = 0; c < layer.W.rows(); ++c)
    {
        Vec w = layer.W.row(c).transpose();
        float lo = w.minCoeff();
        float hi = w.maxCoeff();

        std::ofstream out(dir + "/class_" + std::to_string(c) + ".pgm", std::ios::binary);
        out << "P5\n" << cols << " " << rows << "\n255\n";
        for (int p = 0; p < rows * cols; ++p)
            out.put(char(255.0f * (w[p] - lo) / (hi - lo)));
    }
}
