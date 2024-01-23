//
// Created by David on 1/9/2024.
//

#include "../header/LRN.h"

namespace LRN {
    LRN::LRN(size_t depth, size_t height, size_t width, size_t local_size, double alpha, float beta, float k) :
    depth(depth), height(height), width(width), local_size(local_size), alpha(alpha), beta(beta), k(k) {}

    std::vector<float> LRN::forward(const std::vector<float> &input) {
        std::vector<float> output(input.size(), 0.0f);
        int half_local_size = static_cast<int>(local_size / 2);

        for (size_t d = 0; d < depth; ++d) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t idx = (d * height + h) * width + w;
                    float sum = 0.0f;

                    // Sum over the local window
                    for (int j = -half_local_size; j <= half_local_size; ++j) {
                        if (static_cast<int>(d) + j >= 0 && static_cast<int>(d) + j < static_cast<int>(depth)) {
                            size_t local_idx = ((d + j) * height + h) * width + w;
                            sum += std::pow(input[local_idx], 2);
                        }
                    }

                    // Apply normalization formula outside the j loop
                    output[idx] = input[idx] * std::pow(k + alpha * sum, -beta);
                }
            }
        }

        return output;
    }

} // LRN