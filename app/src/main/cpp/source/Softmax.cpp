//
// Created by David on 1/10/2024.
//

#include "../header/Softmax.h"

namespace CNN {
    std::vector<float> Softmax::forward(const std::vector<float> &input) {
        // Assuming that the input is a 1D vector of logits
        std::vector<float> output(input.size());
        float maxLogit = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;

        // Compute the sum of exp(logits - maxLogit) for numerical stability
        for (float logit : input) {
            sum += std::exp(logit - maxLogit);
        }

        // Compute the softmax
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - maxLogit) / sum;
        }

        return output;
    }
} // CNN