//
// Created by David on 11/26/2023.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H

#include <vector>
#include "Layer.h"

namespace POOL {

    class Pooling : public Layer {
    public:
        Pooling(size_t pool_height, size_t pool_width, size_t channels,
                size_t input_height, size_t input_width, size_t stride,
                size_t top_padding, size_t left_padding, size_t bottom_padding, size_t right_padding);
        std::vector<float> forward(const std::vector<float> &input) override;
        std::string getName() override {return "Pooling";}
    private:
        size_t pool_height;
        size_t pool_width;
        size_t channels;
        size_t input_height;
        size_t input_width;
        size_t stride;
        size_t top_padding, bottom_padding, left_padding, right_padding;

        void setWeights(const std::vector<float> &weights) override;
        void setBias(const std::vector<float> &bias) override;
    };

} // POOL

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H
