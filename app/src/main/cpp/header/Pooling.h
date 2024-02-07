//
// Created by David on 11/26/2023.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H

#include <vector>
#include "Layer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/Tensor.h"


namespace POOL {

    class Pooling : public Layer {
    public:
        Pooling(size_t pool_height, size_t pool_width, size_t channels,
                size_t input_height, size_t input_width, size_t stride,
                size_t top_padding, size_t left_padding, size_t bottom_padding, size_t right_padding);
        Pooling(size_t pool_height, size_t pool_width, size_t channels,
                size_t input_height, size_t input_width, size_t stride,
                size_t top_padding, size_t left_padding, size_t bottom_padding, size_t right_padding,
                std::shared_ptr<arm_compute::Tensor> input_tensor, std::shared_ptr<arm_compute::Tensor> output_tensor);
        std::vector<float> forward(const std::vector<float> &input) override;
        std::string getName() override {return "Pooling";}
        void forward_acl() override;
        std::shared_ptr<arm_compute::Tensor> input_tensor, output_tensor;

    private:
        size_t pool_height;
        size_t pool_width;
        size_t channels;
        size_t input_height;
        size_t input_width;
        size_t stride;
        size_t top_padding, bottom_padding, left_padding, right_padding;
        arm_compute::NEPoolingLayer pool_layer;

        void setWeights(const std::vector<float> &weights) override {};
        void setBias(const std::vector<float> &bias) override {};
    };

} // POOL

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H
