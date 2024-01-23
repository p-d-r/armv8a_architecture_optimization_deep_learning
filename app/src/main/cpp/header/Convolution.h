//
// Created by David on 1/8/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CONVOLUTION_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CONVOLUTION_H
#include <vector>
#include "Layer.h"
#include "arm/arm_compute/runtime/Tensor.h"

namespace CNN {

    class Convolution : public Layer {
    public:
        Convolution(size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width,
                    size_t input_height, size_t input_width, size_t stride, size_t padding);
        std::vector<float> forward(const std::vector<float> &input) override;
        void setWeights(const std::vector<float> &weights) override;
        void setBias(const std::vector<float> &bias) override;
        std::string getName() override {return "convolution";}



    private:
        size_t in_channels;
        size_t out_channels;
        size_t kernel_height;
        size_t kernel_width;
        size_t input_height;
        size_t input_width;
        size_t stride;
        size_t padding;
        std::vector<float> weights; // Flattened weights for the convolution kernels
        std::vector<float> bias;    // Bias for each output channel
        arm_compute::Tensor input_tensor, weights_tensor, output_tensor;
    };

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CONVOLUTION_H
