//
// Created by David on 1/8/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CONVOLUTION_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CONVOLUTION_H
#include <vector>
#include "Layer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"


namespace CNN {

    class Convolution : public Layer {
    public:
        Convolution(size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width,
                    size_t input_height, size_t input_width, size_t stride, size_t padding);
        Convolution(size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width,
                    size_t input_height, size_t input_width, size_t stride, size_t padding, size_t groups,
                    std::unique_ptr<arm_compute::Tensor> weights_tensor,
                    std::unique_ptr<arm_compute::Tensor> bias_tensor);
        Convolution(size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width,
                    size_t input_height, size_t input_width, size_t stride, size_t padding, size_t groups,
                    std::shared_ptr<arm_compute::Tensor> input_tensor,
                    std::unique_ptr<arm_compute::Tensor> weights_tensor,
                    std::unique_ptr<arm_compute::Tensor> bias_tensor,
                    std::shared_ptr<arm_compute::Tensor> output_tensor);
        std::vector<float> forward(const std::vector<float> &input) override;
        void setWeights(const std::vector<float> &weights) override;
        void setBias(const std::vector<float> &bias) override;
        std::string getName() override {return "convolution";}
        void forward_acl() override;
        void configure_acl() override;
        std::unique_ptr<arm_compute::Tensor> weights_tensor, bias_tensor;

    private:
        size_t in_channels, out_channels, kernel_height, kernel_width, input_height, input_width, stride, padding, groups;
        std::vector<float> weights; // Flattened weights for the convolution kernels
        std::vector<float> bias;    // Bias for each output channel
        arm_compute::NEConvolutionLayer conv_layer;
    };

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CONVOLUTION_H
