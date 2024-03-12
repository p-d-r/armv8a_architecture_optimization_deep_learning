//
// Created by David Pauli (ti72teta) on 26.11.2023
//
#include "../header/Pooling.h"

namespace CNN {
    Pooling::Pooling(size_t pool_height, size_t pool_width, size_t channels,
                     size_t input_height, size_t input_width, size_t stride,
                     size_t top_padding, size_t left_padding, size_t bottom_padding, size_t right_padding)
            : pool_height(pool_height), pool_width(pool_width), channels(channels),
              input_height(input_height), input_width(input_width), stride(stride),
              top_padding(top_padding), left_padding(left_padding),
              bottom_padding(bottom_padding), right_padding(right_padding) {}


    std::vector<float> Pooling::forward(const std::vector<float> &input) {
        // Adjusted output height and width calculations to account for bottom and right padding
        size_t outputHeight = (input_height - pool_height + top_padding + bottom_padding) / stride + 1;
        size_t outputWidth = (input_width - pool_width + left_padding + right_padding) / stride + 1;
        std::vector<float> output(channels * outputHeight * outputWidth, 0);

        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < outputHeight; ++h) {
                for (size_t w = 0; w < outputWidth; ++w) {
                    float maxVal = -std::numeric_limits<float>::infinity();
                    for (size_t ph = 0; ph < pool_height; ++ph) {
                        for (size_t pw = 0; pw < pool_width; ++pw) {
                            // Calculate the input indices considering the stride and padding
                            int h_index = h * stride + ph - top_padding;
                            int w_index = w * stride + pw - left_padding;

                            // Check if the index is within the bounds of the input dimensions
                            if (h_index >= 0 && h_index < input_height && w_index >= 0 && w_index < input_width) {
                                size_t currentIdx = c * input_height * input_width + h_index * input_width + w_index;
                                maxVal = std::max(maxVal, input[currentIdx]);
                            }
                        }
                    }
                    output[c * outputHeight * outputWidth + h * outputWidth + w] = maxVal;
                }
            }
        }

        return output;
    }


    void Pooling::forward_acl() {
        pool_layer.run();
    }

    void Pooling::configure_acl() {
        // Define the pooling layer parameters
        arm_compute::PoolingLayerInfo pool_info(arm_compute::PoolingType::MAX,
                                                arm_compute::Size2D(pool_width, pool_height),
                                                arm_compute::DataLayout::NCHW,
                                                arm_compute::PadStrideInfo(stride,
                                                                           stride,
                                                                           left_padding,
                                                                           right_padding,
                                                                           top_padding,
                                                                           bottom_padding,
                                                                           arm_compute::DimensionRoundingType::CEIL));
        // validate input & output correspondence
        auto valid = pool_layer.validate(input_tensor.get()->info(), output_tensor.get()->info(), pool_info);
        LOGI_LAYER("%s", valid.error_description().c_str());
        // Configure the pooling layer
        pool_layer.configure(input_tensor.get(), output_tensor.get(), pool_info);
    }
} // POOL