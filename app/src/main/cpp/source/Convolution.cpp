//
// Created by David on 1/8/2024.
//
//
// Convolution.cpp
//
#include "../header/Convolution.h"
#include <algorithm> // for std::max
#include <numeric>   // for std::inner_product

namespace CNN {

    Convolution::Convolution(size_t in_channels, size_t out_channels, size_t kernel_height, size_t kernel_width,
                             size_t input_height, size_t input_width, size_t stride, size_t padding)
            : in_channels(in_channels), out_channels(out_channels), kernel_height(kernel_height), kernel_width(kernel_width),
              input_height(input_height), input_width(input_width), stride(stride), padding(padding) {
        // Initialize weights and biases here
        // Example: weights.resize(in_channels * out_channels * kernel_height * kernel_width);
        //          bias.resize(out_channels);
    }

    std::vector<float> Convolution::forward(const std::vector<float> &input) {
        size_t outputHeight = (input_height - kernel_height + 2 * padding) / stride + 1;
        size_t outputWidth = (input_width - kernel_width + 2 * padding) / stride + 1;
        std::vector<float> output(out_channels * outputHeight * outputWidth, 0);

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < outputHeight; ++oh) {
                for (size_t ow = 0; ow < outputWidth; ++ow) {
                    float sum = 0;
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_height; ++kh) {
                            for (size_t kw = 0; kw < kernel_width; ++kw) {
                                // Compute the position in the input considering stride and padding
                                int h_index = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                                int w_index = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                                // Check for valid index (considering padding)
                                if (h_index >= 0 && h_index < input_height && w_index >= 0 && w_index < input_width) {
                                    size_t inputIdx = (ic * input_height * input_width) + (h_index * input_width) + w_index;
                                    size_t weightIdx = ((oc * in_channels + ic) * kernel_height * kernel_width) + (kh * kernel_width) + kw;
                                    sum += input[inputIdx] * weights[weightIdx];
                                }
                            }
                        }
                    }
                    sum += bias[oc]; // Add bias for the output channel
                    output[oc * outputHeight * outputWidth + oh * outputWidth + ow] = std::max(sum, 0.0f);
                }
            }
        }

        return output;
    }


    void Convolution::setWeights(const std::vector<float> &weights) {
        this->weights = weights;
    }

    void Convolution::setBias(const std::vector<float> &bias) {
        this->bias = bias;
    }
} // CNN
