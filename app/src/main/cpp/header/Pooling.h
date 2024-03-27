//
// Created by David on 11/26/2023.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H

#include <vector>
#include "Layer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/Tensor.h"


namespace CNN {

class Pooling : public CNN::Layer {
    public:
        /** Constructor without tensor parameters
         * @param[in] pool_height      pooling kernel height
         * @param[in] pool_width       pooling kernel width
         * @param[in] channels         number of channels in the input tensor
         * @param[in] input_height     height of input tensor
         * @param[in] input_width      width of input tensor
         * @param[in] stride           stride (only synchronous stride with stride_x=stride_y is supported)
         * @param[in] top_padding      right padding
         * @param[in] left_padding     left padding
         * @param[in] bottom_padding   bottom padding
         * @param[in] right_padding    right padding
         */
        Pooling(size_t pool_height, size_t pool_width, size_t channels,
                size_t input_height, size_t input_width, size_t stride,
                size_t top_padding, size_t left_padding, size_t bottom_padding, size_t right_padding);
        std::vector<float> forward(const std::vector<float> &input) override;
        std::string getName() override {return "Pooling";}
        void forward_acl() override;
        void configure_acl() override;

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

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_POOLING_H
