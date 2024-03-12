//
// Created by David on 1/5/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYER_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYER_H

#include <vector>
#include <string>
#include <android/log.h>
#include "arm_compute/runtime/Tensor.h"
#define LOG_TAG_LAYER "NativeCode:Layer" // Tag for logging
#define LOGI_LAYER(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_LAYER, __VA_ARGS__)

namespace CNN {
    class Layer {
    public:
        arm_compute::TensorShape input_tensor_shape, output_tensor_shape;
        std::shared_ptr<arm_compute::Tensor> input_tensor, output_tensor;

        virtual std::vector<float> forward(const std::vector<float> &input) = 0;
        virtual void forward_acl() = 0;
        virtual void setWeights(const std::vector<float> &weights) = 0;
        virtual void setBias(const std::vector<float> &bias) = 0;
        virtual std::string getName() = 0;

        virtual ~Layer() {
            //Make sure the input and output tensors are deallocated in case a layer is destroyed
            //(making the network invalid anyways)
            if (this->input_tensor)
                this->input_tensor->allocator()->free();
            if (this->output_tensor)
                this->output_tensor->allocator()->free();
        }
    };
}
#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYER_H
