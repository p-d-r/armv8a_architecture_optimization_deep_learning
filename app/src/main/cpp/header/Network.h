//
// Created by David on 1/9/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_NETWORK_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_NETWORK_H

#include <vector>
#include <android/log.h>
#include "Layer.h"
#include "arm_compute/runtime/Tensor.h"

#define LOG_TAG_NETWORK "NativeCode:native-lib" // Tag for logging
#define LOGI_NETWORK(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_NETWORK, __VA_ARGS__)

namespace CNN {

    class Network {
    public:
        Network(arm_compute::DataLayout data_layout):data_layout(data_layout) {};
        // Add a generic layer to the network
        void addLayer(Layer *layer);
        // Add generic layer to network and set up input and output tensor buffers, wire them
        void addLayer(Layer *layer, arm_compute::TensorShape in_shape,
                      arm_compute::TensorShape out_shape);
        std::vector<float> forward(const std::vector<float> &input);
        void forward_acl();
        ~Network() {}

        std::shared_ptr<arm_compute::Tensor> input_tensor;
        std::shared_ptr<arm_compute::Tensor> output_tensor;
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        arm_compute::DataLayout data_layout;
    };

} // NETWORK

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_NETWORK_H
