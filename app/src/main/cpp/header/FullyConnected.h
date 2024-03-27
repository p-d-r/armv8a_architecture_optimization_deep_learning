//
// Created by David on 11/26/2023.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FULLYCONNECTED_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FULLYCONNECTED_H

#include <vector>
#include "Layer.h"
#include <cmath>
#include <algorithm> // for std::max
#include <numeric>   // for std::inner_product
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/function_info/FullyConnectedLayerInfo.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/Types.h"

namespace CNN {

class FullyConnected : public CNN::Layer {
    public:
        FullyConnected(const std::vector<float> & weights, const std::vector<float> & bias,
                       size_t input_size, size_t output_size, int tile_size);

        FullyConnected(std::unique_ptr<arm_compute::Tensor> weights_tensor,
                       std::unique_ptr<arm_compute::Tensor> bias_tensor,
                       size_t input_size,
                       size_t output_size,
                       int tile_size,
                       arm_compute::ActivationLayerInfo::ActivationFunction activation_function
                       = arm_compute::ActivationLayerInfo::ActivationFunction::RELU);


        void setWeights(const std::vector<float> &weights) override;
        void setBias(const std::vector<float> &bias) override;
        std::vector<float> forward(const std::vector<float> &batch_input) override;
        std::vector<float> forward_transposed_in(const std::vector<float> &batch_input_transposed);
        std::vector<float> forward_tiled(const std::vector<float> &batch_input);
        std::vector<float> forward_tiled_transposed_in(const std::vector<float> &batch_input_transposed);
        std::string getName() override {return "fully connected";}

        //ARM Compute Library functions
        void forward_acl() override;
        void configure_acl() override;
        std::unique_ptr<arm_compute::Tensor> weights_tensor, bias_tensor;

    private:
        std::vector<float> weights;
        std::vector<float> bias;
        size_t input_size;
        size_t output_size;
        int tile_size;
        arm_compute::ActivationLayerInfo::ActivationFunction activation_function;
        arm_compute::NEFullyConnectedLayer fc_layer;
    };

} //

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FULLYCONNECTED_H
