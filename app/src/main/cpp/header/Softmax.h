//
// Created by David on 1/10/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_SOFTMAX_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_SOFTMAX_H

#include "Layer.h"
#include <cmath>
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/Tensor.h"

namespace CNN {

    class Softmax : public Layer {
    public:
        Softmax(std::shared_ptr<arm_compute::Tensor> input_tensor=nullptr, std::shared_ptr<arm_compute::Tensor> output_tensor=nullptr);
        std::vector<float> forward(const std::vector<float> &input) override;
        void setWeights(const std::vector<float> &weights) override {};
        void setBias(const std::vector<float> &bias) override {};
        std::string getName() override {return "softmax";}
        void forward_acl() override;

        std::shared_ptr<arm_compute::Tensor>  input_tensor, output_tensor;

    private:
        arm_compute::NESoftmaxLayer softmax_layer;
    };

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_SOFTMAX_H
