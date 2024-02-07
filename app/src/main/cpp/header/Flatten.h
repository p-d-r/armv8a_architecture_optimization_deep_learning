//
// Created by drp on 06.02.24.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FLATTEN_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FLATTEN_H

#include "Layer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/runtime/TensorAllocator.h"

namespace CNN {

    class Flatten : public Layer {
    public:
        Flatten(std::shared_ptr<arm_compute::Tensor> input_tensor, std::shared_ptr<arm_compute::Tensor> output_tensor);
        std::vector<float> forward(const std::vector<float> &input) override {return std::vector<float>();};
        void forward_acl() override;
        std::string getName() override {return "FLATTEN";};
        std::shared_ptr<arm_compute::Tensor> input_tensor, output_tensor;

    private:
        void setWeights(const std::vector<float> &weights) override {};
        void setBias(const std::vector<float> &bias) override {};
    };

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FLATTEN_H
