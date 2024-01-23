//
// Created by David on 1/10/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_SOFTMAX_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_SOFTMAX_H

#include "Layer.h"
#include <cmath>

namespace CNN {

    class Softmax : public Layer {
        std::vector<float> forward(const std::vector<float> &input) override;
        void setWeights(const std::vector<float> &weights) override {};
        void setBias(const std::vector<float> &bias) override {};
        std::string getName() override {return "softmax";}
    };

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_SOFTMAX_H
