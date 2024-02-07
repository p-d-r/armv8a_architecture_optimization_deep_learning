//
// Created by David on 1/5/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYER_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYER_H

#include <vector>
#include <string>

class Layer {
public:
    virtual std::vector<float> forward(const std::vector<float> &input) = 0;
    virtual void forward_acl() = 0;
    virtual void setWeights(const std::vector<float> &weights) = 0;
    virtual void setBias(const std::vector<float> &bias) = 0;
    virtual std::string getName() = 0;
    virtual ~Layer() {}
};
#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYER_H
