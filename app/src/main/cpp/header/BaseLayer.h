//
// Created by David on 1/5/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_BASELAYER_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_BASELAYER_H

class BaseLayer {
public:
    virtual void forward() = 0;
    virtual std::vector<float> setWeights(std::vector<float> &weights) = 0;
    virtual ~BaseLayer() {}
private:
    std::vector<float> weights;
};
#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_BASELAYER_H
