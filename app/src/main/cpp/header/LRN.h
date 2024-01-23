//
// Created by David on 1/9/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LRN_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LRN_H
#include "Layer.h"
#include <cmath>

namespace LRN {

    class LRN : public Layer {
    public:
        LRN(size_t depth=96, size_t height=54,
            size_t width=54, size_t local_size=5,
            double alpha=0.00009999999747378752,
            float beta=0.75, float k=1.0f);
        std::vector<float> forward(const std::vector<float> &input) override;
        void setWeights(const std::vector<float> &weights) override {};
        void setBias(const std::vector<float> &bias) override {};
        std::string getName() override {return "LRN";}
    private:
        size_t depth, height, width, local_size;
        double alpha;
        float beta, k;
    };

} // LRN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LRN_H
