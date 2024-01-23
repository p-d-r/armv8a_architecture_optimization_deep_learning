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

namespace FC {

    class FullyConnected : public Layer {
    public:
        FullyConnected(const std::vector<float> & weights, const std::vector<float> & bias,
                       size_t input_size, size_t output_size, int tile_size);
        void setWeights(const std::vector<float> &weights) override;
        void setBias(const std::vector<float> &bias) override;
        std::vector<float> forward(const std::vector<float> &batch_input) override;
        std::vector<float> forward_transposed_in(const std::vector<float> & batch_input_transposed);
        std::vector<float> forward_tiled(const std::vector<float> & batch_input);
        std::vector<float> forward_tiled_transposed_in(const std::vector<float> & batch_input_transposed);
        std::string getName() override {return "fully connected";}

    private:
        std::vector<float> weights;
        std::vector<float> bias;
        size_t input_size;
        size_t output_size;
        int tile_size;
    };

} // FC

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FULLYCONNECTED_H
