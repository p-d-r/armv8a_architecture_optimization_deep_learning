//
// Created by David on 11/26/2023.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FULLYCONNECTED_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FULLYCONNECTED_H

#include <vector>
#include "BaseLayer.h"

namespace FC {

    class FullyConnected: public BaseLayer {
    public:
        FullyConnected(const std::vector<float> & weights, size_t input_size, size_t output_size, int tile_size);
        std::vector<float> setWeights(std::vector<float> &weights);
        std::vector<float> forward(const std::vector<float> & batch_input);
        std::vector<float> forward_transposed_in(const std::vector<float> & batch_input_transposed);
        std::vector<float> forward_tiled(const std::vector<float> & batch_input);
        std::vector<float> forward_tiled_transposed_in(const std::vector<float> & batch_input_transposed);

    private:
        std::vector<float> weights;
        size_t input_size;
        size_t output_size;
        int tile_size;
    };

} // FC

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_FULLYCONNECTED_H
