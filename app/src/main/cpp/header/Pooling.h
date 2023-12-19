//
// Created by David on 11/26/2023.
//

#ifndef ARMV9_ARCHITECTURE_DEEP_LEARNING_POOLING_H
#define ARMV9_ARCHITECTURE_DEEP_LEARNING_POOLING_H

#include <vector>

namespace POOL {

    class Pooling {
    public:
        Pooling(int pool_size);

        std::vector<std::vector<std::vector<float>>>  forward(const std::vector<std::vector<std::vector<float>>>& input);
        std::vector<std::vector<std::vector<float>>>  forward_tiled(const std::vector<std::vector<std::vector<float>>>& input, size_t tile_size);

    private:
        int pool_size;
        size_t num_rows;
        size_t num_cols;
    };

} // POOL

#endif //ARMV9_ARCHITECTURE_DEEP_LEARNING_POOLING_H
