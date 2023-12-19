//
// Created by David on 12/5/2023.
//

#ifndef ARMV9_ARCHITECTURE_DEEP_LEARNING_BASELAYER_H
#define ARMV9_ARCHITECTURE_DEEP_LEARNING_BASELAYER_H

#include <vector>

namespace FC {
    class BaseLayer {
    public:
        virtual std::vector<float> forward(std::vector<float> input) = 0;
        virtual ~BaseLayer() {};
    private:

    };

}
#endif //ARMV9_ARCHITECTURE_DEEP_LEARNING_BASELAYER_H
