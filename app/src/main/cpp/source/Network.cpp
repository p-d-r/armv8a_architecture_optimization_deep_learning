//
// Created by David on 1/9/2024.
//

#include "../header/Network.h"

namespace NETWORK {

    // Method to perform forward pass through all layers
    std::vector<float> Network::forward(const std::vector<float>& input) {
        std::vector<float> output = input;
        LOGI_NETWORK("START INPUT-SIZE: %zu", output.size());
        for (auto& layer : layers) {
            LOGI_NETWORK("Current Layer:%s", layer->getName().c_str());
            output = layer->forward(output);
            LOGI_NETWORK("NEXT INPUT-SIZE: %zu", output.size());
        }

        return output;
    }

    void Network::forward_acl() {
        for (auto& layer: layers){
            LOGI_NETWORK("Current Layer:%s", layer->getName().c_str());
            layer->forward_acl();
        }
    }

    void Network::addLayer(Layer *layer) {
        layers.emplace_back(layer);
    }
} // NETWORK