//
// Created by David on 1/9/2024.
//

#include "../header/Network.h"

namespace CNN {

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
            //LOGI_NETWORK("Current Layer:%s", layer->getName().c_str());
            layer->forward_acl();
        }
    }

    void Network::addLayer(Layer *layer) {
        if (layer == nullptr)
            return;

        layers.emplace_back(layer);
    }

    void Network::addLayer(Layer *layer, arm_compute::TensorShape in_shape,
                           arm_compute::TensorShape out_shape) {
        if (layer == nullptr)
            return;

        if (this->layers.size() == 0) {
            auto input = std::make_shared<arm_compute::Tensor>();
            input->allocator()->init(arm_compute::TensorInfo(in_shape, 1, arm_compute::DataType::F32, this->data_layout));
            input->allocator()->allocate();
            layer->input_tensor = input;
            this->input_tensor = layer->input_tensor;
            auto output = std::make_shared<arm_compute::Tensor>();
            output->allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32, this->data_layout));
            output->allocator()->allocate();
            layer->output_tensor = output;
        } else {
            layer->input_tensor = layers.back()->output_tensor;
            auto output = std::make_shared<arm_compute::Tensor>();
            output->allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32, this->data_layout));
            output->allocator()->allocate();
            layer->output_tensor = output;
        }

        layers.emplace_back(layer);
        this->output_tensor = layer->output_tensor;

    }
} // NETWORK