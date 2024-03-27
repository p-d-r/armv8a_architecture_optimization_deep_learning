//
// Created by David on 1/9/2024.
//

#include "../header/Network.h"

namespace CNN {

    // Method to perform naive forward pass through all layers
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


    void Network::forward_acl_index(size_t index) {
        if (index < layers.size())
            layers[index]->forward_acl();
    }


    void Network::addLayer(Layer *layer) {
        if (layer == nullptr)
            return;

        layers.emplace_back(layer);
    }

    /*Each added layer has to be wired to its successor / predecessor layer, since they share the same
     * input / output tensors! */
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

    void Network::add_thread(size_t start, size_t end, int affinity) {
        // Create the new thread
        auto new_thread = std::make_unique<LayerThread>(this, start, end, affinity);

        if (!layer_threads.empty()) {
            // Get the last thread in the list
            auto& lastThread = layer_threads.back();

            // Set the new thread as the successor of the last thread
            lastThread->set_successor_thread(new_thread.get());

            // Set the last thread as the predecessor of the new thread
            new_thread->set_predecessor_thread(lastThread.get());
        }

        // Add the new thread to the list
        layer_threads.emplace_back(std::move(new_thread));
    }

    void Network::signal_input_ready() {
        auto vec = generateRandomTensor(224*224*3);
        // Populate the tensor with image data
        std::copy(vec.begin(), vec.end(),
                  reinterpret_cast<float *>(this->input_tensor.get()->buffer()));
        start_timestamps[start_timestamp_index] = (std::chrono::high_resolution_clock::now());
        start_timestamp_index++;
        layer_threads[0].get()->input_ready_signal();
    }


    void Network::signal_output_ready() {
        auto now = std::chrono::high_resolution_clock::now();
        stop_timestamps[stop_timestamp_index] = now;
        stop_timestamp_index++;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - end_ts).count();
        end_ts = now;
        if (measurement_no < 1000) {
            inf_per_sec[measurement_no] = (1.0 / ((double) (duration / 1000000.0)));
            measurement_no++;
        }
        /*calculate and log the measured data to logcat (since the data cannot be written directly
          to the development machine and i wanted to avoid unnecessary file-transfer workloads*/
        if (measurement_no == 1000) {
            std::stringstream sis;
            for(size_t i = 0; i < inf_per_sec.size(); ++i) {
                sis << inf_per_sec[i];
                // Add a comma after each element except the last one
                if (i < inf_per_sec.size() - 1) {
                    sis << ", ";
                }
                // Insert a line break every 100 elements or at the end of the vector
                if ((i + 1) % 100 == 0 || i == inf_per_sec.size() - 1) {
                    LOGI_NETWORK("%s", sis.str().c_str()); // Use your logging function
                    sis.str(""); // Clear the stringstream for the next chunk of data
                    sis.clear(); // Clear any error flags
                }
            }
            // Sort times for median calculation
            std::sort(inf_per_sec.begin(), inf_per_sec.end());
            double median_time{};
            // Even number of elements: average of the two middle elements
            median_time = (inf_per_sec[1000 / 2 - 1] + inf_per_sec[1000 / 2]) / 2;
            LOGI_NETWORK("median inf_per_sec: %f us;", median_time);
        }
        this->output_buffer_ready = true;
        cv.notify_one();
        /*TODO this is only for the synthetic test cases, in real-use scenarios, the output buffer has
          to be read/copied/interpreted before the output buffer ready signal should be sent*/
        this->layer_threads.back()->output_ready_signal();
    }

    std::vector<size_t> Network::wait_for_output() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return output_buffer_ready&&1;});
        auto ret =  find_top_five_indices(this->output_tensor.get());
        this->layer_threads.back()->output_ready_signal();
        return ret;
    }

    int Network::get_layer_count() {
        return this->layers.size();
    }

    void Network::start_threads() {
        for (int i = 0; i < layer_threads.size(); i++) {
            layer_threads[i].get()->start_thread();
        }
    }

    void Network::configure_layer_thread(int start, int end) {
        for (int i = start; i < end; i++) {
            layers[i].get()->configure_acl();
        }
    }
} // NETWORK