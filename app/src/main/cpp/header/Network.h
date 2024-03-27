//
// Created by David on 1/9/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_NETWORK_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_NETWORK_H

#include <vector>
#include <thread>
#include <android/log.h>
#include <string>
#include <sstream>
#include "Layer.h"
#include "LayerThread.h"
#include "Helpers.h"
#include "arm_compute/runtime/Tensor.h"

#define LOG_TAG_NETWORK "NativeCode:native-lib" // Tag for logging
#define LOGI_NETWORK(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_NETWORK, __VA_ARGS__)

namespace CNN {

    class Network {
    public:
        Network(arm_compute::DataLayout data_layout):data_layout(data_layout), durations(1000), inf_per_sec(1000){};
        // Add a generic layer to the network
        void addLayer(Layer *layer);
        // Add generic layer to network and set up input and output tensor buffers, wire them
        void addLayer(Layer *layer, arm_compute::TensorShape in_shape,
                      arm_compute::TensorShape out_shape);
        void add_thread(size_t start, size_t end, int affinity);
        std::vector<float> forward(const std::vector<float> &input);
        void forward_acl();
        void forward_acl_index(size_t index);
        void signal_input_ready();
        void signal_output_ready();
        int get_layer_count();
        void start_threads();
        void configure_layer_thread(int start, int end);
        std::vector<size_t> wait_for_output();
        ~Network() {}

        std::shared_ptr<arm_compute::Tensor> input_tensor;
        std::shared_ptr<arm_compute::Tensor> output_tensor;
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        std::vector<std::unique_ptr<CNN::LayerThread>> layer_threads;
        std::vector<std::chrono::high_resolution_clock::time_point> start_timestamps;
        std::vector<std::chrono::high_resolution_clock::time_point> stop_timestamps;
        int start_timestamp_index, stop_timestamp_index = 0;
        std::vector<long long> durations;
        std::vector<double> inf_per_sec;
        int measurement_no = 0;
        arm_compute::DataLayout data_layout;
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<bool> input_buffer_ready = false, stop_requested = false, output_buffer_ready=false;
        std::chrono::high_resolution_clock::time_point start_ts, end_ts;
    };

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_NETWORK_H
