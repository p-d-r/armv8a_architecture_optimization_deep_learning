//
// Created by David Pauli (ti72teta) on 26.11.2023
//
#include <jni.h>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
// custom implementations
#include "header/FullyConnected.h"
#include "header/Pooling.h"
#include "header/Convolution.h"
#include "header/Network.h"
#include "header/Testing.h"
#include "header/Helpers.h"
#include "header/LRN.h"
#include "header/Softmax.h"
#include "header/CustomAccessors.h"
// onnxruntime
#include "onnxruntime/onnxruntime_cxx_api.h"
// arm compute library
#include "arm_compute/graph.h"
#include "arm_compute/runtime/Scheduler.h"
#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/Utils.h"
#endif /* ARM_COMPUTE_CL */
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include "header/Flatten.h"

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

//android device logging -> debug info mode
#define LOG_TAG_lib "NativeCode:native-lib" // Tag for logging
#define LOGI_lib(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_lib, __VA_ARGS__)


// Define a function pointer type for the forward methods
typedef std::vector<float> (FC::FullyConnected::*ForwardMethod)(const std::vector<float>&);

AAssetManager* assetManager = nullptr;
// Declare a global network object
NETWORK::Network alexnet;
Stream alexnet_graph(0, "AlexNet");
ArrayAccessor* raw_camera_input_accessor = nullptr;
std::unique_ptr<arm_compute::graph::ITensorAccessor> camera_input_accessor;



std::vector<float> profile_fc(ForwardMethod method, size_t inputSize, size_t outputSize, int tile_size, size_t iterations) {
    std::vector<std::chrono::duration<double, std::micro>> times(iterations);
    std::vector<float> output;
    for (int i = 0; i < iterations; i++) {
        // Generate random weight-bias matrix with an additional row for bias
        std::vector<float> weight_matrix = generateRandomTensor(inputSize * outputSize);
        std::vector<float> bias_vector = generateRandomTensor(outputSize);
        // Generate a random input vector
        std::vector<float> input = generateRandomTensor(inputSize * 40);
        // Construct the layer
        FC::FullyConnected layer(weight_matrix, bias_vector, inputSize, outputSize, tile_size);

        // Time the execution of the forward method
        auto start = std::chrono::high_resolution_clock::now();
        output = (layer.*method)(input);
        auto end = std::chrono::high_resolution_clock::now();

        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }

    // Find minimum and maximum times
    auto min_time = *std::min_element(times.begin(), times.end());
    auto max_time = *std::max_element(times.begin(), times.end());

    // Calculate average time
    std::chrono::duration<double, std::micro> total_time(0);
    for (const auto& time : times) {
        total_time += time;
    }
    auto average_time = total_time / iterations;

    // Sort times for median calculation
    std::sort(times.begin(), times.end());
    std::chrono::duration<double, std::micro> median_time{};
    if (iterations % 2 == 0) {
        // Even number of elements: average of the two middle elements
        median_time = (times[iterations / 2 - 1] + times[iterations / 2]) / 2;
    } else {
        // Odd number of elements: middle element
        median_time = times[iterations / 2];
    }

    LOGI_lib("tile_size: %d", tile_size);
    LOGI_lib("forward:  min: %f us;   max: %f us;   median: %f us;   average: %f us", min_time.count(), max_time.count(), median_time.count(), average_time.count());
    return output;
}


int profile() {
    LOGI_lib("start process!");
    //benchmark fully connected layer
    const size_t inputSize = 1024;   // Adjust size as needed
    const size_t outputSize = 1024;   // Adjust size as needed
    const size_t iterations = 100;

    bool identical = true;
    auto out_seq0 = profile_fc(&FC::FullyConnected::forward, inputSize, outputSize, 0, iterations);
    auto out_seq1 = profile_fc(&FC::FullyConnected::forward_transposed_in, inputSize, outputSize, 0, iterations);
    auto out_seq2 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 4, iterations);
    auto out_seq2_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 4, iterations);
    auto out_seq3 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 8, iterations);
    auto out_seq3_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 8, iterations);
    auto out_seq4 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 16, iterations);
    auto out_seq4_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 16, iterations);
    auto out_seq5 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 32, iterations);
    auto out_seq5_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 32, iterations);
    auto out_seq6 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 64, iterations);
    auto out_seq6_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 64, iterations);
    auto out_seq7 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 128, iterations);
    auto out_seq7_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 128, iterations);
    auto out_seq8 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 256, iterations);
    auto out_seq8_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 256, iterations);
    auto out_seq9 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 512, iterations);
    auto out_seq9_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 512, iterations);
    auto out_seq10 = profile_fc(&FC::FullyConnected::forward_tiled, inputSize, outputSize, 1024, iterations);
    auto out_seq10_t = profile_fc(&FC::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 1024, iterations);
    return 0; // Return type changed from 'void' to 'int'
}


std::string getTensorInfo(const arm_compute::Tensor& tensor) {
    std::ostringstream oss;
    const arm_compute::ITensorInfo* tensor_info = tensor.info();

    if (tensor_info != nullptr) {
        // Get tensor dimensions
        const arm_compute::TensorShape& shape = tensor_info->tensor_shape();

        // Append each dimension to the string stream
        oss << "Tensor dimensions: ";
        for (size_t i = 0; i < shape.num_dimensions(); ++i) {
            oss << shape[i] << (i < shape.num_dimensions() - 1 ? " x " : "");
        }

        // Append total number of elements
        oss << "\nTotal number of elements: " << shape.total_size();

        // Append data type
        oss << "\nData type: ";
        switch (tensor_info->data_type()) {
            case arm_compute::DataType::F32:
                oss << "F32";
                break;
            case arm_compute::DataType::F16:
                oss << "F16";
                break;
            case arm_compute::DataType::QASYMM8:
                oss << "QASYMM8";
                break;
                // ... handle other data types as needed
            default:
                oss << "Unknown";
        }
    } else {
        oss << "Tensor info is null.";
    }

    return oss.str();
}

// Usage Example
// arm_compute::Tensor myTensor;
// ... (initialize and use myTensor)
// std::string tensorDetails = getTensorInfo(myTensor);
// Now you can log tensorDetails or use it as needed



std::vector<float> readWeightVector(const std::string& filename) {
    AAsset* asset = AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_UNKNOWN);
    if (asset == nullptr) {
        std::cerr << "Failed to open asset: " << filename << std::endl;
        return {};
    }

    size_t size = AAsset_getLength(asset);

    // Calculate the number of floats in the asset
    size_t numFloats = size / sizeof(float);

    // Make sure that the size of the data is a multiple of the size of float
    if (size % sizeof(float) != 0) {
        LOGI_lib("Asset size is not a multiple of float size");
        AAsset_close(asset);
        return {};
    }

    std::vector<float> floatData(numFloats);
    AAsset_read(asset, floatData.data(), size);
    AAsset_close(asset);

    return floatData;
}


void generate_model_alexnet() {
    std::vector<float> conv1_kernel = readWeightVector("weights/alexnetconv1_w_0.bin");
    std::vector<float> conv1_bias = readWeightVector("weights/alexnetconv1_b_0.bin");
    LOGI_lib("conv1 dimensions: %zu    %zu", conv1_kernel.size(), conv1_bias.size());
    auto *conv1 = new CNN::Convolution(3, 96, 11, 11, 224, 224, 4, 0);
    conv1->setWeights(conv1_kernel);
    conv1->setBias(conv1_bias);
    alexnet.addLayer(conv1);

    auto *lrn1 = new LRN::LRN(96, 54, 54, 5, 0.00009999999747378752, 0.75f, 1.0f);
    alexnet.addLayer(lrn1);

    auto *pooling1 = new POOL::Pooling(3, 3, 96, 54, 54, 2, 0,0,0,0);
    alexnet.addLayer(pooling1);

    std::vector<float> conv2_kernel = readWeightVector("weights/alexnetconv2_w_0.bin");
    std::vector<float> conv2_bias = readWeightVector("weights/alexnetconv2_b_0.bin");
    LOGI_lib("conv2 dimensions: %zu    %zu", conv2_kernel.size(), conv2_bias.size());
    auto *conv2 = new CNN::Convolution(96, 256, 5, 5, 26, 26, 1, 2);
    conv2->setWeights(conv2_kernel);
    conv2->setBias(conv2_bias);
    alexnet.addLayer(conv2);

    auto *lrn2 = new LRN::LRN(256, 26, 26, 5, 0.00009999999747378752, 0.75f, 1.0f);
    alexnet.addLayer(lrn2);

    auto *pooling2 = new POOL::Pooling(3, 3, 256, 26, 26, 2, 0,0,0,0);
    alexnet.addLayer(pooling2);

    std::vector<float> conv3_kernel = readWeightVector("weights/alexnetconv3_w_0.bin");
    std::vector<float> conv3_bias = readWeightVector("weights/alexnetconv3_b_0.bin");
    LOGI_lib("conv3 dimensions: %zu    %zu", conv3_kernel.size(), conv3_bias.size());
    auto *conv3 = new CNN::Convolution(256, 384, 3, 3, 12, 12, 1, 1);
    conv3->setWeights(conv3_kernel);
    conv3->setBias(conv3_bias);
    alexnet.addLayer(conv3);

    std::vector<float> conv4_kernel = readWeightVector("weights/alexnetconv4_w_0.bin");
    std::vector<float> conv4_bias = readWeightVector("weights/alexnetconv4_b_0.bin");
    LOGI_lib("conv4 dimensions: %zu    %zu", conv4_kernel.size(), conv4_bias.size());
    auto *conv4 = new CNN::Convolution(384, 384, 3, 3, 12, 12, 1, 1);
    conv4->setWeights(conv4_kernel);
    conv4->setBias(conv4_bias);
    alexnet.addLayer(conv4);

    std::vector<float> conv5_kernel = readWeightVector("weights/alexnetconv5_w_0.bin");
    std::vector<float> conv5_bias = readWeightVector("weights/alexnetconv5_b_0.bin");
    LOGI_lib("conv5 dimensions: %zu    %zu", conv5_kernel.size(), conv5_bias.size());
    auto *conv5 = new CNN::Convolution(384, 256, 3, 3, 12, 12, 1, 1);
    conv5->setWeights(conv5_kernel);
    conv5->setBias(conv5_bias);
    alexnet.addLayer(conv5);

    auto *pooling3 = new POOL::Pooling(3, 3, 256, 12, 12, 2, 0,0,1,1);
    alexnet.addLayer(pooling3);

    std::vector<float> fc6_weights = readWeightVector("weights/alexnetfc6_w_0.bin");
    std::vector<float> fc6_bias = readWeightVector("weights/alexnetfc6_b_0.bin");
    LOGI_lib("fc6 dimensions: %zu    %zu", fc6_weights.size(), fc6_bias.size());
    auto *fc6 = new FC::FullyConnected(fc6_weights, fc6_bias, 9216, 4096, 0);
    alexnet.addLayer(fc6);

    std::vector<float> fc7_weights = readWeightVector("weights/alexnetfc7_w_0.bin");
    std::vector<float> fc7_bias = readWeightVector("weights/alexnetfc7_b_0.bin");
    LOGI_lib("fc7 dimensions: %zu    %zu", fc7_weights.size(), fc7_bias.size());
    auto *fc7 = new FC::FullyConnected(fc7_weights, fc7_bias, 4096, 4096, 0);
    alexnet.addLayer(fc7);

    std::vector<float> fc8_weights = readWeightVector("weights/alexnetfc8_w_0.bin");
    std::vector<float> fc8_bias = readWeightVector("weights/alexnetfc8_b_0.bin");
    LOGI_lib("fc8 dimensions: %zu    %zu", fc8_weights.size(), fc8_bias.size());
    auto *fc8 = new FC::FullyConnected(fc8_weights, fc8_bias, 4096, 1000, 0);
    alexnet.addLayer(fc8);

    auto *softmax = new CNN::Softmax();
    alexnet.addLayer(softmax);
}


//Layers constructed with new are implicitly converted to std::unique_ptr<Layer> when added to the network; Network has absolute ownership.
//for shared input/output tensors, shared ptrs are used; these are passed by reference to increase the reference counter
//for layer-own tensors like weights, kernels and biases, unique_ptrs are used; these are passed with std::move to ensure absolute ownership
void generate_model_alexnet_acl() {
    //1. Conv Layer 1; Load kernel weights & biases
    auto conv1_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv1_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv1_kernel, readWeightVector("weights/alexnetconv1_w_0.bin"), arm_compute::TensorShape(11, 11, 3, 96));
    vectorToTensor(*conv1_bias, readWeightVector("weights/alexnetconv1_b_0.bin"), arm_compute::TensorShape(96));
    LOGI_lib("conv1 kernel info %s", getTensorInfo(*conv1_kernel).c_str());
    LOGI_lib("conv1 bias info %s", getTensorInfo(*conv1_bias).c_str());
    //allocate input and output vectors for first convolutional layer
    auto conv1_input = std::make_shared<arm_compute::Tensor>();
    conv1_input->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(224, 224, 3), 1, arm_compute::DataType::F32));
    conv1_input->allocator()->allocate();
    auto conv1_output = std::make_shared<arm_compute::Tensor>();
    conv1_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(54, 54, 96), 1, arm_compute::DataType::F32));
    conv1_output->allocator()->allocate();
    auto *conv1 = new CNN::Convolution(3, 96, 11, 11, 224, 224, 4, 0, 1, conv1_input, std::move(conv1_kernel), std::move(conv1_bias), conv1_output);
    alexnet.input_tensor = conv1_input;
    alexnet.addLayer(conv1);

    //2. LRN Layer;
    auto lrn1_output = std::make_shared<arm_compute::Tensor>();
    lrn1_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(54, 54, 96), 1, arm_compute::DataType::F32));
    lrn1_output->allocator()->allocate();
    auto *lrn1 = new LRN::LRN(96, 54, 54, 5, 0.00009999999747378752, 0.75f, 1.0f, conv1_output, lrn1_output);
    alexnet.addLayer(lrn1);

    //3. MAX Pooling Layer
    auto pooling1_output = std::make_shared<arm_compute::Tensor>();
    pooling1_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(26, 26, 96), 1, arm_compute::DataType::F32));
    pooling1_output->allocator()->allocate();
    auto *pooling1 = new POOL::Pooling(3, 3, 96, 54, 54, 2, 0,0,0,0, lrn1_output, pooling1_output);
    alexnet.addLayer(pooling1);

    //4. Conv Layer 2; Load kernel weights & biases
    auto conv2_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv2_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv2_kernel, readWeightVector("weights/alexnetconv2_w_0.bin"), arm_compute::TensorShape(5, 5, 96, 256));
    vectorToTensor(*conv2_bias, readWeightVector("weights/alexnetconv2_b_0.bin"), arm_compute::TensorShape(256));
    LOGI_lib("conv2 kernel info %s", getTensorInfo(*conv2_kernel).c_str());
    LOGI_lib("conv2 bias info %s", getTensorInfo(*conv2_bias).c_str());
    //allocate input and output vectors for first convolutional layer
    auto conv2_output = std::make_shared<arm_compute::Tensor>();
    conv2_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(26, 26, 256), 1, arm_compute::DataType::F32));
    conv2_output->allocator()->allocate();
    auto *conv2 = new CNN::Convolution(96, 256, 5, 5, 26, 26, 1, 2, 2, pooling1_output, std::move(conv2_kernel), std::move(conv2_bias), conv2_output);
    alexnet.addLayer(conv2);

    //5. LRN Layer 2
    auto lrn2_output = std::make_shared<arm_compute::Tensor>();
    lrn2_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(26, 26, 256), 1, arm_compute::DataType::F32));
    lrn2_output->allocator()->allocate();
    auto *lrn2 = new LRN::LRN(256, 26, 26, 5, 0.00009999999747378752, 0.75f, 1.0f, conv2_output, lrn2_output);
    alexnet.addLayer(lrn2);

    //6. Pooling Layer 2
    auto pooling2_output = std::make_shared<arm_compute::Tensor>();
    pooling2_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(12, 12, 256), 1, arm_compute::DataType::F32));
    pooling2_output->allocator()->allocate();
    auto *pooling2 = new POOL::Pooling(3, 3, 256, 26, 26, 2, 0,0,0,0, lrn2_output, pooling2_output);
    alexnet.addLayer(pooling2);

    //7. Conv Layer 3
    auto conv3_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv3_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv3_kernel, readWeightVector("weights/alexnetconv3_w_0.bin"), arm_compute::TensorShape(3, 3, 256, 384));
    vectorToTensor(*conv3_bias, readWeightVector("weights/alexnetconv3_b_0.bin"), arm_compute::TensorShape(384));
    LOGI_lib("conv3 kernel info %s", getTensorInfo(*conv3_kernel).c_str());
    LOGI_lib("conv3 bias info %s", getTensorInfo(*conv3_bias).c_str());
    //allocate input and output vectors for first convolutional layer
    auto conv3_output = std::make_shared<arm_compute::Tensor>();
    conv3_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(12, 12, 384), 1, arm_compute::DataType::F32));
    conv3_output->allocator()->allocate();
    auto *conv3 = new CNN::Convolution(256, 384, 3, 3, 12, 12, 1, 1, 1, pooling2_output, std::move(conv3_kernel), std::move(conv3_bias), conv3_output);
    alexnet.addLayer(conv3);

    //8. Conv Layer 4
    auto conv4_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv4_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv4_kernel, readWeightVector("weights/alexnetconv4_w_0.bin"), arm_compute::TensorShape(3, 3, 384, 256));
    vectorToTensor(*conv4_bias, readWeightVector("weights/alexnetconv4_b_0.bin"), arm_compute::TensorShape(384));
    LOGI_lib("conv4 kernel info %s", getTensorInfo(*conv4_kernel).c_str());
    LOGI_lib("conv4 bias info %s", getTensorInfo(*conv4_bias).c_str());
    //allocate input and output vectors for first convolutional layer
    auto conv4_output = std::make_shared<arm_compute::Tensor>();
    conv4_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(12, 12, 384), 1, arm_compute::DataType::F32));
    conv4_output->allocator()->allocate();
    auto *conv4 = new CNN::Convolution(384, 384, 3, 3, 12, 12, 1, 1, 2, conv3_output, std::move(conv4_kernel), std::move(conv4_bias), conv4_output);
    alexnet.addLayer(conv4);

    //9. Conv Layer 5
    auto conv5_kernel =std::make_unique<arm_compute::Tensor>();
    auto conv5_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv5_kernel, readWeightVector("weights/alexnetconv5_w_0.bin"), arm_compute::TensorShape(3, 3, 384, 256));
    vectorToTensor(*conv5_bias, readWeightVector("weights/alexnetconv5_b_0.bin"), arm_compute::TensorShape(256));
    LOGI_lib("conv5 kernel info %s", getTensorInfo(*conv5_kernel).c_str());
    LOGI_lib("conv5 bias info %s", getTensorInfo(*conv5_bias).c_str());
    //allocate input and output vectors for first convolutional layer
    auto conv5_output = std::make_shared<arm_compute::Tensor>();
    conv5_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(12, 12, 256), 1, arm_compute::DataType::F32));
    conv5_output->allocator()->allocate();
    auto *conv5 = new CNN::Convolution(384, 256, 3, 3, 12, 12, 1, 1, 2, conv4_output, std::move(conv5_kernel), std::move(conv5_bias), conv5_output);
    alexnet.addLayer(conv5);

    //10. Pooling Layer 3
    auto pooling3_output = std::make_shared<arm_compute::Tensor>();
    pooling3_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(6, 6, 256), 1, arm_compute::DataType::F32));
    pooling3_output->allocator()->allocate();
    auto *pooling3 = new POOL::Pooling(3, 3, 256, 12, 12, 2, 0,0,1,1, conv5_output, pooling3_output);
    alexnet.addLayer(pooling3);

    //11. Flatten Layer 1
    auto flatten1_output = std::make_shared<arm_compute::Tensor>();
    flatten1_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorInfo(arm_compute::TensorShape(9216), 1, arm_compute::DataType::F32)));
    flatten1_output->allocator()->allocate();
    auto flatten1 = new CNN::Flatten(pooling3_output, flatten1_output);
    alexnet.addLayer(flatten1);


    //12. Fully Connected Layer 1
    auto fc1_weights = std::make_unique<arm_compute::Tensor>();
    auto fc1_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc1_weights, readWeightVector("weights/alexnetfc6_w_0.bin"), arm_compute::TensorShape(4096, 9216));
    vectorToTensor(*fc1_bias, readWeightVector("weights/alexnetfc6_b_0.bin"), arm_compute::TensorShape(4096));
    LOGI_lib("fc1 weights info %s", getTensorInfo(*fc1_weights).c_str());
    LOGI_lib("fc1 bias info %s", getTensorInfo(*fc1_bias).c_str());
    auto fc1_output = std::make_shared<arm_compute::Tensor>();
    fc1_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorInfo(arm_compute::TensorShape(4096), 1, arm_compute::DataType::F32)));
    fc1_output->allocator()->allocate();
    auto *fc1 = new FC::FullyConnected(flatten1_output, std::move(fc1_weights), std::move(fc1_bias), fc1_output, 9216, 4096, 0);
    alexnet.addLayer(fc1);

    //13. Fully Connected Layer 2
    auto fc2_weights = std::make_unique<arm_compute::Tensor>();
    auto fc2_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc2_weights, readWeightVector("weights/alexnetfc7_w_0.bin"), arm_compute::TensorShape(4096, 4096));
    vectorToTensor(*fc2_bias, readWeightVector("weights/alexnetfc7_b_0.bin"), arm_compute::TensorShape(4096));
    LOGI_lib("fc2 weights info %s", getTensorInfo(*fc2_weights).c_str());
    LOGI_lib("fc2 bias info %s", getTensorInfo(*fc2_bias).c_str());
    auto fc2_output = std::make_shared<arm_compute::Tensor>();
    fc2_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorInfo(arm_compute::TensorShape(4096), 1, arm_compute::DataType::F32)));
    fc2_output->allocator()->allocate();
    auto *fc2 = new FC::FullyConnected(fc1_output, std::move(fc2_weights), std::move(fc2_bias), fc2_output, 4096, 4096, 0);
    alexnet.addLayer(fc2);

    //14. Fully Connected Layer 3
    auto fc3_weights = std::make_unique<arm_compute::Tensor>();
    auto fc3_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc3_weights, readWeightVector("weights/alexnetfc8_w_0.bin"), arm_compute::TensorShape(1000, 4096));
    vectorToTensor(*fc3_bias, readWeightVector("weights/alexnetfc8_b_0.bin"), arm_compute::TensorShape(1000));
    LOGI_lib("fc3 weights info %s", getTensorInfo(*fc3_weights).c_str());
    LOGI_lib("fc3 bias info %s", getTensorInfo(*fc3_bias).c_str());
    auto fc3_output = std::make_shared<arm_compute::Tensor>();
    fc3_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorInfo(arm_compute::TensorShape(1000), 1, arm_compute::DataType::F32)));
    fc3_output->allocator()->allocate();
    auto *fc3 = new FC::FullyConnected(fc2_output, std::move(fc3_weights), std::move(fc3_bias), fc3_output, 4096, 1000, 0);
    alexnet.addLayer(fc3);

    auto softmax1_output = std::make_shared<arm_compute::Tensor>();
    softmax1_output->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorInfo(arm_compute::TensorShape(1000), 1, arm_compute::DataType::F32)));
    softmax1_output->allocator()->allocate();
    auto *softmax = new CNN::Softmax(fc3_output, softmax1_output);
    alexnet.output_tensor = softmax1_output;
    alexnet.addLayer(softmax);
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_profiler_1call(
        JNIEnv *env, jobject thiz) {
    //test_fully_connected();
    //test_pooling();
    //test_convolution();
    generate_model_alexnet_acl();
    arm_compute::Scheduler::get().set_num_threads(12);
    //profile();
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_init_1assets(
        JNIEnv *env, jobject thiz, jobject java_asset_manager) {
    assetManager = AAssetManager_fromJava(env, java_asset_manager);
}


extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_run_1inference(
        JNIEnv *env, jobject thiz, jfloatArray image) {
    jfloat *cArray = env->GetFloatArrayElements(image, nullptr);
    if (cArray == nullptr) {
        // Handle error condition
        return nullptr;
    }
    // Get the size of the array
    jsize length = env->GetArrayLength(image);
    // Create a std::vector and copy the data
    std::vector<float> input_vec(cArray, cArray + length);
    std::vector<float> res = alexnet.forward(input_vec);
    LOGI_lib("prediction_size:%zu", res.size());
    // Release the array elements
    env->ReleaseFloatArrayElements(image, cArray, JNI_ABORT); // Use JNI_ABORT to not copy back changes
    auto indices = find_top_five_indices(res);
    // Create a new Java int array
    jintArray result = env->NewIntArray(indices.size());
    // Allocate a temporary buffer to hold the int values
    std::vector<jint> int_indices(indices.begin(), indices.end());
    // Copy the contents of the std::vector<jint> to the Java array
    env->SetIntArrayRegion(result, 0, int_indices.size(), int_indices.data());
    return result;
}


extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_run_1inference_1acl(
        JNIEnv *env, jobject thiz, jfloatArray image) {
    jfloat *cArray = env->GetFloatArrayElements(image, nullptr);
    if (cArray == nullptr or alexnet.input_tensor == nullptr) {
        // Handle error condition
        return nullptr;
    }
    // Ensure the tensor is allocated
    if (alexnet.input_tensor->info()->is_resizable()) {
        alexnet.input_tensor->allocator()->allocate();
    }

// Check if the buffer is valid
    if (alexnet.input_tensor->buffer() == nullptr) {
        // Handle error condition: Buffer not allocated
        LOGI_lib("Input Tensor Buffer is not allocated");
    }
    // Get the size of the array
    jsize length = env->GetArrayLength(image);
    // Create a std::vector and copy the data
    std::vector<float> imageVector(cArray, cArray + length);
    env->ReleaseFloatArrayElements(image, cArray, 0);

    // Step 2: Construct an input arm_compute::Tensor
    arm_compute::Tensor *inputTensor = alexnet.input_tensor.get();

    // Populate the tensor with image data
    std::copy(imageVector.begin(), imageVector.end(), reinterpret_cast<float*>(inputTensor->buffer()));
    // Time the execution of the forward method
    auto start = std::chrono::high_resolution_clock::now();
    alexnet.forward_acl();

    auto indices = find_top_five_indices(alexnet.output_tensor.get());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    LOGI_lib("inference-time:%lld", duration);
    // Create a new Java int array
    jintArray result = env->NewIntArray(indices.size());
    // Allocate a temporary buffer to hold the int values
    std::vector<jint> int_indices(indices.begin(), indices.end());
    // Copy the contents of the std::vector<jint> to the Java array
    env->SetIntArrayRegion(result, 0, int_indices.size(), int_indices.data());
    return result;
}