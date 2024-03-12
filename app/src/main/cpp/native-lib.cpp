//
// Created by David Pauli (ti72teta) on 26.11.2023
//
#include <jni.h>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <android/log.h>
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
#include "header/AssetManagement.h"
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
typedef std::vector<float> (CNN::FullyConnected::*ForwardMethod)(const std::vector<float>&);


// Declare a global network object
CNN::Network alexnet(arm_compute::DataLayout::NCHW);
CNN::Network vgg16(arm_compute::DataLayout::NCHW);



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
        CNN::FullyConnected layer(weight_matrix, bias_vector, inputSize, outputSize, tile_size);

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
    auto out_seq0 = profile_fc(&CNN::FullyConnected::forward, inputSize, outputSize, 0, iterations);
    auto out_seq1 = profile_fc(&CNN::FullyConnected::forward_transposed_in, inputSize, outputSize, 0, iterations);
    auto out_seq2 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 4, iterations);
    auto out_seq2_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 4, iterations);
    auto out_seq3 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 8, iterations);
    auto out_seq3_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 8, iterations);
    auto out_seq4 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 16, iterations);
    auto out_seq4_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 16, iterations);
    auto out_seq5 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 32, iterations);
    auto out_seq5_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 32, iterations);
    auto out_seq6 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 64, iterations);
    auto out_seq6_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 64, iterations);
    auto out_seq7 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 128, iterations);
    auto out_seq7_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 128, iterations);
    auto out_seq8 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 256, iterations);
    auto out_seq8_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 256, iterations);
    auto out_seq9 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 512, iterations);
    auto out_seq9_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 512, iterations);
    auto out_seq10 = profile_fc(&CNN::FullyConnected::forward_tiled, inputSize, outputSize, 1024, iterations);
    auto out_seq10_t = profile_fc(&CNN::FullyConnected::forward_tiled_transposed_in, inputSize, outputSize, 1024, iterations);
    return 0; // Return type changed from 'void' to 'int'
}



void generate_model_alexnet() {
    std::vector<float> conv1_kernel = read_binary_float_vector_asset("weights/alexnetconv1_w_0.bin");
    std::vector<float> conv1_bias = read_binary_float_vector_asset("weights/alexnetconv1_b_0.bin");
    LOGI_lib("conv1 dimensions: %zu    %zu", conv1_kernel.size(), conv1_bias.size());
    auto *conv1 = new CNN::Convolution(3, 96, 11, 11, 224, 224, 4, 0);
    conv1->setWeights(conv1_kernel);
    conv1->setBias(conv1_bias);
    alexnet.addLayer(conv1);

    auto *lrn1 = new LRN::LRN(96, 54, 54, 5, 0.00009999999747378752, 0.75f, 1.0f);
    alexnet.addLayer(lrn1);

    auto *pooling1 = new CNN::Pooling(3, 3, 96, 54, 54, 2, 0,0,0,0);
    alexnet.addLayer(pooling1);

    std::vector<float> conv2_kernel = read_binary_float_vector_asset("weights/alexnetconv2_w_0.bin");
    std::vector<float> conv2_bias = read_binary_float_vector_asset("weights/alexnetconv2_b_0.bin");
    LOGI_lib("conv2 dimensions: %zu    %zu", conv2_kernel.size(), conv2_bias.size());
    auto *conv2 = new CNN::Convolution(96, 256, 5, 5, 26, 26, 1, 2);
    conv2->setWeights(conv2_kernel);
    conv2->setBias(conv2_bias);
    alexnet.addLayer(conv2);

    auto *lrn2 = new LRN::LRN(256, 26, 26, 5, 0.00009999999747378752, 0.75f, 1.0f);
    alexnet.addLayer(lrn2);

    auto *pooling2 = new CNN::Pooling(3, 3, 256, 26, 26, 2, 0,0,0,0);
    alexnet.addLayer(pooling2);

    std::vector<float> conv3_kernel = read_binary_float_vector_asset("weights/alexnetconv3_w_0.bin");
    std::vector<float> conv3_bias = read_binary_float_vector_asset("weights/alexnetconv3_b_0.bin");
    LOGI_lib("conv3 dimensions: %zu    %zu", conv3_kernel.size(), conv3_bias.size());
    auto *conv3 = new CNN::Convolution(256, 384, 3, 3, 12, 12, 1, 1);
    conv3->setWeights(conv3_kernel);
    conv3->setBias(conv3_bias);
    alexnet.addLayer(conv3);

    std::vector<float> conv4_kernel = read_binary_float_vector_asset("weights/alexnetconv4_w_0.bin");
    std::vector<float> conv4_bias = read_binary_float_vector_asset("weights/alexnetconv4_b_0.bin");
    LOGI_lib("conv4 dimensions: %zu    %zu", conv4_kernel.size(), conv4_bias.size());
    auto *conv4 = new CNN::Convolution(384, 384, 3, 3, 12, 12, 1, 1);
    conv4->setWeights(conv4_kernel);
    conv4->setBias(conv4_bias);
    alexnet.addLayer(conv4);

    std::vector<float> conv5_kernel = read_binary_float_vector_asset("weights/alexnetconv5_w_0.bin");
    std::vector<float> conv5_bias = read_binary_float_vector_asset("weights/alexnetconv5_b_0.bin");
    LOGI_lib("conv5 dimensions: %zu    %zu", conv5_kernel.size(), conv5_bias.size());
    auto *conv5 = new CNN::Convolution(384, 256, 3, 3, 12, 12, 1, 1);
    conv5->setWeights(conv5_kernel);
    conv5->setBias(conv5_bias);
    alexnet.addLayer(conv5);

    auto *pooling3 = new CNN::Pooling(3, 3, 256, 12, 12, 2, 0,0,1,1);
    alexnet.addLayer(pooling3);

    std::vector<float> fc6_weights = read_binary_float_vector_asset("weights/alexnetfc6_w_0.bin");
    std::vector<float> fc6_bias = read_binary_float_vector_asset("weights/alexnetfc6_b_0.bin");
    LOGI_lib("fc6 dimensions: %zu    %zu", fc6_weights.size(), fc6_bias.size());
    auto *fc6 = new CNN::FullyConnected(fc6_weights, fc6_bias, 9216, 4096, 0);
    alexnet.addLayer(fc6);

    std::vector<float> fc7_weights = read_binary_float_vector_asset("weights/alexnetfc7_w_0.bin");
    std::vector<float> fc7_bias = read_binary_float_vector_asset("weights/alexnetfc7_b_0.bin");
    LOGI_lib("fc7 dimensions: %zu    %zu", fc7_weights.size(), fc7_bias.size());
    auto *fc7 = new CNN::FullyConnected(fc7_weights, fc7_bias, 4096, 4096, 0);
    alexnet.addLayer(fc7);

    std::vector<float> fc8_weights = read_binary_float_vector_asset("weights/alexnetfc8_w_0.bin");
    std::vector<float> fc8_bias = read_binary_float_vector_asset("weights/alexnetfc8_b_0.bin");
    LOGI_lib("fc8 dimensions: %zu    %zu", fc8_weights.size(), fc8_bias.size());
    auto *fc8 = new CNN::FullyConnected(fc8_weights, fc8_bias, 4096, 1000, 0);
    alexnet.addLayer(fc8);

    auto *softmax = new CNN::Softmax();
    alexnet.addLayer(softmax);
}



void generate_model_alexnet_acl() {
    auto conv0_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv0_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv0_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_0_weights_nchw.bin"), arm_compute::TensorShape(11, 11, 3, 64), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv0_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_0_bias.bin"), arm_compute::TensorShape(64), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv0 = new CNN::Convolution(3, 64, 11, 11, 224, 224, 4, 2, 1, std::move(conv0_kernel), std::move(conv0_bias));
    alexnet.addLayer(conv0, arm_compute::TensorShape(224, 224, 3), arm_compute::TensorShape(55, 55, 64));
    conv0->configure_acl();

    auto *pool0 = new CNN::Pooling(3, 3, 64, 55, 55, 2, 0,0,0,0);
    alexnet.addLayer(pool0, arm_compute::TensorShape(55, 55, 64), arm_compute::TensorShape(27, 27, 64));
    pool0->configure_acl();

    auto conv1_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv1_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv1_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_1_weights_nchw.bin"), arm_compute::TensorShape(5, 5, 64, 192), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv1_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_1_bias.bin"), arm_compute::TensorShape(192), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv1 = new CNN::Convolution(64, 192, 5, 5, 27, 27, 1, 2, 1, std::move(conv1_kernel), std::move(conv1_bias));
    alexnet.addLayer(conv1, arm_compute::TensorShape(27, 27, 64), arm_compute::TensorShape(27, 27, 192));
    conv1->configure_acl();

    auto *pool1 = new CNN::Pooling(3, 3, 192, 27, 27, 2, 0,0,0,0);
    alexnet.addLayer(pool1, arm_compute::TensorShape(27, 27, 192), arm_compute::TensorShape(13, 13, 192));
    pool1->configure_acl();

    auto conv2_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv2_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv2_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_2_weights_nchw.bin"), arm_compute::TensorShape(3, 3, 192, 384), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv2_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_2_bias.bin"), arm_compute::TensorShape(384), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv2 = new CNN::Convolution(192, 384, 3, 3, 13, 13, 1, 1, 1, std::move(conv2_kernel), std::move(conv2_bias));
    alexnet.addLayer(conv2, arm_compute::TensorShape(13, 13, 192), arm_compute::TensorShape(13, 13, 384));
    conv2->configure_acl();

    auto conv3_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv3_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv3_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_3_weights_nchw.bin"), arm_compute::TensorShape(3, 3, 384, 256), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv3_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_3_bias.bin"), arm_compute::TensorShape(256), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv3 = new CNN::Convolution(192, 384, 3, 3, 13, 13, 1, 1, 1, std::move(conv3_kernel), std::move(conv3_bias));
    alexnet.addLayer(conv3, arm_compute::TensorShape(13, 13, 384), arm_compute::TensorShape(13, 13, 256));
    conv3->configure_acl();

    auto conv4_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv4_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv4_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_4_weights_nchw.bin"), arm_compute::TensorShape(3, 3, 256, 256), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv4_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_4_bias.bin"), arm_compute::TensorShape(256), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv4 = new CNN::Convolution(256, 256, 3, 3, 13, 13, 1, 1, 1, std::move(conv4_kernel), std::move(conv4_bias));
    alexnet.addLayer(conv4, arm_compute::TensorShape(13, 13, 256), arm_compute::TensorShape(13, 13, 256));
    conv4->configure_acl();

    auto *pool2 = new CNN::Pooling(3, 3, 256, 13, 13, 2, 0,0,0,0);
    alexnet.addLayer(pool2, arm_compute::TensorShape(13, 13, 256), arm_compute::TensorShape(6, 6, 256));
    pool2->configure_acl();

    auto fc_0_weights = std::make_unique<arm_compute::Tensor>();
    auto fc_0_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc_0_weights, read_binary_float_vector_asset("alexnet_torch/weights/fc0_weights.bin"), arm_compute::TensorShape(9216, 4096), arm_compute::DataLayout::NCHW);
    vectorToTensor(*fc_0_bias,read_binary_float_vector_asset("alexnet_torch/weights/fc0_bias.bin"), arm_compute::TensorShape(4096), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *fc0 = new CNN::FullyConnected(std::move(fc_0_weights), std::move(fc_0_bias), 9216, 4096, 0);
    alexnet.addLayer(fc0, arm_compute::TensorShape(9216), arm_compute::TensorShape(4096));
    fc0->configure_acl();

    auto fc_1_weights = std::make_unique<arm_compute::Tensor>();
    auto fc_1_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc_1_weights, read_binary_float_vector_asset("alexnet_torch/weights/fc1_weights.bin"), arm_compute::TensorShape(4096, 4096), arm_compute::DataLayout::NCHW);
    vectorToTensor(*fc_1_bias,read_binary_float_vector_asset("alexnet_torch/weights/fc1_bias.bin"), arm_compute::TensorShape(4096), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *fc1 = new CNN::FullyConnected(std::move(fc_1_weights), std::move(fc_1_bias), 4096, 4096, 0);
    alexnet.addLayer(fc1, arm_compute::TensorShape(4096), arm_compute::TensorShape(4096));
    fc1->configure_acl();

    auto fc_2_weights = std::make_unique<arm_compute::Tensor>();
    auto fc_2_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc_2_weights, read_binary_float_vector_asset("alexnet_torch/weights/fc2_weights.bin"), arm_compute::TensorShape(4096, 1000), arm_compute::DataLayout::NCHW);
    vectorToTensor(*fc_2_bias,read_binary_float_vector_asset("alexnet_torch/weights/fc2_bias.bin"), arm_compute::TensorShape(1000), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *fc2 = new CNN::FullyConnected(std::move(fc_2_weights), std::move(fc_2_bias), 4096, 1000, 0, arm_compute::ActivationLayerInfo::ActivationFunction::LINEAR);
    alexnet.addLayer(fc2, arm_compute::TensorShape(4096), arm_compute::TensorShape(1000));
    fc2->configure_acl();
}



extern "C"
JNIEXPORT void JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_profiler_1call(
        JNIEnv *env, jobject thiz) {
    //test_fully_connected();
    test_pooling_acl();
    //test_convolution_acl();
    test_alexnet_torch_nchw();
    generate_model_alexnet_acl();
    //arm_compute::Scheduler::get().set_num_threads(1);
    //profile();
}


//This function has to be located in native-lib.cpp because of JNI linking
//This function has to be called before loading any resources of the packaged assets folder
extern "C"
JNIEXPORT void JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_init_1assets(
        JNIEnv *env, jobject thiz, jobject java_asset_manager) {
    set_up_asset_manager(env, java_asset_manager);
}


//pass a preprocessed image from the camera-stream as float array and run inference
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
        LOGI_lib("Network is not valid");
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

    //TODO remove test: directly load dog binary data
    //auto dog_vector = read_binary_float_vector_asset("flattened_dog.bin");
    //std::copy(dog_vector.begin(), dog_vector.end(), reinterpret_cast<float*>(inputTensor->buffer()));

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