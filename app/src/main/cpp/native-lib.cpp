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
#include "/header/FullyConnected.h"
#include "/header/Pooling.h"
#include "/header/Convolution.h"
#include "/header/Network.h"
#include "/header/Testing.h"
#include "/header/Helpers.h"
#include "/header/LRN.h"
#include "/header/Softmax.h"
#include "/onnxruntime/onnxruntime_cxx_api.h"
#include "/arm/arm_compute/runtime/Tensor.h"

#define LOG_TAG_lib "NativeCode:native-lib" // Tag for logging
#define LOGI_lib(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_lib, __VA_ARGS__)


// Define a function pointer type for the forward methods
typedef std::vector<float> (FC::FullyConnected::*ForwardMethod)(const std::vector<float>&);

AAssetManager* assetManager = nullptr;
// Declare a global network object
NETWORK::Network alexnet;

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


std::vector<float> readAssetAsFloats(const std::string& filename) {
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
    std::vector<float> conv1_kernel = readAssetAsFloats("weights/alexnetconv1_w_0.bin");
    std::vector<float> conv1_bias = readAssetAsFloats("weights/alexnetconv1_b_0.bin");
    LOGI_lib("conv1 dimensions: %zu    %zu", conv1_kernel.size(), conv1_bias.size());
    auto *conv1 = new CNN::Convolution(3, 96, 11, 11, 224, 224, 4, 0);
    conv1->setWeights(conv1_kernel);
    conv1->setBias(conv1_bias);
    alexnet.addLayer(conv1);

    auto *lrn1 = new LRN::LRN(96, 54, 54, 5, 0.00009999999747378752, 0.75f, 1.0f);
    alexnet.addLayer(lrn1);

    auto *pooling1 = new POOL::Pooling(3, 3, 96, 54, 54, 2, 0,0,0,0);
    alexnet.addLayer(pooling1);

    std::vector<float> conv2_kernel = readAssetAsFloats("weights/alexnetconv2_w_0.bin");
    std::vector<float> conv2_bias = readAssetAsFloats("weights/alexnetconv2_b_0.bin");
    LOGI_lib("conv2 dimensions: %zu    %zu", conv2_kernel.size(), conv2_bias.size());
    auto *conv2 = new CNN::Convolution(96, 256, 5, 5, 26, 26, 1, 2);
    conv2->setWeights(conv2_kernel);
    conv2->setBias(conv2_bias);
    alexnet.addLayer(conv2);

    auto *lrn2 = new LRN::LRN(256, 26, 26, 5, 0.00009999999747378752, 0.75f, 1.0f);
    alexnet.addLayer(lrn2);

    auto *pooling2 = new POOL::Pooling(3, 3, 256, 26, 26, 2, 0,0,0,0);
    alexnet.addLayer(pooling2);

    std::vector<float> conv3_kernel = readAssetAsFloats("weights/alexnetconv3_w_0.bin");
    std::vector<float> conv3_bias = readAssetAsFloats("weights/alexnetconv3_b_0.bin");
    LOGI_lib("conv3 dimensions: %zu    %zu", conv3_kernel.size(), conv3_bias.size());
    auto *conv3 = new CNN::Convolution(256, 384, 3, 3, 12, 12, 1, 1);
    conv3->setWeights(conv3_kernel);
    conv3->setBias(conv3_bias);
    alexnet.addLayer(conv3);

    std::vector<float> conv4_kernel = readAssetAsFloats("weights/alexnetconv4_w_0.bin");
    std::vector<float> conv4_bias = readAssetAsFloats("weights/alexnetconv4_b_0.bin");
    LOGI_lib("conv4 dimensions: %zu    %zu", conv4_kernel.size(), conv4_bias.size());
    auto *conv4 = new CNN::Convolution(384, 384, 3, 3, 12, 12, 1, 1);
    conv4->setWeights(conv4_kernel);
    conv4->setBias(conv4_bias);
    alexnet.addLayer(conv4);

    std::vector<float> conv5_kernel = readAssetAsFloats("weights/alexnetconv5_w_0.bin");
    std::vector<float> conv5_bias = readAssetAsFloats("weights/alexnetconv5_b_0.bin");
    LOGI_lib("conv5 dimensions: %zu    %zu", conv5_kernel.size(), conv5_bias.size());
    auto *conv5 = new CNN::Convolution(384, 256, 3, 3, 12, 12, 1, 1);
    conv5->setWeights(conv5_kernel);
    conv5->setBias(conv5_bias);
    alexnet.addLayer(conv5);

    auto *pooling3 = new POOL::Pooling(3, 3, 256, 12, 12, 2, 0,0,1,1);
    alexnet.addLayer(pooling3);

    std::vector<float> fc6_weights = readAssetAsFloats("weights/alexnetfc6_w_0.bin");
    std::vector<float> fc6_bias = readAssetAsFloats("weights/alexnetfc6_b_0.bin");
    LOGI_lib("fc6 dimensions: %zu    %zu", fc6_weights.size(), fc6_bias.size());
    auto *fc6 = new FC::FullyConnected(fc6_weights, fc6_bias, 9216, 4096, 0);
    alexnet.addLayer(fc6);

    std::vector<float> fc7_weights = readAssetAsFloats("weights/alexnetfc7_w_0.bin");
    std::vector<float> fc7_bias = readAssetAsFloats("weights/alexnetfc7_b_0.bin");
    LOGI_lib("fc7 dimensions: %zu    %zu", fc7_weights.size(), fc7_bias.size());
    auto *fc7 = new FC::FullyConnected(fc7_weights, fc7_bias, 4096, 4096, 0);
    alexnet.addLayer(fc7);

    std::vector<float> fc8_weights = readAssetAsFloats("weights/alexnetfc8_w_0.bin");
    std::vector<float> fc8_bias = readAssetAsFloats("weights/alexnetfc8_b_0.bin");
    LOGI_lib("fc8 dimensions: %zu    %zu", fc8_weights.size(), fc8_bias.size());
    auto *fc8 = new FC::FullyConnected(fc8_weights, fc8_bias, 4096, 1000, 0);
    alexnet.addLayer(fc8);

    auto *softmax = new CNN::Softmax();
    alexnet.addLayer(softmax);
}




extern "C"
JNIEXPORT void JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_profiler_1call(
        JNIEnv *env, jobject thiz) {
    test_fully_connected();
    test_pooling();
    test_convolution();
    generate_model_alexnet();
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