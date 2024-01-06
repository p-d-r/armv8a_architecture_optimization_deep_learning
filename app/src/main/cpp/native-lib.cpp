//
// Created by David Pauli (ti72teta) on 26.11.2023
//
#include <jni.h>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <onnxruntime_training/onnxruntime_training_cxx_api.h>
#include "/header/FullyConnected.h"


#define LOG_TAG "NativeCode" // Tag for logging
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Define a function pointer type for the forward methods
typedef std::vector<float> (FC::FullyConnected::*ForwardMethod)(const std::vector<float>&);


// Simple assertion function to compare vectors
void assert_equal(const std::vector<float>& expected, const std::vector<float>& actual, float epsilon = 0.01f) {
    if (expected.size() != actual.size()) {
        LOGI("FC Test failed: Size mismatch");
        return;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > epsilon) {
            LOGI("expected value: %f,  actual value: %f", expected[i], actual[i]);
            LOGI("FC Test failed: Value mismatch at index %d", i);
            return;
        }
    }

    LOGI("FC Test Passed!");
}

// Test function for FullyConnected class
void test_fully_connected() {
    // Initialize your FullyConnected object and test data
    std::vector<float> weights = {0.1, 0.2, 0.3, 0.4, 0.5,
                                  0.6, 0.7, 0.8, 0.9, 1.0,
                                  1.1, 1.2, 1.3, 1.4, 1.5,
                                  1.6, 1.7, 1.8, 1.9, 2.0};
    size_t input_size = 4;
    size_t output_size = 4;
    int tile_size = 2;

    FC::FullyConnected fc(weights, input_size, output_size, tile_size);
    std::vector<float> input = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> expected_output = {3.5, 9, 14.5, 20};

    // Test standard forward method
    std::vector<float> output_1 = fc.forward(input);
    assert_equal(expected_output, output_1);

    std::vector<float> output_2 = fc.forward_tiled(input);
    assert_equal(expected_output, output_2);

    assert_equal(output_1, output_2);
}


// Function to generate a random vector
std::vector<float> generateRandomTensor(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 10.0);

    std::vector<float> v(size);
    for (auto& element : v) {
        element = dis(gen);
    }
    return v;
}



std::vector<float> profile_fc(ForwardMethod method, size_t inputSize, size_t outputSize, int tile_size, size_t iterations) {
    std::vector<std::chrono::duration<double, std::micro>> times(iterations);
    std::vector<float> output;
    for (int i = 0; i < iterations; i++) {
        // Generate random weight-bias matrix with an additional row for bias
        std::vector<float> weight_bias_matrix = generateRandomTensor((inputSize + 1) * outputSize);
        // Generate a random input vector
        std::vector<float> input = generateRandomTensor(inputSize * 40);
        // Construct the layer
        FC::FullyConnected layer(weight_bias_matrix, inputSize, outputSize, tile_size);

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

    LOGI("tile_size: %d", tile_size);
    LOGI("forward:  min: %f us;   max: %f us;   median: %f us;   average: %f us", min_time.count(), max_time.count(), median_time.count(), average_time.count());
    return output;
}



int profile() {

    LOGI("start process!");
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





extern "C"
JNIEXPORT void JNICALL
Java_com_example_armv8a_1architecture_1optimization_1deep_1learning_MainActivity_profiler_1call(
        JNIEnv *env, jobject thiz) {
    //test_fully_connected();
    //generate_model_alexnet();
    profile();
}
