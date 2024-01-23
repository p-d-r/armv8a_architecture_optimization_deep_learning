//
// Created by David on 1/9/2024.
//
#include "../header/Testing.h"

// Simple assertion function to compare vectors
void assert_equal(const std::vector<float>& expected, const std::vector<float>& actual, float epsilon) {
    if (expected.size() != actual.size()) {
        LOGI("Test failed: Size mismatch expected: %zu, actual:%zu", expected.size(), actual.size());
        return;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > epsilon) {
            LOGI("expected value: %f,  actual value: %f", expected[i], actual[i]);
            LOGI("Test failed: Value mismatch at index %zu", i);
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
    std::vector<float> bias = {1.0, 2.0, 3.0, 4.0};
    size_t input_size = 5;
    size_t output_size = 4;
    int tile_size = 2;

    FC::FullyConnected fc(weights, bias, input_size, output_size, tile_size);
    std::vector<float> input = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<float> expected_output = {6.5, 15, 23.5, 32};

    // Test standard forward method
    std::vector<float> output_1 = fc.forward(input);
    assert_equal(expected_output, output_1);

    std::vector<float> output_2 = fc.forward_tiled(input);
    //assert_equal(expected_output, output_2);

    //assert_equal(output_1, output_2);


    // Initialize your FullyConnected object and test data
    weights = {1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0,
               13.0, 14.0, 15.0, 16.0};
    bias = {1.0, 2.0, 3.0, 4.0};
    input_size = 4;
    output_size = 4;
    tile_size = 2;

    FC::FullyConnected fc2(weights, bias, input_size, output_size, tile_size);
    input = {2.0, 3.0, 4.0, 5.0,
             6.0, 7.0, 8.0, 9.0,
             10.0, 11.0, 12.0, 13.0,
             14.0, 15.0, 16.0, 17.0};
    expected_output = {101.0, 111.0, 121.0, 131.0,
                       230.0, 256.0, 282.0, 308.0,
                       359.0, 401.0, 443.0, 485.0,
                       488.0, 546.0, 604.0, 662.0};

    // Test standard forward method
    output_1 = fc2.forward(input);
    assert_equal(expected_output, output_1);

    output_2 = fc2.forward_tiled(input);
    //assert_equal(expected_output, output_2);

    //assert_equal(output_1, output_2);
}


void test_pooling() {
    size_t pool_height = 2;
    size_t pool_width = 2;
    size_t channels = 1;
    size_t input_height = 4;
    size_t input_width = 4;
    size_t stride = 2;

    POOL::Pooling poolingLayer(pool_height, pool_width, channels, input_height, input_width, stride, 0,0,0,0);
    std::vector<float> input = {
            // Channel 1
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
    };
    std::vector<float> expected_output = {
            // Channel 1
            6, 8,
            14, 16
    };

    std::vector<float> output = poolingLayer.forward(input);
    assert_equal(expected_output, output);

    // Test with multiple channels
    pool_height = 2;
    pool_width = 2;
    channels = 2;  // Two channels this time
    input_height = 4;
    input_width = 4;
    stride = 2;

    POOL::Pooling poolingLayerMulti(pool_height, pool_width, channels, input_height, input_width, stride, 0,0,0,0);
    std::vector<float> inputMulti = {
            // Channel 1
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            // Channel 2
            17, 18, 19, 20,
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 32
    };
    std::vector<float> expected_outputMulti = {
            // Channel 1
            6, 8,
            14, 16,
            // Channel 2
            22, 24,
            30, 32
    };

    std::vector<float> outputMulti = poolingLayerMulti.forward(inputMulti);
    assert_equal(expected_outputMulti, outputMulti);
}


void test_convolution() {
    size_t in_channels = 1;
    size_t out_channels = 1;
    size_t kernel_height = 2;
    size_t kernel_width = 2;
    size_t input_height = 3;
    size_t input_width = 3;
    size_t stride = 1;
    size_t padding = 0;

    CNN::Convolution convLayer(in_channels, out_channels, kernel_height, kernel_width, input_height, input_width, stride, padding);
    // Initialize weights and bias
    convLayer.setWeights({1, 0, 0, 1}); // Simple identity kernel
    convLayer.setBias({0});             // No bias

    std::vector<float> input = {
            // Channel 1
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
    };
    std::vector<float> expected_output = {
            // Channel 1
            6, 8,
            12, 14
    };

    std::vector<float> output = convLayer.forward(input);
    assert_equal(expected_output, output);

    in_channels = 2;  // Two input channels
    out_channels = 2;  // Two output channels
    kernel_height = 2;
    kernel_width = 2;
    input_height = 4;
    input_width = 4;
    stride = 2;
    padding = 0;

    CNN::Convolution convLayerMulti(in_channels, out_channels, kernel_height, kernel_width, input_height, input_width, stride, padding);
    // Initialize weights and bias for a more complex kernel
    convLayerMulti.setWeights({
                                      1, 0, -1, 0,  // Kernel for first input channel to first output channel
                                      0, 1, 0, -1,  // Kernel for second input channel to first output channel
                                      0, 0, 0, 0,  // Kernel for first input channel to second output channel
                                      1, 1, 1, 1   // Kernel for second input channel to second output channel
                              });
    convLayerMulti.setBias({0, 0});  // No bias for simplicity

    std::vector<float> inputMulti = {
            // Channel 1
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            // Channel 2
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1
    };

    std::vector<float> expected_outputMulti = {
            // Output Channel 1
            0, 0,
            0, 0,
            // Output Channel 2
            54, 46,
            22, 14,
    };

    std::vector<float> outputMulti = convLayerMulti.forward(inputMulti);
    assert_equal(expected_outputMulti, outputMulti);
}