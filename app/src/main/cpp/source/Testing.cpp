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


void printTensor(const arm_compute::Tensor& tensor) {
    // Ensure tensor is allocated
    if (!tensor.info()->is_resizable()) {
        const auto tensor_info = tensor.info();
        const auto shape = tensor_info->tensor_shape();
        const size_t total_elements = shape.total_size();

        // Assuming the tensor holds floats
        auto data_ptr = reinterpret_cast<const float*>(tensor.buffer());

        for (size_t i = 0; i < total_elements; ++i) {
            LOGI("element: %zu, value:    %f", i, data_ptr[i]);
        }
    } else {
        LOGI("tensor is not allocated");
    }
}



// Test function for FullyConnected class
void test_fully_connected() {
    // Initialize your FullyConnected object and test data
    std::vector<float> weights = {0.1, 0.2, 0.3, 0.4, 0.5,
                                  0.6, 0.7, 0.8, 0.9, 1.0,
                                  1.1, 1.2, 1.3, 1.4, 1.5,
                                  1.6, 1.7, 1.8, 1.9, 2.0};
    std::vector<float> weights_T = {0.1, 0.6, 1.1, 1.6,
                                    0.2, 0.7, 1.2, 1.7,
                                    0.3, 0.8, 1.3, 1.8,
                                    0.4, 0.9, 1.4, 1.9,
                                    0.5, 1.0, 1.5, 2.0};
    std::vector<float> bias = {1.0, 2.0, 3.0, 4.0};
    size_t input_size = 5;
    size_t output_size = 4;
    int tile_size = 2;

    FC::FullyConnected fc(weights, bias, input_size, output_size, tile_size);
    std::vector<float> input = {1.0,
                                2.0,
                                3.0,
                                4.0,
                                5.0};
    std::vector<float> expected_output = {6.5, 15, 23.5, 32};

    // Test standard forward method
    std::vector<float> output_1 = fc.forward(input);
    assert_equal(expected_output, output_1);

//    // Initialize ACL tensors for the new version
//    arm_compute::Tensor acl_input, acl_output;
//    acl_input.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(input_size, 1), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_output.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(output_size, 1), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_input.allocator()->allocate();
//    acl_output.allocator()->allocate();
//    std::copy(input.begin(), input.end(), reinterpret_cast<float*>(acl_input.buffer()));
//    LOGI("print input content");
//    printTensor(acl_input);
//    arm_compute::Tensor *acl_weights;
//    arm_compute::Tensor *acl_bias;
//    vectorToTensor(*acl_weights, weights, arm_compute::TensorShape(4,5));
//    vectorToTensor(*acl_bias, bias, arm_compute::TensorShape(4));
//
//    // Initialize and test the ACL-based FullyConnected object
//    FC::FullyConnected fc_acl(&acl_input, acl_weights, acl_bias, &acl_output, input_size, output_size, tile_size);
//    LOGI("print weights");
//    printTensor(*acl_weights);
//    LOGI("print bias");
//    printTensor(*acl_bias);
//    fc_acl.forward_acl(); // Assuming forward method modified to work without arguments
//
//    // Extract output from ACL tensor
//    std::vector<float> output_acl(output_size);
//    std::copy(reinterpret_cast<float*>(acl_output.buffer()), reinterpret_cast<float*>(acl_output.buffer()) + output_size, output_acl.begin());
//
//    LOGI("print output content");
//    printTensor(acl_output);
//    // Compare ACL output with original output
//    assert_equal(output_1, output_acl);

    // Initialize your FullyConnected object and test data
    weights = {1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0,
               13.0, 14.0, 15.0, 16.0};
    weights_T = {1.0, 5.0, 9.0, 13.0,
                 2.0, 6.0, 10.0, 14.0,
                 3.0, 7.0, 11.0, 15.0,
                 4.0, 8.0, 12.0, 16.0};
    bias = {1.0, 2.0, 3.0, 4.0};
    input_size = 4;
    output_size = 4;
    tile_size = 2;

    FC::FullyConnected fc2(weights, bias, input_size, output_size, tile_size);
    input = {2.0, 3.0, 4.0, 5.0,
             6.0, 7.0, 8.0, 9.0,
             10.0, 11.0, 12.0, 13.0,
             14.0, 15.0, 16.0, 17.0};

    //actually this is the result of WEIGHTS*INPUT, not INPUT*WEIGHTS
//    expected_output = {101.0, 111.0, 121.0, 131.0,
//                       230.0, 256.0, 282.0, 308.0,
//                       359.0, 401.0, 443.0, 485.0,
//                       488.0, 546.0, 604.0, 662.0};

    expected_output = {119.0, 133.0, 147.0, 161.0,
                       232.0, 262.0, 292.0, 322.0,
                       345.0, 391.0, 437.0, 483.0,
                       458.0, 520.0, 582.0, 644.0};

    // Test standard forward method
    output_1 = fc2.forward(input);
    assert_equal(expected_output, output_1);

//    // Initialize ACL tensors for the new version
//    arm_compute::Tensor acl_input_2, acl_output_2;
//    acl_input_2.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(input_size, output_size), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_output_2.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(output_size, output_size), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_input_2.allocator()->allocate();
//    acl_output_2.allocator()->allocate();
//    std::copy(input.begin(), input.end(), reinterpret_cast<float*>(acl_input_2.buffer()));
//
//    LOGI("print input content");
//    printTensor(acl_input_2);
//    arm_compute::Tensor *acl_weights_2;
//    arm_compute::Tensor *acl_bias_2;
//    vectorToTensor(*acl_weights_2, weights, arm_compute::TensorShape(4,4));
//    vectorToTensor(*acl_bias_2, bias, arm_compute::TensorShape(4));
//
//    // Initialize and test the ACL-based FullyConnected object
//    FC::FullyConnected fc_acl_2(&acl_input_2, acl_weights_2, acl_bias_2, &acl_output_2, input_size, output_size, tile_size);
//    fc_acl_2.forward_acl(); // Assuming forward method modified to work without arguments
//
//    // Extract output from ACL tensor
//    std::vector<float> output_acl_2(output_size*output_size);
//    std::copy(reinterpret_cast<float*>(acl_output_2.buffer()), reinterpret_cast<float*>(acl_output_2.buffer()) + output_size, output_acl_2.begin());
//
//    LOGI("print output content");
//    printTensor(acl_output_2);
//
//    assert_equal(expected_output, output_acl_2);
//    acl_input.allocator()->free();
//    acl_output.allocator()->free();
//    acl_input_2.allocator()->free();
//    acl_output_2.allocator()->free();
//    acl_weights->allocator()->free();
//    acl_weights_2->allocator()->free();
//    acl_bias->allocator()->free();
//    acl_bias_2->allocator()->free();
}


void test_pooling() {
    size_t pool_height = 2;
    size_t pool_width = 2;
    size_t channels = 1;
    size_t input_height = 4;
    size_t input_width = 4;
    size_t stride = 2;

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

//    arm_compute::Tensor acl_input, acl_output;
//    acl_input.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(4, 4), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_output.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(2, 2), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_input.allocator()->allocate();
//    acl_output.allocator()->allocate();
//    std::copy(input.begin(), input.end(), reinterpret_cast<float*>(acl_input.buffer()));
//
//    POOL::Pooling poolingLayer(pool_height, pool_width, channels, input_height, input_width, stride, 0,0,0,0, &acl_input, &acl_output);
//    LOGI("Pooling input:");
//    printTensor(poolingLayer.input_tensor);
//    std::vector<float> output = poolingLayer.forward(input);
//    assert_equal(expected_output, output);
//    poolingLayer.forward_acl();
//    auto output_2 = poolingLayer.output_tensor;
//    LOGI("Pooling ACL output:");
//    printTensor(*output_2);
//
//    // Extract output from ACL tensor
//    std::vector<float> output_acl(2*2);
//    std::copy(reinterpret_cast<float*>(acl_output.buffer()), reinterpret_cast<float*>(acl_output.buffer()) + 2*2, output_acl.begin());
//    assert_equal(expected_output, output_acl);

    // Test with multiple channels
    pool_height = 2;
    pool_width = 2;
    channels = 2;  // Two channels this time
    input_height = 4;
    input_width = 4;
    stride = 2;

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

//    arm_compute::Tensor acl_input_2, acl_output_2;
//    acl_input_2.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(4, 4, 2), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_output_2.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(2, 2, 2), 1, arm_compute::DataType::F32)); // Assuming batch size of 1
//    acl_input_2.allocator()->allocate();
//    acl_output_2.allocator()->allocate();
//    std::copy(inputMulti.begin(), inputMulti.end(), reinterpret_cast<float*>(acl_input_2.buffer()));
//    POOL::Pooling poolingLayerMulti(pool_height, pool_width, channels, input_height, input_width, stride, 0,0,0,0, &acl_input_2, &acl_output_2);
//    std::vector<float> outputMulti = poolingLayerMulti.forward(inputMulti);
//    poolingLayerMulti.forward_acl();
//    assert_equal(expected_outputMulti, outputMulti);
//
//    LOGI("Input Pooling 2 Channels");
//    printTensor(*poolingLayerMulti.input_tensor());
//    LOGI("Output Pooling 2 Channels");
//    printTensor(*poolingLayerMulti.output_tensor());
//    // Extract output from ACL tensor
//    std::vector<float> output_acl_2(2*2*2);
//    std::copy(reinterpret_cast<float*>(acl_output_2.buffer()), reinterpret_cast<float*>(acl_output_2.buffer()) + 2*2*2, output_acl_2.begin());
//    assert_equal(expected_outputMulti, output_acl_2);
//
//    acl_input.allocator()->free();
//    acl_output.allocator()->free();
//    acl_input_2.allocator()->free();
//    acl_output_2.allocator()->free();
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