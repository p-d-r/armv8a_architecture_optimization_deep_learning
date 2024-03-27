//
// Created by David Pauli (ti72teta) on 26.11.2023
//

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "../header/Profiler.h"
#include "../header/FullyConnected.h"


// Include or copy the FullyConnectedLayer class definition here

// Function to generate a random vector
std::vector<float> generateRandomVector(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    std::vector<float> v(size);
    for (auto& element : v) {
        element = dis(gen);
    }
    return v;
}

void profile_acl() {
    int iterations = 500;
    std::vector<long> times(iterations);
    for (int i = 0; i < iterations; i++) {
        times[i] = profile_neconv_acl_small();
        LOGI_PROFILE("run %us, time: %ld", i, times[i]);
    }
    // Find minimum and maximum times
    auto min_time = *std::min_element(times.begin(), times.end());
    auto max_time = *std::max_element(times.begin(), times.end());

    // Calculate average time
    long long total_time = 0;
    for (const auto& time : times) {
        total_time += time;
    }
    auto average_time = total_time / iterations;

    // Sort times for median calculation
    std::sort(times.begin(), times.end());
    long median_time{};
    if (iterations % 2 == 0) {
        // Even number of elements: average of the two middle elements
        median_time = (times[iterations / 2 - 1] + times[iterations / 2]) / 2;
    } else {
        // Odd number of elements: middle element
        median_time = times[iterations / 2];
    }

    LOGI_PROFILE("forward neconvlayer:  min: %ld us;   max: %ld us;   median: %ld us;   average: %lld us", min_time, max_time, median_time, average_time);
}

long profile_neconv_acl(){
    LOGI_PROFILE("-------------PROFILE NECONVOLUTIONLAYER------------------");
    arm_compute::Tensor input_tensor, kernel_tensor, bias_tensor, output_tensor;
    vectorToTensor(input_tensor, generateRandomVector(1000*1000*12), arm_compute::TensorShape(1000, 1000, 12), arm_compute::DataLayout::NCHW);
    vectorToTensor(kernel_tensor, generateRandomTensor(5*5*12*128), arm_compute::TensorShape(5,5,12,128), arm_compute::DataLayout::NCHW);
    vectorToTensor(bias_tensor, generateRandomVector(128), arm_compute::TensorShape(128));
    output_tensor.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(992, 992, 128),
                                                            1,
                                                            arm_compute::DataType::F32,
                                                            arm_compute::DataLayout::NCHW));
    output_tensor.allocator()->allocate();

    arm_compute::PadStrideInfo pad_stride_info(1, 1, 1,
                                               1, 1,
                                               1,
                                               arm_compute::DimensionRoundingType::FLOOR);
    arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::WeightsInfo weights_info(false, 5, 5, 128);
    arm_compute::ConvolutionInfo conv_info;
    arm_compute::NEConvolutionLayer convLayer;
    auto valid = convLayer.validate(input_tensor.info(), kernel_tensor.info(),
                                    bias_tensor.info(), output_tensor.info(),
                                    pad_stride_info, weights_info, arm_compute::Size2D(1,1),
                                    act_info, false, 1);
    LOGI_PROFILE("%s", valid.error_description().c_str());
    convLayer.configure(&input_tensor, &kernel_tensor, &bias_tensor,
                        &output_tensor, pad_stride_info, weights_info,
                        arm_compute::Size2D(1,1), act_info, false, 1);


    auto start = std::chrono::high_resolution_clock::now();
    convLayer.run();
    auto end = std::chrono::high_resolution_clock::now();

    //free acl internally managed resources
    input_tensor.allocator()->free();
    kernel_tensor.allocator()->free();
    bias_tensor.allocator()->free();
    output_tensor.allocator()->free();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

long profile_neconv_acl_small(){
    LOGI_PROFILE("-------------PROFILE NECONVOLUTIONLAYER SMALL------------------");
    arm_compute::Tensor input_tensor, kernel_tensor, bias_tensor, output_tensor;
    vectorToTensor(input_tensor, generateRandomVector(13*13*192), arm_compute::TensorShape(13, 13, 192), arm_compute::DataLayout::NCHW);
    vectorToTensor(kernel_tensor, generateRandomTensor(3*3*192*384), arm_compute::TensorShape(3, 3, 192, 384), arm_compute::DataLayout::NCHW);
    vectorToTensor(bias_tensor, generateRandomVector(384), arm_compute::TensorShape(384));
    output_tensor.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(13, 13, 384),
                                                            1,
                                                            arm_compute::DataType::F32,
                                                            arm_compute::DataLayout::NCHW));
    output_tensor.allocator()->allocate();

    arm_compute::PadStrideInfo pad_stride_info(1, 1, 1,
                                               1, 1,
                                               1,
                                               arm_compute::DimensionRoundingType::FLOOR);
    arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::WeightsInfo weights_info(false, 3, 3, 384);
    arm_compute::ConvolutionInfo conv_info;
    arm_compute::NEConvolutionLayer convLayer;
    auto valid = convLayer.validate(input_tensor.info(), kernel_tensor.info(),
                                    bias_tensor.info(), output_tensor.info(),
                                    pad_stride_info, weights_info, arm_compute::Size2D(1,1),
                                    act_info, false, 1);
    LOGI_PROFILE("%s", valid.error_description().c_str());
    convLayer.configure(&input_tensor, &kernel_tensor, &bias_tensor,
                        &output_tensor, pad_stride_info, weights_info,
                        arm_compute::Size2D(1,1), act_info, false, 1);


    auto start = std::chrono::high_resolution_clock::now();
    convLayer.run();
    auto end = std::chrono::high_resolution_clock::now();

    //free acl internally managed resources
    input_tensor.allocator()->free();
    kernel_tensor.allocator()->free();
    bias_tensor.allocator()->free();
    output_tensor.allocator()->free();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}