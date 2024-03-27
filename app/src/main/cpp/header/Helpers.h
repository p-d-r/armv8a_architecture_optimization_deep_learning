//
// Created by David on 1/9/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_HELPERS_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_HELPERS_H
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <android/log.h>
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#define LOG_TAG_HELP "NativeCode:Helpers" // Tag for logging
#define LOGI_HELP(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_HELP, __VA_ARGS__)

std::vector<float> generateRandomTensor(size_t size);
std::vector<size_t> find_top_five_indices(const std::vector<float>& values);
std::vector<size_t> find_top_five_indices(const arm_compute::Tensor *tensor);
void vectorToTensor(arm_compute::Tensor& tensor, const std::vector<float>& data,
                    const arm_compute::TensorShape& shape,
                    const arm_compute::DataLayout data_layout=arm_compute::DataLayout::NCHW);
bool assert_equal(const std::vector<float>& actual, const std::vector<float>& expected, float epsilon = 0.0001f);
bool assert_equal(const arm_compute::Tensor& actual, const std::vector<float>& expected, float epsilon = 0.0001f);
void printTensor(const arm_compute::Tensor& tensor);
std::string getTensorInfo(const arm_compute::Tensor& tensor);

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_HELPERS_H
