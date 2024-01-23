//
// Created by David on 1/9/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_TESTING_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_TESTING_H

#include <vector>
#include <cmath>
#include <android/log.h>
#include "FullyConnected.h"
#include "Pooling.h"
#include "Convolution.h"

#define LOG_TAG "NativeCode:Testing" // Tag for logging
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

void test_fully_connected();
void test_pooling();
void test_convolution();
void assert_equal(const std::vector<float>& expected, const std::vector<float>& actual, float epsilon = 0.001f);

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_TESTING_H
