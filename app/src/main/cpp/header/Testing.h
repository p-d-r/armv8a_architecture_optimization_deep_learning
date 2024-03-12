//
// Created by David on 1/9/2024.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_TESTING_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_TESTING_H

#include <vector>
#include <cmath>
#include "FullyConnected.h"
#include "Pooling.h"
#include "Convolution.h"
#include "Helpers.h"
#include "AssetManagement.h"
#include <iostream>
#include <android/log.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Coordinates.h>
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#define LOG_TAG_TEST "NativeCode:Testing" // Tag for logging
#define LOGI_TEST(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_TEST, __VA_ARGS__)

void test_fully_connected();
void test_pooling_acl();
void test_convolution_acl();
void test_alexnet_torch_nchw();
void test_alexnet_torch_nhwc();

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_TESTING_H
