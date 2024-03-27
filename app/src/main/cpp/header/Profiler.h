//
// Created by drp on 13.03.24.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_PROFILER_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_PROFILER_H
#include "Helpers.h"
#include "Testing.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include <android/log.h>
#define LOG_TAG_PROFILE "NativeCode:Profile" // Tag for logging
#define LOGI_PROFILE(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_PROFILE, __VA_ARGS__)
void profile_acl();
long profile_neconv_acl();
long profile_neconv_acl_small();
#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_PROFILER_H
