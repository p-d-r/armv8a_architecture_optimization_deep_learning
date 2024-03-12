//
// Created by drp on 07.03.24.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_ASSETMANAGEMENT_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_ASSETMANAGEMENT_H
#include <jni.h>
#include <vector>
#include <iostream>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#define LOG_TAG "NativeCode:AssetManagement" // Tag for logging
#define LOGI_ASS(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

void set_up_asset_manager(JNIEnv *env, jobject java_asset_manager);
std::vector<float> read_binary_float_vector_asset(const std::string& filename);
#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_ASSETMANAGEMENT_H
