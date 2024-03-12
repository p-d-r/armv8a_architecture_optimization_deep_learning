//
// Created by drp on 07.03.24.
//
#include "../header/AssetManagement.h"

//AAssetManager itself does not require explicit deletion, since its managed by the android runtime
//Synchronization primitives
AAssetManager* assetManager = nullptr;

void set_up_asset_manager(JNIEnv *env, jobject java_asset_manager){
    assetManager = AAssetManager_fromJava(env, java_asset_manager);
}

std::vector<float> read_binary_float_vector_asset(const std::string& filename) {
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
        LOGI_ASS("Asset size is not a multiple of float size");
        AAsset_close(asset);
        return {};
    }

    std::vector<float> floatData(numFloats);
    AAsset_read(asset, floatData.data(), size);
    AAsset_close(asset);

    return std::move(floatData);
}