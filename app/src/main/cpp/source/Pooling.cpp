//
// Created by David Pauli (ti72teta) on 26.11.2023
//
#include "../header/Pooling.h"

namespace POOL {
    Pooling::Pooling(int pool_size)
            : pool_size(pool_size){}


    std::vector<std::vector<std::vector<float>>>
            Pooling::forward(const std::vector<std::vector<std::vector<float>>>& input) {
        size_t channels = input.size();
        size_t height = input[0].size();
        size_t width = input[0][0].size();

        // Calculate output dimensions
        size_t outHeight = height / pool_size;
        size_t outWidth = width / pool_size;

        std::vector<std::vector<std::vector<float>>>
                output(channels,
                       std::vector<std::vector<float>>(outHeight,
                                                          std::vector<float>(outWidth, 0)));

        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < outHeight; ++y) {
                for (int x = 0; x < outWidth; ++x) {
                    float maxVal = -FLT_MAX;
                    for (int dy = 0; dy < pool_size; ++dy) {
                        for (int dx = 0; dx < pool_size; ++dx) {
                            int iy = y * pool_size + dy;
                            int ix = x * pool_size + dx;
                            maxVal = std::max(maxVal, input[c][iy][ix]);
                        }
                    }
                    output[c][y][x] = maxVal;
                }
            }
        }

        return output;
    }


    std::vector<std::vector<std::vector<float>>>
            Pooling::forward_tiled(const std::vector<std::vector<std::vector<float>>> &input,
                                   size_t tile_size) {
        size_t channels = input.size();
        size_t height = input[0].size();
        size_t width = input[0][0].size();
        int c, tx, ty, y, x, dy, dx;

        // Calculate output dimensions
        size_t outHeight = height / pool_size;
        size_t outWidth = width / pool_size;

        std::vector<std::vector<std::vector<float>>>
                output(channels,
                       std::vector<std::vector<float>>(outHeight,
                                                          std::vector<float>(outWidth, 0)));

        for (c = 0; c < channels; ++c) {
            // Loop over tiles
            for (ty = 0; ty < outHeight; ty += tile_size) {
                for (tx = 0; tx < outWidth; tx += tile_size) {
                    // Loop inside each tile
                    for (y = ty; y < ty + tile_size; ++y) {
                        for (x = tx; x < tx + tile_size; ++x) {
                            float maxVal = -FLT_MAX;
                            for (dy = 0; dy < tile_size; ++dy) {
                                for (dx = 0; dx < tile_size; ++dx) {
                                    int iy = y * tile_size + dy;
                                    int ix = x * tile_size + dx;
                                    maxVal = std::max(maxVal, input[c][iy][ix]);
                                }
                            }
                            output[c][y][x] = maxVal;
                        }
                    }
                }
            }
        }

        return output;
    }
} // POOL