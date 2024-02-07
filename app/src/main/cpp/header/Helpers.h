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
#include "arm_compute/runtime/Tensor.h"

std::vector<float> generateRandomTensor(size_t size);
std::vector<size_t> find_top_five_indices(const std::vector<float>& values);
std::vector<size_t> find_top_five_indices(const arm_compute::Tensor *tensor);
void vectorToTensor(arm_compute::Tensor& tensor, const std::vector<float>& data, const arm_compute::TensorShape& shape);
bool loadFromBinary(const std::string& filename, std::vector<float>& data);

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_HELPERS_H
