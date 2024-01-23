//
// Created by David on 1/9/2024.
//
#include "../header/Helpers.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>

// Function to generate a random vector
std::vector<float> generateRandomTensor(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 10.0);

    std::vector<float> v(size);
    for (auto& element : v) {
        element = dis(gen);
    }
    return v;
}


bool loadFromBinary(const std::string& filename, std::vector<float>& data) {
    std::ifstream input(filename, std::ios::binary);

    // Files can be large, so we check if it was opened successfully.
    if (!input.is_open()) {
        std::cerr << "Unable to open the file: " << filename << std::endl;
        return false;
    }

    // Get the size of the file
    input.seekg(0, std::ios::end);
    size_t size = input.tellg();
    input.seekg(0, std::ios::beg);

    // Resize the vector to hold all the data
    data.resize(size / sizeof(float));

    // Read the data all at once
    input.read(reinterpret_cast<char*>(data.data()), size);

    // Check for reading error
    if (!input) {
        std::cerr << "Error occurred while reading from the file: " << filename << std::endl;
        input.close();
        return false;
    }

    input.close();
    return true;
}


std::vector<size_t> find_top_five_indices(const std::vector<float>& values) {
    std::vector<size_t> indices(values.size());
    // Initialize indices to 0, 1, 2, ..., n-1
    std::iota(indices.begin(), indices.end(), 0);

    // Partially sort the first 5 indices based on the values they map to
    std::nth_element(indices.begin(), indices.begin() + 5, indices.end(),
                     [&values](size_t a, size_t b) { return values[a] > values[b]; });

    // Resize the vector to only contain the top 5 indices
    indices.resize(5);

    // Sort the top 5 indices for clearer output
    std::sort(indices.begin(), indices.end(),
              [&values](size_t a, size_t b) { return values[a] > values[b]; });

    return indices;
}
