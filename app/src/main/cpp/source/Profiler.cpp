//
// Created by David Pauli (ti72teta) on 26.11.2023
//

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
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

int profile() {
    const size_t inputSize = 100;   // Adjust size as needed
    const size_t outputSize = 50;   // Adjust size as needed

    // Generate random weights and biases
    std::vector<std::vector<float>> weights(inputSize, generateRandomVector(outputSize));
    std::vector<float> biases = generateRandomVector(outputSize);

    // Generate a random input vector
    std::vector<float> input = generateRandomVector(inputSize);

    // Construct the layer
    FC::FullyConnected layer(weights, biases);

    // Time the execution of the forward method
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> output = layer.forward(input);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " ms" << std::endl;

    // Output some results for verification
    std::cout << "Output size: " << output.size() << std::endl;
    std::cout << "First few outputs: ";
    for (size_t i = 0; i < std::min(size_t(5), output.size()); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
