//
// Created by David on 1/9/2024.
//
#include "../header/Helpers.h"

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


void vectorToTensor(arm_compute::Tensor& tensor, const std::vector<float>& data,
                    const arm_compute::TensorShape& shape,
                    const arm_compute::DataLayout data_layout) {
    // Initialize the tensor
    tensor.allocator()->init(arm_compute::TensorInfo(shape, 1,
                                                     arm_compute::DataType::F32,
                                                     data_layout));
    // Allocate memory for the tensor
    tensor.allocator()->allocate();
    // Copy data into the tensor
    std::copy(data.begin(), data.end(), reinterpret_cast<float*>(tensor.buffer()));
    //make sure the copy process was successful, tensorshape evaluates to actual data count etc.
    if (!assert_equal(tensor, data))
        LOGI_HELP("Tensor Conversion Error!");
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


// Function to get the indexes of the top 5 elements in a tensor
std::vector<size_t> find_top_five_indices(const arm_compute::Tensor *tensor) {
    // Check if tensor is not empty and is a float tensor
    if (!tensor->info()->tensor_shape().total_size() || tensor->info()->data_type() != arm_compute::DataType::F32) {
        return {};
    }

    // Flatten the tensor to a 1D vector
    size_t totalSize = tensor->info()->tensor_shape().total_size();
    std::vector<float> flattened(totalSize);
    std::copy(reinterpret_cast<float*>(tensor->buffer()),
              reinterpret_cast<float*>(tensor->buffer()) + totalSize,
              flattened.begin());

    return find_top_five_indices(flattened);
}


// Simple assertion function to compare vectors
bool assert_equal(const std::vector<float>& actual, const std::vector<float>& expected, float epsilon) {
    if (actual.size() != expected.size()) {
        LOGI_HELP("Test failed: Size mismatch actual: %zu, expected:%zu", actual.size(), expected.size());
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > epsilon) {
            LOGI_HELP("actual value: %f,  expected value: %f", actual[i], expected[i]);
            LOGI_HELP("Test failed: Value mismatch at index %zu", i);
            return false;
        }
    }

    return true;
}


// Simple assertion function to compare tensor and vector
bool assert_equal(const arm_compute::Tensor& actual, const std::vector<float>& expected, float epsilon) {
    const auto& tensor_info = actual.info();
    const arm_compute::TensorShape& shape = tensor_info->tensor_shape();
    const size_t num_elements = shape.total_size();

    if (num_elements != expected.size()) {
        LOGI_HELP("Test failed: Size mismatch expected: %zu, actual:%zu", expected.size(), num_elements);
        return false;
    }

    const float* tensor_data = reinterpret_cast<const float*>(actual.buffer());

    for (size_t i = 0; i < num_elements; ++i) {
        if (std::fabs(expected[i] - tensor_data[i]) > epsilon) {
            LOGI_HELP("expected value: %f, actual value: %f", expected[i], tensor_data[i]);
            LOGI_HELP("Test failed: Value mismatch at index %zu", i);
            return false;
        }
    }

    return true;
}


void printTensor(const arm_compute::Tensor& tensor) {
    // Ensure tensor is allocated
    LOGI_HELP("tensor info %s", getTensorInfo(tensor).c_str());
    if (!tensor.info()->is_resizable()) {
        const auto tensor_info = tensor.info();
        const auto shape = tensor_info->tensor_shape();
        const size_t total_elements = shape.total_size();

        // Assuming the tensor holds floats
        auto data_ptr = reinterpret_cast<const float*>(tensor.buffer());

        for (size_t i = 0; i < total_elements; ++i) {
            LOGI_HELP("element: %zu, value:    %f", i, data_ptr[i]);
        }
    } else {
        LOGI_HELP("tensor is not allocated");
    }
}


std::string getTensorInfo(const arm_compute::Tensor& tensor) {
    std::ostringstream oss;
    const arm_compute::ITensorInfo* tensor_info = tensor.info();

    if (tensor_info != nullptr) {
        // Get tensor dimensions
        const arm_compute::TensorShape& shape = tensor_info->tensor_shape();

        // Append each dimension to the string stream
        oss << "Tensor dimensions: ";
        for (size_t i = 0; i < shape.num_dimensions(); ++i) {
            oss << shape[i] << (i < shape.num_dimensions() - 1 ? " x " : "");
        }

        // Append total number of elements
        oss << "\nTotal number of elements: " << shape.total_size();

        // Append data type
        oss << "\nData type: ";
        switch (tensor_info->data_type()) {
            case arm_compute::DataType::F32: oss << "F32"; break;
            case arm_compute::DataType::F16: oss << "F16"; break;
            case arm_compute::DataType::QASYMM8: oss << "QASYMM8"; break;
                // ... handle other data types as needed
            default: oss << "Unknown";
        }

        // Append data layout
        oss << "\nData layout: ";
        switch (tensor_info->data_layout()) {
            case arm_compute::DataLayout::NCHW: oss << "NCHW"; break;
            case arm_compute::DataLayout::NHWC: oss << "NHWC"; break;
            default: oss << "Unknown";
        }

        //TODO: expand to float16 etc if necessary

        // Assuming the data is ready to be accessed and is of type F32 for this example
        // Note: Ensure that the tensor is allocated and the data is ready to be accessed
        auto data_ptr = reinterpret_cast<float*>(tensor.buffer());
        if (data_ptr != nullptr) {
            oss << "\nFirst 20 elements: ";
            for (size_t i = 0; i < std::min(static_cast<size_t>(20), shape.total_size()); ++i) {
                oss << data_ptr[i] << " ";
            }
        } else {
            oss << "\nUnable to access data buffer.";
        }

    } else {
        oss << "Tensor info is null.";
    }

    return oss.str();
}