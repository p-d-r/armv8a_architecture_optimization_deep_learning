//
// Created by drp on 31.01.24.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CUSTOMACCESSORS_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CUSTOMACCESSORS_H

#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/graph.h"

using namespace arm_compute;

class VectorAccessor : public graph::ITensorAccessor {
public:
    VectorAccessor(std::vector<float>& data, const TensorShape& shape)
            : _data(data), _shape(shape) {}

    bool access_tensor(ITensor& tensor) override {
        ARM_COMPUTE_ERROR_ON(tensor.info()->tensor_shape().total_size() != _data.size());

        // Assuming the data layout is NCHW and the data type is F32
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());

        Iterator it(&tensor, window);
        execute_window_loop(window, [&](const Coordinates& ) {
            std::copy_n(_data.data(), _data.size(), reinterpret_cast<float*>(it.ptr()));
        });

        return true;
    }

private:
    std::vector<float>& _data;
    TensorShape _shape;
};


class ArrayAccessor : public graph::ITensorAccessor {
public:
    ArrayAccessor(const TensorShape& shape)
            : _data(nullptr), _size(0), _shape(shape) {}

    void update_data(float* data, size_t size) {
        _data = data;
        _size = size;
    }

    bool access_tensor(ITensor& tensor) override {
        ARM_COMPUTE_ERROR_ON(_data == nullptr);
        ARM_COMPUTE_ERROR_ON(tensor.info()->tensor_shape().total_size() > _size);

        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());

        Iterator it(&tensor, window);
        execute_window_loop(window, [&](const Coordinates& ) {
            std::copy_n(_data, _size, reinterpret_cast<float*>(it.ptr()));
        });

        return true;
    }

private:
    float* _data;
    size_t _size;
    TensorShape _shape;
};


#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_CUSTOMACCESSORS_H
