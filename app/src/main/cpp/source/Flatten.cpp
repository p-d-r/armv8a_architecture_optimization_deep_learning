//
// Created by drp on 06.02.24.
//

#include "../header/Flatten.h"

namespace CNN {
    Flatten::Flatten(std::shared_ptr<arm_compute::Tensor> input_tensor, std::shared_ptr<arm_compute::Tensor> output_tensor) :
            input_tensor(input_tensor), output_tensor(output_tensor){}

    void Flatten::forward_acl() {
        const size_t totalElements = input_tensor->info()->tensor_shape().total_size();
        const auto* inputDataPtr = reinterpret_cast<const float*>(input_tensor->buffer());
        auto* outputDataPtr = reinterpret_cast<float*>(output_tensor->buffer());

        // Copy the input tensor data to the output tensor
        std::copy(inputDataPtr, inputDataPtr + totalElements, outputDataPtr);
    }
} // CNN