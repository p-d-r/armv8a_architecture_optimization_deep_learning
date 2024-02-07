//
// Created by David on 1/9/2024.
//

#include "../header/LRN.h"

namespace LRN {
    LRN::LRN(size_t depth, size_t height, size_t width, size_t local_size, double alpha, float beta,
             float k, std::shared_ptr<arm_compute::Tensor> input_tensor, std::shared_ptr<arm_compute::Tensor> output_tensor):
            depth(depth), height(height), width(width), local_size(local_size), alpha(alpha), beta(beta), k(k),
            input_tensor(input_tensor), output_tensor(output_tensor){

        if (input_tensor != nullptr) {
            arm_compute::NormalizationLayerInfo norm_info(
                    arm_compute::NormType::CROSS_MAP, // LRN type
                    this->local_size,             // normalization size
                    this->alpha,                           // alpha
                    this->beta,                            // beta
                    this->k                          // kappa
            );

            lrn_layer.configure(input_tensor.get(), output_tensor.get(), norm_info);
        }
    }

    std::vector<float> LRN::forward(const std::vector<float> &input) {
        std::vector<float> output(input.size(), 0.0f);
        int half_local_size = static_cast<int>(local_size / 2);

        for (size_t d = 0; d < depth; ++d) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t idx = (d * height + h) * width + w;
                    float sum = 0.0f;

                    // Sum over the local window
                    for (int j = -half_local_size; j <= half_local_size; ++j) {
                        if (static_cast<int>(d) + j >= 0 && static_cast<int>(d) + j < static_cast<int>(depth)) {
                            size_t local_idx = ((d + j) * height + h) * width + w;
                            sum += std::pow(input[local_idx], 2);
                        }
                    }

                    // Apply normalization formula outside the j loop
                    output[idx] = input[idx] * std::pow(k + alpha * sum, -beta);
                }
            }
        }

        return output;
    }

    void LRN::forward_acl() {
        this->lrn_layer.run();
    }


} // LRN