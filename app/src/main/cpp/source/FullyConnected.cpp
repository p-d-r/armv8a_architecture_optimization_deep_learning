#include "../header/FullyConnected.h"

namespace CNN {

    FullyConnected::FullyConnected(const std::vector<float> &weights,
                                   const std::vector<float> &bias,
                                   size_t input_size,
                                   size_t output_size,
                                   int tile_size)
                    : weights(weights),
                      bias(bias),
                      input_size(input_size),
                      output_size(output_size),
                      tile_size(tile_size) {}


    FullyConnected::FullyConnected(std::unique_ptr<arm_compute::Tensor> weights_tensor,
                                   std::unique_ptr<arm_compute::Tensor> bias_tensor,
                                   size_t input_size, size_t output_size, int tile_size,
                                   arm_compute::ActivationLayerInfo::ActivationFunction activation_function)
            : input_size(input_size),
              output_size(output_size),
              tile_size(tile_size),
              weights_tensor(std::move(weights_tensor)),
              bias_tensor(std::move(bias_tensor)),
              activation_function(activation_function) {}


    std::vector<float> FullyConnected::forward(const std::vector<float> &batch_input) {
        size_t num_batches = batch_input.size() / (input_size);
        std::vector<float> batch_output(num_batches * output_size);
        float sum;
        size_t b, j, i;

        for (i = 0; i < this->output_size; ++i) {
            for (b = 0; b < num_batches; ++b) {
                sum = this->bias[i];
                for (j = 0; j < this->input_size; ++j) {
                    sum += batch_input[j * num_batches + b] * this->weights[i * (this->input_size) + j];
                }
                batch_output[i * num_batches + b] = std::max(sum, 0.0f);
            }
        }

        return batch_output;
    }

    std::vector<float> FullyConnected::forward_transposed_in(const std::vector<float> & batch_input_transposed) {
        size_t num_batches = batch_input_transposed.size() / (input_size);
        std::vector<float> transposed_batch_output(num_batches * output_size);
        float sum;
        size_t b, j, i;

        for (b = 0; b < num_batches; ++b) {
            for (i = 0; i < output_size; ++i) {
                sum = this->bias[i];
                for (j = 0; j < input_size; ++j) {
                    sum += batch_input_transposed[j * num_batches + b] * weights[i * output_size + j];
                }

                transposed_batch_output[i * num_batches + b] = sum;
            }
        }

        return transposed_batch_output;
    }


    std::vector<float> FullyConnected::forward_tiled(const std::vector<float> & batch_input) {
        size_t num_batches = batch_input.size() / (input_size);
        std::vector<float> batch_output(num_batches * output_size);
        float sum;
        size_t b, i, j, ii, jj;

        for (b = 0; b < num_batches; ++b) {
            for (i = 0; i < output_size; i += tile_size) {
                for (j = 0; j < input_size; j += tile_size) { // Iterate up to the last row (bias row)
                    for (ii = i; ii < i + tile_size && ii < output_size; ++ii) {
                        for (jj = j; jj < j + tile_size && jj < input_size; ++jj) {
                            batch_output[b * output_size + ii] += batch_input[b * input_size + jj] * weights[ii * output_size + jj];
                        }
                    }
                }
            }
        }

        return batch_output;
    }



    std::vector<float> FullyConnected::forward_tiled_transposed_in(const std::vector<float> & transposed_input) {
        size_t num_batches = transposed_input.size() / input_size; // Assuming transposed_input.size() is batch_size * input_size
        std::vector<float> transposed_batch_output(num_batches * output_size, 0.0f); // Initialize with zeros
        float sum;
        size_t b, j, i, jj, ii;

        for (b = 0; b < num_batches; ++b) {
            for (i = 0; i < output_size; i += tile_size) {
                for (j = 0; j < input_size; j += tile_size) {
                    for (ii = i; ii < std::min(i + tile_size, output_size); ++ii) {
                        sum = weights[ii * output_size + input_size]; // Initialize with bias
                        for (jj = j; jj < std::min(j + tile_size, input_size); ++jj) {
                            // Accessing the input in a transposed manner
                            sum += transposed_input[jj * num_batches + b] * weights[ii * output_size + jj];
                        }

                        transposed_batch_output[ii * num_batches + b] += sum;
                    }
                }
            }
        }

        return transposed_batch_output;
    }

    void FullyConnected::setWeights(const std::vector<float> &weights) {
        this->weights = weights;
    }

    void FullyConnected::setBias(const std::vector<float> &bias) {
        this->bias = bias;
    }


    // ARM Compute Library functions
    void FullyConnected::forward_acl() {
        fc_layer.run();
    }

    void FullyConnected::configure_acl() {
        // Set up FullyConnectedLayerInfo
        arm_compute::FullyConnectedLayerInfo fc_info;
        fc_info.set_weights_trained_layout(arm_compute::DataLayout::NCHW);
        fc_info.set_transpose_weights(true);
        arm_compute::ActivationLayerInfo act_info(this->activation_function, 1, 0);
        fc_info.activation_info = act_info;
        auto valid = fc_layer.validate(this->input_tensor.get()->info(),
                          this->weights_tensor.get()->info(),
                          this->bias_tensor.get()->info(),
                          this->output_tensor.get()->info(),
                          fc_info);
        LOGI_LAYER("%s", valid.error_description().c_str());
        fc_layer.configure(this->input_tensor.get(),
                           this->weights_tensor.get(),
                           this->bias_tensor.get(),
                           this->output_tensor.get(),
                           fc_info);
    }
} // namespace CNN
