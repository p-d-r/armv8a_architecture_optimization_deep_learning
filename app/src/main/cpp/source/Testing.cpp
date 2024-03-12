//
// Created by David on 1/9/2024.
//
#include "../header/Testing.h"
#include "../header/Network.h"


// Test function for FullyConnected class
void test_fully_connected() {

}


void test_pooling_acl() {
    LOGI_TEST("-------------TEST MAXPOOLING NCHW------------------");
    arm_compute::Tensor input_tensor, output_tensor;
    vectorToTensor(input_tensor, read_binary_float_vector_asset("weights/vgg16/layer_outputs/layer_3.bin"), arm_compute::TensorShape(224, 224, 64, 1), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST input info %s", getTensorInfo(input_tensor).c_str());
    output_tensor.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(112, 112, 64, 1),
                                                            1,
                                                            arm_compute::DataType::F32,
                                                            arm_compute::DataLayout::NCHW));
    output_tensor.allocator()->allocate();

    // Define the pooling layer parameters
    arm_compute::PoolingLayerInfo pool_info(arm_compute::PoolingType::MAX,
                                            arm_compute::Size2D(2, 2),
                                            arm_compute::DataLayout::NCHW,
                                            arm_compute::PadStrideInfo(2,
                                                                       2,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       arm_compute::DimensionRoundingType::FLOOR));

    arm_compute::NEPoolingLayer pool_layer;
    auto valid = pool_layer.validate(input_tensor.info(), output_tensor.info(), pool_info);
    LOGI_TEST("%s", valid.error_description().c_str());
    pool_layer.configure(&input_tensor, &output_tensor, pool_info);
    pool_layer.run();
    auto expected = read_binary_float_vector_asset("weights/vgg16/layer_outputs/layer_4.bin");
    LOGI_TEST("TEST output info %s", getTensorInfo(output_tensor).c_str());
    //printTensor(output_tensor);
    if (assert_equal(output_tensor, expected)) {
        LOGI_TEST("TEST MAX POOLING NCHW WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST MAX POOLING NCHW FAILED");
    }

    LOGI_TEST("-------------TEST MAXPOOLING NHWC------------------");
    arm_compute::Tensor input_tensor_nhwc, output_tensor_nhwc;
    vectorToTensor(input_tensor_nhwc, read_binary_float_vector_asset("OHWI/outputs/layer_3.bin"), arm_compute::TensorShape(64, 224, 224, 1), arm_compute::DataLayout::NHWC);
    LOGI_TEST("TEST input info %s", getTensorInfo(input_tensor_nhwc).c_str());
    output_tensor_nhwc.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(64, 112, 112, 1),
                                                            1,
                                                            arm_compute::DataType::F32,
                                                            arm_compute::DataLayout::NHWC));
    output_tensor_nhwc.allocator()->allocate();

    // Define the pooling layer parameters
    arm_compute::PoolingLayerInfo pool_info_nhwc(arm_compute::PoolingType::MAX,
                                            arm_compute::Size2D(2, 2),
                                            arm_compute::DataLayout::NHWC,
                                            arm_compute::PadStrideInfo(2,
                                                                       2,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       0,
                                                                       arm_compute::DimensionRoundingType::FLOOR));

    arm_compute::NEPoolingLayer pool_layer_nhwc;
    auto valid_nhwc = pool_layer_nhwc.validate(input_tensor_nhwc.info(), output_tensor_nhwc.info(), pool_info_nhwc);
    LOGI_TEST("%s", valid_nhwc.error_description().c_str());
    pool_layer_nhwc.configure(&input_tensor_nhwc, &output_tensor_nhwc, pool_info_nhwc);
    pool_layer_nhwc.run();
    auto expected_nhwc = read_binary_float_vector_asset("OHWI/outputs/layer_4.bin");
    LOGI_TEST("TEST output info %s", getTensorInfo(output_tensor_nhwc).c_str());
    //printTensor(output_tensor);
    if (assert_equal(output_tensor_nhwc, expected_nhwc)) {
        LOGI_TEST("TEST MAX POOLING NHWC WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST MAX POOLING NHWC FAILED");
    }

    //free acl internally managed resources
    input_tensor_nhwc.allocator()->free();
    output_tensor_nhwc.allocator()->free();
}


//The test samples are exported inputs / outputs from vgg16 pytorch convolutions;
void test_convolution_acl() {
    LOGI_TEST("-------------TEST CONVOLUTION NCHW------------------");
    arm_compute::Tensor input_tensor, kernel_tensor, bias_tensor, output_tensor;
    vectorToTensor(input_tensor, read_binary_float_vector_asset("weights/vgg16/layer_outputs/layer_1.bin"), arm_compute::TensorShape(224, 224, 64, 1), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST input info %s", getTensorInfo(input_tensor).c_str());
    vectorToTensor(kernel_tensor, read_binary_float_vector_asset("weights/vgg16/weights/features_1_weight.bin"), arm_compute::TensorShape(3,3,64,64), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST conv kernel info %s", getTensorInfo(kernel_tensor).c_str());
    vectorToTensor(bias_tensor,read_binary_float_vector_asset("weights/vgg16/weights/features_1_bias.bin"), arm_compute::TensorShape(64));
    LOGI_TEST("TEST conv bias info %s", getTensorInfo(bias_tensor).c_str());
    output_tensor.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(224, 224, 64, 1),
                                                            1,
                                                            arm_compute::DataType::F32,
                                                            arm_compute::DataLayout::NCHW));
    output_tensor.allocator()->allocate();

    arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::WeightsInfo weights_info;
    // Convolution layer info, including stride, padding, and activation info
    arm_compute::PadStrideInfo pad_stride_info(1, 1, 1, 1);
    arm_compute::NEConvolutionLayer convLayer;
    auto valid = convLayer.validate(input_tensor.info(), kernel_tensor.info(),
                       bias_tensor.info(), output_tensor.info(),
                       pad_stride_info, weights_info, arm_compute::Size2D(1,1),
                       act_info, false, 1);
    LOGI_TEST("%s", valid.error_description().c_str());
    convLayer.configure(&input_tensor, &kernel_tensor, &bias_tensor,
                        &output_tensor, pad_stride_info, weights_info,
                        arm_compute::Size2D(1,1), act_info, false, 1);

    // Run the convolution
    convLayer.run();

    auto expected = read_binary_float_vector_asset("weights/vgg16/layer_outputs/layer_2.bin");
    LOGI_TEST("TEST output info %s", getTensorInfo(output_tensor).c_str());
    //printTensor(output_tensor);
    if (assert_equal(output_tensor, expected)) {
        LOGI_TEST("TEST CONVOLUTION WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST CONVOLUTION FAILED");
    }

    //free acl internally managed resources
    input_tensor.allocator()->free();
    kernel_tensor.allocator()->free();
    bias_tensor.allocator()->free();
    output_tensor.allocator()->free();

    LOGI_TEST("-------------TEST CONVOLUTION NHWC------------------");
    arm_compute::Tensor input_tensor_nhwc, kernel_tensor_nhwc, bias_tensor_nhwc, output_tensor_nhwc;
    vectorToTensor(input_tensor_nhwc, read_binary_float_vector_asset("OHWI/outputs/layer_1.bin"), arm_compute::TensorShape(64, 224, 224, 1), arm_compute::DataLayout::NHWC);
    LOGI_TEST("TEST input info %s", getTensorInfo(input_tensor_nhwc).c_str());
    vectorToTensor(kernel_tensor_nhwc, read_binary_float_vector_asset("OHWI/conv_layer_1_weights_iwho.bin"), arm_compute::TensorShape(64, 3, 3, 64), arm_compute::DataLayout::NHWC);
    LOGI_TEST("TEST conv kernel info %s", getTensorInfo(kernel_tensor_nhwc).c_str());
    vectorToTensor(bias_tensor_nhwc,read_binary_float_vector_asset("weights/vgg16/weights/features_1_bias.bin"), arm_compute::TensorShape(64), arm_compute::DataLayout::NHWC);
    LOGI_TEST("TEST conv bias info %s", getTensorInfo(bias_tensor_nhwc).c_str());
    output_tensor_nhwc.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(64, 224, 224, 1),
                                                            1,
                                                            arm_compute::DataType::F32,
                                                            arm_compute::DataLayout::NHWC));
    output_tensor_nhwc.allocator()->allocate();

    arm_compute::ActivationLayerInfo act_info_nhwc(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::WeightsInfo weights_info_nhwc(false, 3, 3, 64, false, arm_compute::WeightFormat::UNSPECIFIED);
    // Convolution layer info, including stride, padding, and activation info
    arm_compute::PadStrideInfo pad_stride_info_nhwc(1, 1, 1, 1);
    arm_compute::NEConvolutionLayer convLayer_nhwc;
    auto valid_nhwc = convLayer_nhwc.validate(input_tensor_nhwc.info(), kernel_tensor_nhwc.info(),
                                    bias_tensor_nhwc.info(), output_tensor_nhwc.info(),
                                    pad_stride_info_nhwc, weights_info_nhwc, arm_compute::Size2D(1,1),
                                    act_info_nhwc, false, 1);
    LOGI_TEST("%s", valid_nhwc.error_description().c_str());
    convLayer_nhwc.configure(&input_tensor_nhwc, &kernel_tensor_nhwc, &bias_tensor_nhwc,
                        &output_tensor_nhwc, pad_stride_info_nhwc, weights_info_nhwc,
                        arm_compute::Size2D(1,1), act_info_nhwc, false, 1);

    // Run the convolution
    convLayer_nhwc.run();

    auto expected_nhwc = read_binary_float_vector_asset("OHWI/outputs/layer_3.bin");
    LOGI_TEST("TEST output info %s", getTensorInfo(output_tensor_nhwc).c_str());
    //printTensor(output_tensor);
    if (assert_equal(output_tensor_nhwc, expected_nhwc)) {
        LOGI_TEST("TEST CONVOLUTION WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST CONVOLUTION FAILED");
    }

    //free acl internally managed resources
    input_tensor_nhwc.allocator()->free();
    kernel_tensor_nhwc.allocator()->free();
    bias_tensor_nhwc.allocator()->free();
    output_tensor_nhwc.allocator()->free();


    LOGI_TEST("-------------TEST CONVOLUTION ALEXNET NCHW------------------");
    arm_compute::Tensor input_tensor_anchw, kernel_tensor_anchw, bias_tensor_anchw, output_tensor_anchw;
    vectorToTensor(input_tensor_anchw, read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_2.bin"), arm_compute::TensorShape(27, 27, 64, 1), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST input info %s", getTensorInfo(input_tensor_anchw).c_str());
    vectorToTensor(kernel_tensor_anchw, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_1_weights_nchw.bin"), arm_compute::TensorShape(5, 5, 64, 192), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST conv kernel info %s", getTensorInfo(kernel_tensor_anchw).c_str());
    vectorToTensor(bias_tensor_anchw,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_1_bias.bin"), arm_compute::TensorShape(192), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST conv bias info %s", getTensorInfo(bias_tensor_anchw).c_str());
    output_tensor_anchw.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(27, 27, 192, 1),
                                                                 1,
                                                                 arm_compute::DataType::F32,
                                                                 arm_compute::DataLayout::NCHW));
    output_tensor_anchw.allocator()->allocate();

    arm_compute::ActivationLayerInfo act_info_anchw(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::WeightsInfo weights_info_anchw(false, 5, 5, 192, false, arm_compute::WeightFormat::UNSPECIFIED);
    // Convolution layer info, including stride, padding, and activation info
    arm_compute::PadStrideInfo pad_stride_info_anchw(1, 1, 2, 2);
    arm_compute::NEConvolutionLayer convLayer_anchw;
    auto valid_anchw = convLayer_anchw.validate(input_tensor_anchw.info(), kernel_tensor_anchw.info(),
                                              bias_tensor_anchw.info(), output_tensor_anchw.info(),
                                              pad_stride_info_anchw, weights_info_anchw, arm_compute::Size2D(1,1),
                                              act_info_nhwc, false, 1);
    LOGI_TEST("%s", valid_anchw.error_description().c_str());
    convLayer_anchw.configure(&input_tensor_anchw, &kernel_tensor_anchw, &bias_tensor_anchw,
                             &output_tensor_anchw, pad_stride_info_anchw, weights_info_anchw,
                             arm_compute::Size2D(1,1), act_info_anchw, false, 1);

    // Run the convolution
    convLayer_anchw.run();

    auto expected_anchw= read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_3.bin");
    LOGI_TEST("TEST output info %s", getTensorInfo(output_tensor_anchw).c_str());
    //printTensor(output_tensor);
    if (assert_equal(output_tensor_anchw, expected_anchw)) {
        LOGI_TEST("TEST CONVOLUTION WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST CONVOLUTION FAILED");
    }

    //free acl internally managed resources
    input_tensor_anchw.allocator()->free();
    kernel_tensor_anchw.allocator()->free();
    bias_tensor_anchw.allocator()->free();
    output_tensor_anchw.allocator()->free();


    LOGI_TEST("-------------TEST CONVOLUTION ALEXNET 3x3------------------");
    arm_compute::Tensor input_tensor_3x3, kernel_tensor_3x3, bias_tensor_3x3, output_tensor_3x3;
    vectorToTensor(input_tensor_3x3, read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_5.bin"), arm_compute::TensorShape(13, 13, 192, 1), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST input info %s", getTensorInfo(input_tensor_3x3).c_str());
    vectorToTensor(kernel_tensor_3x3, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_2_weights_nchw.bin"), arm_compute::TensorShape(3, 3, 192, 384), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST conv kernel info %s", getTensorInfo(kernel_tensor_3x3).c_str());
    vectorToTensor(bias_tensor_3x3,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_2_bias.bin"), arm_compute::TensorShape(384), arm_compute::DataLayout::NCHW);
    LOGI_TEST("TEST conv bias info %s", getTensorInfo(bias_tensor_3x3).c_str());
    output_tensor_3x3.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(13, 13, 384, 1),
                                                                  1,
                                                                  arm_compute::DataType::F32,
                                                                  arm_compute::DataLayout::NCHW));
    output_tensor_3x3.allocator()->allocate();

    arm_compute::ActivationLayerInfo act_info_3x3(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::WeightsInfo weights_info_3x3(false, 3, 3, 384, false, arm_compute::WeightFormat::UNSPECIFIED);
    // Convolution layer info, including stride, padding, and activation info
    arm_compute::PadStrideInfo pad_stride_info_3x3(1, 1, 1, 1);
    arm_compute::NEConvolutionLayer convLayer_3x3;
    auto valid_3x3 = convLayer_3x3.validate(input_tensor_3x3.info(), kernel_tensor_3x3.info(),
                                                bias_tensor_3x3.info(), output_tensor_3x3.info(),
                                                pad_stride_info_3x3, weights_info_3x3, arm_compute::Size2D(1,1),
                                                act_info_3x3, false, 1);
    LOGI_TEST("%s", valid_3x3.error_description().c_str());
    convLayer_3x3.configure(&input_tensor_3x3, &kernel_tensor_3x3, &bias_tensor_3x3,
                              &output_tensor_3x3, pad_stride_info_3x3, weights_info_3x3,
                              arm_compute::Size2D(1,1), act_info_3x3, false, 1);

    // Run the convolution
    convLayer_3x3.run();

    auto expected_3x3= read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_6.bin");
    LOGI_TEST("TEST output info %s", getTensorInfo(output_tensor_3x3).c_str());
    //printTensor(output_tensor);
    if (assert_equal(output_tensor_3x3, expected_3x3)) {
        LOGI_TEST("TEST CONVOLUTION WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST CONVOLUTION FAILED");
    }

    //free acl internally managed resources
    input_tensor_3x3.allocator()->free();
    kernel_tensor_3x3.allocator()->free();
    bias_tensor_3x3.allocator()->free();
    output_tensor_3x3.allocator()->free();
}

/*
 * Function loads an Imagenet (dog) image and compares the alexnet NCHW ACL implementation
 * with the output from pytorch; Here, NCHW is used as DataLayout
 * The accuracy is greater than 0.00001f for every element of every intermediate result
 */
void test_alexnet_torch_nchw() {
    CNN::Network alexnet(arm_compute::DataLayout::NCHW);
    LOGI_TEST("-------------TEST ALEXNET TORCH NCHW------------------");
    auto conv0_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv0_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv0_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_0_weights_nchw.bin"), arm_compute::TensorShape(11, 11, 3, 64), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv0_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_0_bias.bin"), arm_compute::TensorShape(64), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv0 = new CNN::Convolution(3, 64, 11, 11, 224, 224, 4, 2, 1, std::move(conv0_kernel), std::move(conv0_bias));
    alexnet.addLayer(conv0, arm_compute::TensorShape(224, 224, 3), arm_compute::TensorShape(55, 55, 64));
    conv0->configure_acl();

    //copy input image to input tensor of network
    arm_compute::Tensor *input_tensor = alexnet.input_tensor.get();
    auto dog_vector = read_binary_float_vector_asset("flattened_dog.bin");
    std::copy(dog_vector.begin(), dog_vector.end(), reinterpret_cast<float*>(input_tensor->buffer()));
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_1.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 0 CONV0  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 0 CONV0 FAILED");
    }

    auto *pool0 = new CNN::Pooling(3, 3, 64, 55, 55, 2, 0,0,0,0);
    alexnet.addLayer(pool0, arm_compute::TensorShape(55, 55, 64), arm_compute::TensorShape(27, 27, 64));
    pool0->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_2.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 1 POOL0  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 1 POOL0 FAILED");
    }

    auto conv1_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv1_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv1_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_1_weights_nchw.bin"), arm_compute::TensorShape(5, 5, 64, 192), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv1_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_1_bias.bin"), arm_compute::TensorShape(192), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv1 = new CNN::Convolution(64, 192, 5, 5, 27, 27, 1, 2, 1, std::move(conv1_kernel), std::move(conv1_bias));
    alexnet.addLayer(conv1, arm_compute::TensorShape(27, 27, 64), arm_compute::TensorShape(27, 27, 192));
    conv1->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_4.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 3 CONV1  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 3 CONV1 FAILED");
    }

    auto *pool1 = new CNN::Pooling(3, 3, 192, 27, 27, 2, 0,0,0,0);
    alexnet.addLayer(pool1, arm_compute::TensorShape(27, 27, 192), arm_compute::TensorShape(13, 13, 192));
    pool1->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_5.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 4 POOL1  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 4 POOL1 FAILED");
    }

    auto conv2_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv2_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv2_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_2_weights_nchw.bin"), arm_compute::TensorShape(3, 3, 192, 384), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv2_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_2_bias.bin"), arm_compute::TensorShape(384), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv2 = new CNN::Convolution(192, 384, 3, 3, 13, 13, 1, 1, 1, std::move(conv2_kernel), std::move(conv2_bias));
    alexnet.addLayer(conv2, arm_compute::TensorShape(13, 13, 192), arm_compute::TensorShape(13, 13, 384));
    conv2->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_7.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 5 CONV2  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 5 CONV2 FAILED");
    }

    auto conv3_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv3_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv3_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_3_weights_nchw.bin"), arm_compute::TensorShape(3, 3, 384, 256), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv3_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_3_bias.bin"), arm_compute::TensorShape(256), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv3 = new CNN::Convolution(192, 384, 3, 3, 13, 13, 1, 1, 1, std::move(conv3_kernel), std::move(conv3_bias));
    alexnet.addLayer(conv3, arm_compute::TensorShape(13, 13, 384), arm_compute::TensorShape(13, 13, 256));
    conv3->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_9.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 6 CONV3  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 6 CONV3 FAILED");
    }

    auto conv4_kernel = std::make_unique<arm_compute::Tensor>();
    auto conv4_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*conv4_kernel, read_binary_float_vector_asset("alexnet_torch/weights/NCHW/conv_layer_4_weights_nchw.bin"), arm_compute::TensorShape(3, 3, 256, 256), arm_compute::DataLayout::NCHW);
    vectorToTensor(*conv4_bias,read_binary_float_vector_asset("alexnet_torch/weights/conv_layer_4_bias.bin"), arm_compute::TensorShape(256), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *conv4 = new CNN::Convolution(256, 256, 3, 3, 13, 13, 1, 1, 1, std::move(conv4_kernel), std::move(conv4_bias));
    alexnet.addLayer(conv4, arm_compute::TensorShape(13, 13, 256), arm_compute::TensorShape(13, 13, 256));
    conv4->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_11.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 7 CONV4  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 7 CONV4 FAILED");
    }

    auto *pool2 = new CNN::Pooling(3, 3, 256, 13, 13, 2, 0,0,0,0);
    alexnet.addLayer(pool2, arm_compute::TensorShape(13, 13, 256), arm_compute::TensorShape(6, 6, 256));
    pool2->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_12.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 8 POOL2  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 8 POOL2 FAILED");
    }

    auto fc_0_weights = std::make_unique<arm_compute::Tensor>();
    auto fc_0_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc_0_weights, read_binary_float_vector_asset("alexnet_torch/weights/fc0_weights.bin"), arm_compute::TensorShape(9216, 4096), arm_compute::DataLayout::NCHW);
    vectorToTensor(*fc_0_bias,read_binary_float_vector_asset("alexnet_torch/weights/fc0_bias.bin"), arm_compute::TensorShape(4096), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *fc0 = new CNN::FullyConnected(std::move(fc_0_weights), std::move(fc_0_bias), 9216, 4096, 0);
    alexnet.addLayer(fc0, arm_compute::TensorShape(9216), arm_compute::TensorShape(4096));
    fc0->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_14.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 9 FC0  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 9 FC0 FAILED");
    }

    auto fc_1_weights = std::make_unique<arm_compute::Tensor>();
    auto fc_1_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc_1_weights, read_binary_float_vector_asset("alexnet_torch/weights/fc1_weights.bin"), arm_compute::TensorShape(4096, 4096), arm_compute::DataLayout::NCHW);
    vectorToTensor(*fc_1_bias,read_binary_float_vector_asset("alexnet_torch/weights/fc1_bias.bin"), arm_compute::TensorShape(4096), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *fc1 = new CNN::FullyConnected(std::move(fc_1_weights), std::move(fc_1_bias), 4096, 4096, 0);
    alexnet.addLayer(fc1, arm_compute::TensorShape(4096), arm_compute::TensorShape(4096));
    fc1->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_17.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 10 FC1  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 10 FC1 FAILED");
    }

    auto fc_2_weights = std::make_unique<arm_compute::Tensor>();
    auto fc_2_bias = std::make_unique<arm_compute::Tensor>();
    vectorToTensor(*fc_2_weights, read_binary_float_vector_asset("alexnet_torch/weights/fc2_weights.bin"), arm_compute::TensorShape(4096, 1000), arm_compute::DataLayout::NCHW);
    vectorToTensor(*fc_2_bias,read_binary_float_vector_asset("alexnet_torch/weights/fc2_bias.bin"), arm_compute::TensorShape(1000), arm_compute::DataLayout::NCHW);
    //allocate input and output vectors for first convolutional layer
    auto *fc2 = new CNN::FullyConnected(std::move(fc_2_weights), std::move(fc_2_bias), 4096, 1000, 0, arm_compute::ActivationLayerInfo::ActivationFunction::LINEAR);
    alexnet.addLayer(fc2, arm_compute::TensorShape(4096), arm_compute::TensorShape(1000));
    fc2->configure_acl();
    alexnet.forward_acl();
    if (assert_equal(*alexnet.output_tensor.get(), read_binary_float_vector_asset("alexnet_torch/outputs/NCHW/layer_19.bin"))) {
        LOGI_TEST("TEST ALEXNET LAYER 11 FC2  WAS SUCCESSFUL");
    } else {
        LOGI_TEST("TEST ALEXNET LAYER 11 FC2 FAILED");
    }
}
