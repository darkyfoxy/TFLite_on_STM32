#!/bin/bash

mkdir ./TFLite
cd ./TFLite

mkdir ./tensorflow
cd ./tensorflow
mkdir ./lite
cd ./lite

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/portable_type_to_tflitetype.h

mkdir ./c
cd ./c

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/c/builtin_op_data.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/c/c_api_types.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/c/common.c
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/c/common.h

cd ./../
mkdir ./core
cd ./core
mkdir ./api
cd ./api

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/error_reporter.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/error_reporter.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/flatbuffer_conversions.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/flatbuffer_conversions.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/op_resolver.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/op_resolver.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/tensor_utils.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/core/api/tensor_utils.h

cd ./../../
mkdir ./schema
cd ./schema

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/schema/schema_generated.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/schema/schema_utils.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/schema/schema_utils.h

cd ./../
mkdir ./kernels
cd ./kernels

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/kernel_util.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/kernel_util.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/op_macros.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/padding.h

mkdir ./internal
cd ./internal

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/common.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/compatibility.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/cppmath.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/max.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/min.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/portable_tensor.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/portable_tensor_utils.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/quantization_util.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/quantization_util.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/runtime_shape.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/strided_slice_logic.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/tensor_ctypes.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/types.h

mkdir ./optimized
cd ./optimized

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/optimized/neon_check.h

cd ./../

mkdir ./reference
cd ./reference

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/add.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/add_n.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/arg_min_max.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/batch_to_space_nd.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/binary_function.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/ceil.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/comparisons.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/concatenation.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/conv.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/cumsum.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/depth_to_space.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/dequantize.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/elu.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/exp.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/fill.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/floor.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/floor_div.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/floor_mod.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/fully_connected.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/hard_swish.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/l2normalization.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/leaky_relu.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/logistic.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/maximum_minimum.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/mul.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/neg.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/pad.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/pooling.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/prelu.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/quantize.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/reduce.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/requantize.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/resize_bilinear.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/round.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/softmax.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/space_to_batch_nd.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/space_to_depth.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/strided_slice.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/sub.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/tanh.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/transpose.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/transpose_conv.h

mkdir ./integer_ops
cd ./integer_ops

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/add.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/conv.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/mean.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/mul.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/kernels/internal/reference/integer_ops/transpose_conv.h

cd ./../../../../
mkdir ./micro
cd ./micro

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/all_ops_resolver.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/all_ops_resolver.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/compatibility.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/debug_log.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/debug_log.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/flatbuffer_utils.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/flatbuffer_utils.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/memory_helpers.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/memory_helpers.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_allocator.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_allocator.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_arena_constants.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_error_reporter.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_error_reporter.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_graph.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_graph.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_interpreter.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_interpreter.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_mutable_op_resolver.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_op_resolver.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_profiler.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_profiler.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_resource_variable.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_resource_variable.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_string.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_string.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_time.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_time.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_utils.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/micro_utils.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/simple_memory_allocator.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/simple_memory_allocator.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/system_setup.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/system_setup.h

#mkdir ./models
#cd ./models
#
#cp ./../../../../../models_source/person_detect_model_data.h ./
#cp ./../../../../../models_source/person_detect_model_data.cc ./
#
#cd ./../
mkdir ./memory_planner
cd ./memory_planner

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/memory_planner/greedy_memory_planner.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/memory_planner/micro_memory_planner.h

cd ./../
mkdir ./examples
cd ./examples
mkdir ./person_detection
cd ./person_detection

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/detection_responder.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/detection_responder.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/image_provider.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/image_provider.h
#wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/main_functions.cc
#wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/main_functions.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/model_settings.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/examples/person_detection/model_settings.h

cd ./../../
mkdir ./kernels
cd ./kernels

wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/activation_utils.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/activations.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/activations.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/activations_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/add.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/add.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/add_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/add_n.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/arg_min_max.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/assign_variable.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/batch_to_space_nd.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/call_once.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/ceil.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/comparisons.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/concatenation.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/conv.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/conv.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/conv_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/cumsum.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/depth_to_space.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/depthwise_conv.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/depthwise_conv.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/depthwise_conv_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/dequantize.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/dequantize.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/dequantize_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/detection_postprocess.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/elementwise.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/elu.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/ethosu.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/ethosu.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/exp.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/expand_dims.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/fill.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/floor.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/floor_div.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/floor_mod.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/fully_connected.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/fully_connected.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/fully_connected_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/hard_swish.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/hard_swish.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/hard_swish_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/kernel_util.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/kernel_util.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/l2_pool_2d.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/l2norm.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/leaky_relu.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/leaky_relu.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/leaky_relu_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/logical.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/logical.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/logical_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/logistic.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/logistic.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/logistic_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/maximum_minimum.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/micro_ops.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/micro_utils.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/mul.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/mul.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/mul_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/neg.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/pack.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/pad.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/pooling.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/pooling.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/pooling_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/prelu.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/prelu.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/prelu_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/quantize.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/quantize.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/quantize_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/read_variable.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/reduce.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/reshape.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/resize_bilinear.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/round.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/shape.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/softmax.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/softmax.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/softmax_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/space_to_batch_nd.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/space_to_depth.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/split.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/split_v.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/squeeze.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/strided_slice.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/sub.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/sub.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/sub_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/svdf.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/svdf.h
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/svdf_common.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/tanh.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/transpose.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/transpose_conv.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/unpack.cc
wget -nv https://raw.githubusercontent.com/tensorflow/tflite-micro/main/tensorflow/lite/micro/kernels/var_handle.cc

cd ./../../../../
mkdir ./ruy
cd ./ruy
mkdir ./profiler
cd ./profiler

wget -nv https://raw.githubusercontent.com/google/ruy/master/ruy/profiler/instrumentation.cc
wget -nv https://raw.githubusercontent.com/google/ruy/master/ruy/profiler/instrumentation.h

cd ./../../
mkdir ./flatbuffers
cd ./flatbuffers

wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/allocator.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/array.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/base.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/buffer.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/buffer_ref.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/default_allocator.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/detached_buffer.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/flatbuffer_builder.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/flatbuffers.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/flexbuffers.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/stl_emulation.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/string.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/struct.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/table.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/util.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/vector.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/vector_downward.h
wget -nv https://raw.githubusercontent.com/google/flatbuffers/master/include/flatbuffers/verifier.h

cd ./../
mkdir ./fixedpoint
cd ./fixedpoint

wget -nv https://raw.githubusercontent.com/google/gemmlowp/master/fixedpoint/fixedpoint.h

cd ./../
mkdir ./internal
cd ./internal

wget -nv https://raw.githubusercontent.com/google/gemmlowp/master/internal/detect_platform.h

cd ./../
