#!/bin/bash


set -e


echo "Creating output directories..."
mkdir -p ../models/ir_fp32/resnet18 ../models/ir_fp16/resnet18
mkdir -p ../models/ir_fp32/bert ../models/ir_fp16/bert
mkdir -p ../models/ir_fp32/smollm ../models/ir_fp16/smollm

# --- Конвертация ResNet18 ---
echo "Converting ResNet18..."
# Для FP32 явно отключаем сжатие в FP16
mo --input_model ../models/onnx/resnet18.onnx \
   --output_dir ../models/ir_fp32/resnet18 \
   --model_name resnet18_fp32 \
   --compress_to_fp16=False

# Для FP16 просто запускаем mo (режим по умолчанию)
mo --input_model ../models/onnx/resnet18.onnx \
   --output_dir ../models/ir_fp16/resnet18 \
   --model_name resnet18_fp16

# --- Конвертация BERT ---
echo "Converting BERT..."
mo --input_model ../models/onnx/bert_model.onnx \
   --input_shape [1,128],[1,128],[1,128] \
   --output_dir ../models/ir_fp32/bert \
   --model_name bert_fp32 \
   --compress_to_fp16=False

mo --input_model ../models/onnx/bert_model.onnx \
   --input_shape [1,128],[1,128],[1,128] \
   --output_dir ../models/ir_fp16/bert \
   --model_name bert_fp16

# --- Конвертация Smollm ---
echo "Converting Smollm..."
mo --input_model ../models/onnx/smollm_135m.onnx \
   --input_shape [1,128],[1,128] \
   --output_dir ../models/ir_fp32/smollm \
   --model_name smollm_fp32 \
   --compress_to_fp16=False

mo --input_model ../models/onnx/smollm_135m.onnx \
   --input_shape [1,128],[1,128] \
   --output_dir ../models/ir_fp16/smollm \
   --model_name smollm_fp16

echo "ALL CONVERSIONS FINISHED SUCCESSFULLY!"