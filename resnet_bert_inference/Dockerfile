FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt ./

RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

COPY . .


# --- Resnet18 Pipeline ---
RUN echo "INFO: Starting ResNet18 pipeline..." && \
    echo "INFO: Running resnet_onnx_1.py..." && \
    python resnet_onnx_1.py && \
    echo "INFO: resnet_onnx_1.py completed. resnet18.onnx should be created." && \
    echo "INFO: Running onnx2tf for ResNet18..." && \
    onnx2tf -i resnet18.onnx -o resnet18_tf && \
    echo "INFO: onnx2tf for ResNet18 completed. resnet18_tf folder should be created." && \
    echo "INFO: Running compare_resnet_inference_1.py..." && \
    python compare_resnet_inference_1.py && \
    echo "INFO: ResNet18 pipeline finished."

# --- Bert Pipeline ---
RUN echo "INFO: Starting BERT pipeline..." && \
    echo "INFO: Running bert_onnx_1.py..." && \
    python bert_onnx_1.py && \
    echo "INFO: bert_onnx_1.py completed. bert_model.onnx should be created." && \
    echo "INFO: Running bert_tflite_1.py..." && \
    python bert_tflite_1.py && \
    echo "INFO: bert_tflite_1.py completed. BERT TFLite models should be created." && \
    echo "INFO: Running compare_bert_inference_1.py..." && \
    python compare_bert_inference_1.py && \
    echo "INFO: BERT pipeline finished."

CMD ["echo", "All scripts executed during Docker image build. Models and comparison results are part of the image or were printed to logs."]