FROM python:3.10-slim AS builder

WORKDIR /app


COPY requirements.txt ./

RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt


COPY . .


RUN echo "INFO: Starting SmolLM pipeline..." && \
    \
    echo "INFO: Running smollm_onnx.py to convert PyTorch model to ONNX..." && \
    python smollm_onnx.py && \
    echo "INFO: smollm_onnx.py completed. smollm_135m.onnx should be created." && \
    \
    echo "INFO: Running smollm_tflite.py to simplify ONNX and convert to TFLite..." && \
    python smollm_tflite.py && \
    echo "INFO: smollm_tflite.py completed. Simplified ONNX, TF SavedModel, and TFLite models should be created." && \
    \
    echo "INFO: Running compare_smollm_inference.py to benchmark models..." && \
    python compare_smollm_inference.py && \
    echo "INFO: SmolLM pipeline finished. Comparison plot should be created."

CMD ["echo", "All scripts executed during Docker image build. Models and comparison results are part of the image."]