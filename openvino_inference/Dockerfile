FROM openvino/ubuntu22_dev:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./models/ ./models/
COPY ./data/ ./data/
COPY ./scripts/ ./scripts/

WORKDIR /app/scripts

RUN bash convert_to_ir.sh && \
    echo "--- Starting NNCF Quantization ---" && \
    python run_nncf_quantization.py --model_type resnet && \
    python run_nncf_quantization.py --model_type bert && \
    python run_nncf_quantization.py --model_type smollm && \
    echo "--- Starting Full Benchmark ---" && \
    python run_full_benchmark.py --model_type resnet && \
    python run_full_benchmark.py --model_type bert && \
    python run_full_benchmark.py --model_type smollm

WORKDIR /app

CMD ["echo", "OpenVINO pipeline finished during build. Use 'docker cp' to extract generated models and plots."]
