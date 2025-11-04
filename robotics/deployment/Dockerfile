FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip install vllm transformers accelerate tensorrt pycuda

WORKDIR /app
COPY . .

EXPOSE 8000
CMD ["python", "src/vllm_server.py", "--model", "llava-hf/llava-1.6-mistral-7b-hf"]
