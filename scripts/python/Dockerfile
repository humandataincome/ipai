FROM python:3.8

RUN pip install onnx onnxruntime datasets tf2onnx #optimum

RUN pip install git+https://github.com/huggingface/optimum.git

ENV HF_HOME=/.cache/huggingface

COPY onnxgen.py .

RUN mkdir -p models

ENTRYPOINT [ "python", "onnxgen.py" ]

# USAGE EXAMPLE
# docker build . --tag onnxgen
# docker run --rm -v "$PWD"/data/huggingface:/.cache/huggingface -v "$PWD"/data/models:/models "onnxgen"  --model_id "EleutherAI/gpt-neo-125M" --file_name "decoder_model.onnx" --type "causal-lm"
# docker run --rm -v "$PWD"/data/huggingface:/.cache/huggingface -v "$PWD"/data/models:/models "onnxgen"  --model_id "google/flan-t5-small" --file_name "decoder_model.onnx" --type "seq2seq"
# docker run --rm -v "$PWD"/data/huggingface:/.cache/huggingface -v "$PWD"/data/models:/models "onnxgen"  --model_id "google/flan-t5-base" --file_name "decoder_model.onnx" --type "seq2seq"
# docker run --rm -v "$PWD"/data/huggingface:/.cache/huggingface -v "$PWD"/data/models:/models "onnxgen"  --model_id "gpt2" --file_name "decoder_model.onnx" --type "causal-lm"
# docker run --rm -v "$PWD"/data/huggingface:/.cache/huggingface -v "$PWD"/data/models:/models "onnxgen"  --model_id "distilgpt2" --file_name "decoder_model.onnx" --type "causal-lm"
# docker run --rm -v "$PWD"/data/huggingface:/.cache/huggingface -v "$PWD"/data/models:/models "onnxgen"  --model_id "microsoft/DialoGPT-small" --file_name "decoder_model.onnx" --type "causal-lm"

# docker run --rm -v "$PWD"/data/huggingface:/.cache/huggingface -v "$PWD"/data/models:/models "onnxgen"  --model_id "EleutherAI/pythia-70m" --file_name "decoder_model.onnx" --type "causal-lm"

# --model_id "EleutherAI/gpt-neo-125M"
# --model_id "EleutherAI/gpt-neo-1.3B"
# --model_id "microsoft/DialoGPT-small"
# --model_id "facebook/opt-125m"
# --model_id "gpt2"
# --model_id "distilgpt2"
