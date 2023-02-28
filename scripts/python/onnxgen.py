import argparse

from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoOptimizationConfig


def pipeline(model_id, file_name, type):
    # Load PyTorch model and convert to ONNX
    class_by_type = ORTModelForCausalLM if type == "causal-lm" else ORTModelForSeq2SeqLM
    onnx_model = class_by_type.from_pretrained(model_id, from_transformers=True)

    # Create quantizer
    quantizer = ORTQuantizer.from_pretrained(onnx_model.model_save_dir, file_name)

    # Define the quantization strategy by creating the appropriate configuration
    quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    # Quantize the model
    quantizer.quantize(save_dir="models/onnx/quantized/" + model_id, quantization_config=quantization_config)

def main():
    parser = argparse.ArgumentParser(description="Export onnx model")
    parser.add_argument(
        "--model_id",
        type=str,
        nargs="?",
        help="The model to be exported from huggingface",
    )

    parser.add_argument(
        "--file_name",
        type=str,
        nargs="?",
        help="The filename for the model inside the dir",
    )

    parser.add_argument(
        "--type",
        type=str,
        nargs="?",
        help="The model type [causal-lm, seq2seq]",
    )

    args = parser.parse_args()

    pipeline(args.model_id, args.file_name, args.type)


if __name__ == "__main__":
    main()
