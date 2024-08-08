import argparse
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Intel/neural-chat-7b-v3-1")
    parser.add_argument(
        "--model_file",
        type=str,
        required=False,
        default=None,
        help="Path to GGUF model file. This is only required when using GGUF model. Make sure you also pass `model_name` and `tokenizer_name`.",
    )
    parser.add_argument("--tokenizer_name", type=str, required=False)
    parser.add_argument("--no_neural_speed", action="store_true", required=False)
    parser.add_argument(
        "--quantize", choices=["int4", "int8", "fp32"], default="fp32", required=False
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", required=False
    )
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    # if pytorch model
    if not args.model_file:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True
        )
        kwargs = {}
        if args.no_neural_speed:
            kwargs["use_neural_speed"] = False
        print(f"use_neural_speed: {kwargs.get('use_neural_speed', True)}")

        if args.quantize == "int4":
            kwargs["load_in_4bit"] = True
            print(f"load_in_4bit: {kwargs.get('load_in_4bit', False)}")
        elif args.quantize == "int8":
            kwargs["load_in_8bit"] = True
            print(f"load_in_8bit: {kwargs.get('load_in_8bit', False)}")

        model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)

    # if GGUF model
    else:
        if args.tokenizer_name and args.model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, model_file=args.model_file
            )
        else:
            raise ValueError(
                "Please provide both `tokenizer_name` and `model_name` for GGUF model"
            )

    streamer = TextStreamer(tokenizer)
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids
    model.generate(inputs, streamer=streamer, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
