import sys
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def is_comment_or_empty(line: str, in_block_comment: bool):
    stripped = line.strip()
    if in_block_comment:
        if "*/" in stripped:
            return True, False
        else:
            return True, True

    if stripped.startswith("/*"):
        if "*/" in stripped:
            return True, False
        else:
            return True, True

    if not stripped or stripped.startswith("//"):
        return True, False

    return False, False


def predict_line_cfp(model, tokenizer, line: str, device: torch.device):
    inputs = tokenizer(line, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.squeeze().cpu().numpy()
    cfp_pred = int(np.rint(np.sum(preds)))
    return cfp_pred


def process_lines(lines, model, tokenizer, device):
    total_pred = 0
    line_count = 0
    in_block = False

    for raw_line in lines:
        skip, in_block = is_comment_or_empty(raw_line, in_block)
        if skip:
            continue

        line = raw_line.rstrip("\n")
        pred_cfp = predict_line_cfp(model, tokenizer, line, device)

        print(f"LINE {line_count+1:03d}: {line}")
        print(f"  Predicted CFP: {pred_cfp}\n")

        total_pred += pred_cfp
        line_count += 1

    print("="*40)
    print(f"Total lines evaluated : {line_count}")
    print(f"Sum Predicted CFP      : {total_pred}")
    print("="*40)


def main():
    parser = argparse.ArgumentParser(
        description="Predict COSMIC CFP per line using fine-tuned CodeBERT"
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to the source code file (omit for stdin)",
    )
    parser.add_argument(
        "--model_dir",
        default="codebert-cfp/best-model",
        help="Directory of the fine-tuned model"
    )
    args = parser.parse_args()

    # Load model & tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Read input: file or stdin
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = f.readlines()
    else:
        print("Reading from stdin. Enter code, then Ctrl-D to finish:")
        lines = sys.stdin.readlines()

    process_lines(lines, model, tokenizer, device)

if __name__ == "__main__":
    main()

