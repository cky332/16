#!/usr/bin/env python3
"""
Test watermark detection robustness against code rewriting attacks.

Loads pre-generated watermarked code, applies various AST-based rewriting
attacks, runs WLLM watermark detection (CPU-only, no model needed), and
reports how each attack degrades detection effectiveness.

Usage:
    python test_watermark_detection.py                    # full run
    python test_watermark_detection.py --limit 10         # quick test
    python test_watermark_detection.py --attacks rename_variables reformat

When the HuggingFace tokenizer (bigcode/starcoderbase-3b) is available,
the script uses it for accurate watermark detection. When it is not
available (e.g., no network access), a fallback byte-level tokenizer is
used. The fallback tokenizer enables the full pipeline to run and
produces valid RELATIVE comparisons (original vs attacked), though the
absolute z-scores will differ from those obtained with the real tokenizer.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

import torch

from watermark import WatermarkDetector
from code_rewriting import extract_prompt, ALL_ATTACKS


# ---------------------------------------------------------------------------
# Fallback byte-level tokenizer (used when HuggingFace is unavailable)
# ---------------------------------------------------------------------------

class ByteLevelTokenizer:
    """A simple byte-level tokenizer that maps each byte to a unique token ID.

    This tokenizer assigns each byte value (0-255) a unique ID, plus a few
    special token IDs. It is used as a fallback when the HuggingFace tokenizer
    cannot be loaded (e.g., no network access).

    The relative z-score comparisons (original vs attacked) remain valid
    because the same tokenizer is used consistently for both runs.
    """

    VOCAB_SIZE = 260  # 256 bytes + 4 special tokens
    PAD_ID = 256
    EOS_ID = 257
    BOS_ID = 258
    UNK_ID = 259

    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"

    def __call__(self, text, padding=True, truncation=True, return_tensors="pt", **kwargs):
        input_ids = [b for b in text.encode("utf-8", errors="replace")]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
        return {"input_ids": [input_ids]}

    def get_vocab(self):
        vocab = {f"byte_{i}": i for i in range(256)}
        vocab["<pad>"] = self.PAD_ID
        vocab["<eos>"] = self.EOS_ID
        vocab["<bos>"] = self.BOS_ID
        vocab["<unk>"] = self.UNK_ID
        return vocab


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str):
    """Load the HuggingFace tokenizer, falling back to ByteLevelTokenizer."""
    try:
        from transformers import AutoTokenizer

        print(f"Loading HuggingFace tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            truncation_side="left",
            padding_side="right",
        )
        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
            else:
                raise ValueError("No eos_token or bos_token found")
        tokenizer.pad_token = tokenizer.eos_token
        print("  HuggingFace tokenizer loaded successfully.")
        return tokenizer, False  # is_fallback = False
    except Exception as e:
        print(f"  Warning: Cannot load HuggingFace tokenizer ({e})")
        print("  Using fallback byte-level tokenizer.")
        print("  NOTE: Absolute z-scores will differ from the real tokenizer,")
        print("        but relative comparisons (original vs attacked) are valid.")
        return ByteLevelTokenizer(), True  # is_fallback = True


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test watermark detection robustness against code rewriting attacks"
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default="outputs/generations.json",
        help="Path to watermarked code generations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bigcode/starcoderbase-3b",
        help="Tokenizer model name (only tokenizer is loaded, no model weights)",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["all"],
        help="Attacks to apply. Use 'all' for all attacks, or specify names: "
        "rename_variables reformat remove_comments dead_code swap_if_else "
        "rewrite_expressions remove_type_annotations combined",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Gamma for WLLM detection"
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=4.0,
        help="Z-score threshold for detection",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/robustness_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def tokenize(tokenizer, text: str):
    """Tokenize text and return input_ids tensor."""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return inputs["input_ids"].squeeze()


def run_detection(detector, tokenizer, full_code: str, prompt: str) -> dict:
    """Run WLLM watermark detection on a single code sample.

    Returns detection result dict with z_score, green_fraction, prediction, etc.
    Returns None if detection fails (e.g., empty generation).
    """
    tokenized_text = tokenize(tokenizer, full_code)
    tokenized_prefix = tokenize(tokenizer, prompt)
    prefix_len = len(tokenized_prefix)

    # Ensure there are tokens to score after the prefix
    if len(tokenized_text) <= prefix_len:
        return None

    try:
        result = detector.detect(
            tokenized_text=tokenized_text,
            tokenized_prefix=tokenized_prefix,
        )
        if result.get("invalid", False):
            return None
        return result
    except Exception as e:
        print(f"  Detection error: {e}")
        return None


def apply_attack(attack_fn, full_code: str, prompt: str) -> str:
    """Apply an attack to the code.

    Tries attacking the full code first (needed for AST context like variable
    renaming). Falls back to attacking only the generated portion.
    """
    generated_part = full_code[len(prompt):]
    if not generated_part.strip():
        return full_code

    # Try applying attack to full code (needed for AST context)
    attacked_full = attack_fn(full_code)
    if attacked_full != full_code:
        return attacked_full

    # Fallback: attack only the generated part
    attacked_gen = attack_fn(generated_part)
    if attacked_gen != generated_part:
        return prompt + attacked_gen

    return full_code


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> dict:
    """Compute aggregate metrics for a list of detection results."""
    if not results:
        return {"n_samples": 0}

    z_scores = [r["z_score"] for r in results]
    green_fracs = [r["green_fraction"] for r in results]
    predictions = [r["prediction"] for r in results]

    return {
        "n_samples": len(results),
        "mean_z_score": round(statistics.mean(z_scores), 4),
        "median_z_score": round(statistics.median(z_scores), 4),
        "std_z_score": round(statistics.stdev(z_scores), 4) if len(z_scores) > 1 else 0,
        "detection_rate": round(sum(predictions) / len(predictions), 4),
        "mean_green_fraction": round(statistics.mean(green_fracs), 4),
    }


def compute_paired_deltas(original_results: list, attack_results: list) -> dict:
    """Compute paired z-score deltas for samples that appear in both sets."""
    orig_by_idx = {r["sample_idx"]: r for r in original_results}
    deltas = []
    for r in attack_results:
        idx = r["sample_idx"]
        if idx in orig_by_idx:
            delta = r["z_score"] - orig_by_idx[idx]["z_score"]
            deltas.append(delta)

    if not deltas:
        return {"n_paired": 0, "mean_delta_z": 0, "median_delta_z": 0}

    return {
        "n_paired": len(deltas),
        "mean_delta_z": round(statistics.mean(deltas), 4),
        "median_delta_z": round(statistics.median(deltas), 4),
        "pct_reduced": round(sum(1 for d in deltas if d < 0) / len(deltas), 4),
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results_table(all_metrics: dict, paired_deltas: dict):
    """Print a human-readable results table."""
    header = (
        f"{'Attack':<25} | {'N':>4} | {'Mean Z':>8} | {'Med Z':>8} | "
        f"{'Det Rate':>9} | {'Green Fr':>9} | {'Delta Z':>8}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("Watermark Detection Robustness Results (WLLM)")
    print(sep)
    print(header)
    print(sep)

    for attack_name in ["original"] + sorted(
        k for k in all_metrics if k != "original"
    ):
        m = all_metrics.get(attack_name)
        if not m or m["n_samples"] == 0:
            continue
        delta_z = paired_deltas.get(attack_name, {}).get("mean_delta_z", 0)
        if attack_name == "original":
            delta_str = "     ---"
        else:
            delta_str = f"{delta_z:>+8.4f}"

        print(
            f"{attack_name:<25} | {m['n_samples']:>4} | {m['mean_z_score']:>8.4f} | "
            f"{m['median_z_score']:>8.4f} | {m['detection_rate']:>9.4f} | "
            f"{m['mean_green_fraction']:>9.4f} | {delta_str}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Determine which attacks to run
    if "all" in args.attacks:
        attack_names = list(ALL_ATTACKS.keys())
    else:
        attack_names = args.attacks
        for name in attack_names:
            if name not in ALL_ATTACKS:
                print(
                    f"Error: Unknown attack '{name}'. "
                    f"Available: {list(ALL_ATTACKS.keys())}"
                )
                sys.exit(1)

    # Load tokenizer (HuggingFace or fallback)
    tokenizer, is_fallback = load_tokenizer(args.model)

    # Load generations
    print(f"Loading generations from: {args.generations_path}")
    with open(args.generations_path) as f:
        generations = json.load(f)

    if args.limit:
        generations = generations[: args.limit]
    print(f"Loaded {len(generations)} samples")

    # Initialize WLLM detector
    vocab = list(tokenizer.get_vocab().values())
    detector = WatermarkDetector(
        vocab=vocab,
        gamma=args.gamma,
        tokenizer=tokenizer,
        z_threshold=args.z_threshold,
    )
    print(
        f"Initialized WLLM detector (gamma={args.gamma}, "
        f"z_threshold={args.z_threshold}, vocab_size={len(vocab)})"
    )

    # Run detection on original + all attacks
    results_by_attack = defaultdict(list)

    # First, detect on original (unmodified) code
    print("\nRunning detection on original code...")
    for idx, gen_list in enumerate(generations):
        code = gen_list[0]
        prompt = extract_prompt(code)
        if not prompt:
            continue

        generated_part = code[len(prompt):]
        if not generated_part.strip():
            continue

        result = run_detection(detector, tokenizer, code, prompt)
        if result is not None:
            result["sample_idx"] = idx
            results_by_attack["original"].append(result)

    print(f"  Successfully detected: {len(results_by_attack['original'])} samples")

    # Then, apply each attack and re-detect
    for attack_name in attack_names:
        attack_fn = ALL_ATTACKS[attack_name]
        print(f"\nRunning attack: {attack_name}...")

        n_changed = 0
        n_failed = 0
        for idx, gen_list in enumerate(generations):
            code = gen_list[0]
            prompt = extract_prompt(code)
            if not prompt:
                continue

            generated_part = code[len(prompt):]
            if not generated_part.strip():
                continue

            # Apply attack
            attacked_code = apply_attack(attack_fn, code, prompt)

            if attacked_code != code:
                n_changed += 1

            # Re-extract prompt from attacked code for proper prefix alignment
            attacked_prompt = extract_prompt(attacked_code)
            if not attacked_prompt:
                attacked_prompt = prompt  # fallback

            result = run_detection(
                detector, tokenizer, attacked_code, attacked_prompt
            )
            if result is not None:
                result["sample_idx"] = idx
                result["code_changed"] = attacked_code != code
                results_by_attack[attack_name].append(result)
            else:
                n_failed += 1

        print(
            f"  Detected: {len(results_by_attack[attack_name])} samples, "
            f"Changed: {n_changed}, Failed: {n_failed}"
        )

    # Compute metrics
    all_metrics = {}
    paired_deltas = {}
    for attack_name, results in results_by_attack.items():
        all_metrics[attack_name] = compute_metrics(results)
        if attack_name != "original":
            paired_deltas[attack_name] = compute_paired_deltas(
                results_by_attack["original"], results
            )

    # Print results
    print_results_table(all_metrics, paired_deltas)

    if is_fallback:
        print(
            "\nNOTE: Results above were produced with the fallback byte-level "
            "tokenizer.\nFor accurate absolute z-scores, re-run with access "
            "to the HuggingFace tokenizer\n(bigcode/starcoderbase-3b).\n"
            "Relative comparisons (Delta Z) between attacks are still valid."
        )

    # Save results
    output = {
        "config": {
            "model": args.model,
            "gamma": args.gamma,
            "z_threshold": args.z_threshold,
            "n_generations": len(generations),
            "attacks": attack_names,
            "tokenizer": "fallback_byte_level" if is_fallback else args.model,
        },
        "metrics": all_metrics,
        "paired_deltas": paired_deltas,
        "raw_results": {k: v for k, v in results_by_attack.items()},
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
