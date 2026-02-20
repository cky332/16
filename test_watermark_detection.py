#!/usr/bin/env python3
"""
Test watermark detection robustness against code rewriting attacks.

Supports three detection methods:
  - WLLM:  Basic green-list watermark detection (CPU-only, no model needed)
  - SWEET: Entropy-aware watermark detection (requires GPU + model for
           per-token entropy computation; only high-entropy positions are scored)
  - EXP:   EXP-edit detection based on Levenshtein distance between token
           sequence and a secret key (CPU, permutation-test p-value)

Usage:
    # WLLM (default, CPU-only)
    python test_watermark_detection.py --detector wllm

    # SWEET (needs GPU for entropy computation)
    python test_watermark_detection.py --detector sweet \
        --generations_path outputs_sweet/generations.json

    # EXP-edit
    python test_watermark_detection.py --detector exp

    # Quick test with subset
    python test_watermark_detection.py --detector sweet --limit 10

    # Select specific attacks
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
# Tokenizer and model loading
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


def load_model(model_name: str, device: str, precision: str = "fp32"):
    """Load the model for entropy computation (SWEET detector only).

    Returns the model moved to the specified device.
    """
    from transformers import AutoModelForCausalLM

    precision_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = precision_map.get(precision, torch.float32)

    print(f"Loading model for entropy computation: {model_name} ({precision})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}.")
    return model


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test watermark detection robustness against code rewriting attacks"
    )
    # --- General arguments ---
    parser.add_argument(
        "--detector",
        type=str,
        choices=["wllm", "sweet", "exp"],
        default="wllm",
        help="Detection method: wllm (default), sweet, or exp",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default=None,
        help="Path to watermarked code generations. "
        "Defaults: outputs/generations.json (wllm), "
        "outputs_sweet/generations.json (sweet), "
        "outputs/generations.json (exp)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bigcode/starcoderbase-3b",
        help="Tokenizer/model name (only tokenizer for WLLM/EXP, full model for SWEET)",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["all"],
        help="Attacks to apply. Use 'all' for all attacks, or specify names: "
        "rename_variables reformat remove_comments dead_code swap_if_else "
        "rewrite_expressions remove_type_annotations structural_paraphrase "
        "llm_paraphrase combined",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results (default: outputs/robustness_results_{detector}.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )

    # --- WLLM / SWEET arguments ---
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Gamma for WLLM/SWEET detection"
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=4.0,
        help="Z-score threshold for WLLM/SWEET detection",
    )

    # --- SWEET-specific arguments ---
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.5,
        help="Entropy threshold for SWEET (tokens below this are not scored)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Model precision for SWEET (fp32, fp16, bf16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for SWEET model (default: cuda if available, else cpu)",
    )

    # --- EXP-edit-specific arguments ---
    parser.add_argument(
        "--key_length",
        type=int,
        default=512,
        help="Secret key length for EXP-edit detection",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for EXP-edit (None = use full token length)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=100,
        help="Number of permutation runs for EXP-edit p-value computation",
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.1,
        help="P-value threshold for EXP-edit detection",
    )
    parser.add_argument(
        "--hash_key",
        type=int,
        default=15485863,
        help="Hash key / seed for watermark (must match generation)",
    )

    args = parser.parse_args()

    # Set defaults based on detector type
    if args.generations_path is None:
        if args.detector == "sweet":
            args.generations_path = "outputs_sweet/generations.json"
        else:
            args.generations_path = "outputs/generations.json"

    if args.output_path is None:
        args.output_path = f"outputs/robustness_results_{args.detector}.json"

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


# ---------------------------------------------------------------------------
# Entropy computation (SWEET)
# ---------------------------------------------------------------------------

def calculate_entropy(model, tokenized_text, device):
    """Compute per-token entropy using the model's output distribution.

    Returns a list of entropy values, one per token position.
    """
    with torch.no_grad():
        input_ids = tokenized_text.unsqueeze(0).to(device)
        output = model(input_ids, return_dict=True)
        probs = torch.softmax(output.logits, dim=-1)
        entropy = -torch.where(
            probs > 0, probs * probs.log(), probs.new([0.0])
        ).sum(dim=-1)
        return entropy[0].cpu().tolist()


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


def run_wllm_detection(detector, tokenizer, full_code: str, prompt: str) -> dict:
    """Run WLLM watermark detection on a single code sample."""
    tokenized_text = tokenize(tokenizer, full_code)
    tokenized_prefix = tokenize(tokenizer, prompt)
    prefix_len = len(tokenized_prefix)

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


def run_sweet_detection(
    detector, tokenizer, model, full_code: str, prompt: str, device: str
) -> dict:
    """Run SWEET watermark detection with entropy-aware scoring.

    Computes per-token entropy using the model, then passes it to the
    SweetDetector which only scores high-entropy token positions.
    """
    tokenized_text = tokenize(tokenizer, full_code)
    tokenized_prefix = tokenize(tokenizer, prompt)
    prefix_len = len(tokenized_prefix)

    if len(tokenized_text) <= prefix_len:
        return None

    try:
        # Compute entropy for all token positions
        entropy = calculate_entropy(model, tokenized_text, device)
        # Shift entropy right to align with generation-time convention:
        # entropy[i] represents the model's uncertainty *before* generating token i
        entropy = [0.0] + entropy[:-1]

        result = detector.detect(
            tokenized_text=tokenized_text,
            tokenized_prefix=tokenized_prefix,
            entropy=entropy,
        )
        if result.get("invalid", False):
            return None
        return result
    except Exception as e:
        print(f"  Detection error: {e}")
        return None


def run_exp_detection(
    detector, tokenizer, full_code: str, prompt: str, n_runs: int = 100
) -> dict:
    """Run EXP-edit detection using Levenshtein distance and permutation test.

    Only the generated tokens (after the prompt) are scored.
    Returns p-value and prediction.
    """
    tokenized_text = tokenize(tokenizer, full_code)
    tokenized_prefix = tokenize(tokenizer, prompt)
    prefix_len = len(tokenized_prefix)

    generated_tokens = tokenized_text[prefix_len:]
    if len(generated_tokens) == 0:
        return None

    try:
        result = detector.detect(
            generated_tokens=generated_tokens,
            n_runs=n_runs,
        )
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

def compute_metrics_zscore(results: list) -> dict:
    """Compute aggregate metrics for z-score based detectors (WLLM, SWEET)."""
    if not results:
        return {"n_samples": 0}

    z_scores = [r["z_score"] for r in results]
    green_fracs = [r["green_fraction"] for r in results]
    predictions = [r["prediction"] for r in results]

    metrics = {
        "n_samples": len(results),
        "mean_z_score": round(statistics.mean(z_scores), 4),
        "median_z_score": round(statistics.median(z_scores), 4),
        "std_z_score": round(statistics.stdev(z_scores), 4) if len(z_scores) > 1 else 0,
        "detection_rate": round(sum(predictions) / len(predictions), 4),
        "mean_green_fraction": round(statistics.mean(green_fracs), 4),
    }

    # SWEET-specific: watermarking_fraction (pct of tokens above entropy threshold)
    if "watermarking_fraction" in results[0]:
        wf = [r["watermarking_fraction"] for r in results]
        metrics["mean_watermarking_fraction"] = round(statistics.mean(wf), 4)

    return metrics


def compute_metrics_pvalue(results: list) -> dict:
    """Compute aggregate metrics for p-value based detectors (EXP-edit)."""
    if not results:
        return {"n_samples": 0}

    p_values = [r["p_value"] for r in results]
    predictions = [r["prediction"] for r in results]
    scores = [r.get("true_key_score", 0) for r in results]

    return {
        "n_samples": len(results),
        "mean_p_value": round(statistics.mean(p_values), 4),
        "median_p_value": round(statistics.median(p_values), 4),
        "std_p_value": round(statistics.stdev(p_values), 4) if len(p_values) > 1 else 0,
        "detection_rate": round(sum(predictions) / len(predictions), 4),
        "mean_score": round(statistics.mean(scores), 4),
    }


def compute_paired_deltas_zscore(original_results: list, attack_results: list) -> dict:
    """Compute paired z-score deltas for WLLM/SWEET."""
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


def compute_paired_deltas_pvalue(original_results: list, attack_results: list) -> dict:
    """Compute paired p-value deltas for EXP-edit."""
    orig_by_idx = {r["sample_idx"]: r for r in original_results}
    deltas_p = []
    deltas_score = []
    for r in attack_results:
        idx = r["sample_idx"]
        if idx in orig_by_idx:
            deltas_p.append(r["p_value"] - orig_by_idx[idx]["p_value"])
            deltas_score.append(
                r.get("true_key_score", 0) - orig_by_idx[idx].get("true_key_score", 0)
            )

    if not deltas_p:
        return {"n_paired": 0, "mean_delta_p": 0, "median_delta_p": 0}

    return {
        "n_paired": len(deltas_p),
        "mean_delta_p": round(statistics.mean(deltas_p), 4),
        "median_delta_p": round(statistics.median(deltas_p), 4),
        "mean_delta_score": round(statistics.mean(deltas_score), 4),
        "pct_increased_p": round(sum(1 for d in deltas_p if d > 0) / len(deltas_p), 4),
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_results_table_zscore(
    all_metrics: dict, paired_deltas: dict, detector_name: str
):
    """Print results table for z-score based detectors (WLLM/SWEET)."""
    # Build header
    extra_col = ""
    if detector_name == "SWEET":
        extra_col = " | {'WM Frac':>8}"

    header = (
        f"{'Attack':<28} | {'N':>4} | {'Mean Z':>8} | {'Med Z':>8} | "
        f"{'Det Rate':>9} | {'Green Fr':>9}"
    )
    if detector_name == "SWEET":
        header += f" | {'WM Frac':>8}"
    header += f" | {'Delta Z':>8}"

    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"Watermark Detection Robustness Results ({detector_name})")
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

        line = (
            f"{attack_name:<28} | {m['n_samples']:>4} | {m['mean_z_score']:>8.4f} | "
            f"{m['median_z_score']:>8.4f} | {m['detection_rate']:>9.4f} | "
            f"{m['mean_green_fraction']:>9.4f}"
        )
        if detector_name == "SWEET":
            wm_frac = m.get("mean_watermarking_fraction", 0)
            line += f" | {wm_frac:>8.4f}"
        line += f" | {delta_str}"
        print(line)

    print(sep)


def print_results_table_pvalue(all_metrics: dict, paired_deltas: dict):
    """Print results table for p-value based detectors (EXP-edit)."""
    header = (
        f"{'Attack':<28} | {'N':>4} | {'Mean p':>8} | {'Med p':>8} | "
        f"{'Det Rate':>9} | {'Mean Scr':>9} | {'Delta p':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("Watermark Detection Robustness Results (EXP-edit)")
    print(sep)
    print(header)
    print(sep)

    for attack_name in ["original"] + sorted(
        k for k in all_metrics if k != "original"
    ):
        m = all_metrics.get(attack_name)
        if not m or m["n_samples"] == 0:
            continue
        delta_p = paired_deltas.get(attack_name, {}).get("mean_delta_p", 0)
        if attack_name == "original":
            delta_str = "     ---"
        else:
            delta_str = f"{delta_p:>+8.4f}"

        print(
            f"{attack_name:<28} | {m['n_samples']:>4} | {m['mean_p_value']:>8.4f} | "
            f"{m['median_p_value']:>8.4f} | {m['detection_rate']:>9.4f} | "
            f"{m['mean_score']:>9.4f} | {delta_str}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Determine which attacks to run
    if "all" in args.attacks:
        attack_names = [k for k in ALL_ATTACKS.keys() if k != "llm_paraphrase"]
        # Include llm_paraphrase only if OPENAI_API_KEY is set
        if os.environ.get("OPENAI_API_KEY"):
            attack_names.append("llm_paraphrase")
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

    # Load model for SWEET detector (entropy computation)
    model = None
    if args.detector == "sweet":
        if is_fallback:
            print("ERROR: SWEET detector requires the HuggingFace tokenizer and model.")
            print("       Cannot use fallback byte-level tokenizer with SWEET.")
            sys.exit(1)
        model = load_model(args.model, args.device, args.precision)

    # Load generations
    print(f"Loading generations from: {args.generations_path}")
    with open(args.generations_path) as f:
        generations = json.load(f)

    if args.limit:
        generations = generations[: args.limit]
    print(f"Loaded {len(generations)} samples")

    # Initialize detector
    vocab = list(tokenizer.get_vocab().values())

    if args.detector == "wllm":
        detector = WatermarkDetector(
            vocab=vocab,
            gamma=args.gamma,
            tokenizer=tokenizer,
            z_threshold=args.z_threshold,
            hash_key=args.hash_key,
        )
        detect_fn = lambda code, prompt: run_wllm_detection(
            detector, tokenizer, code, prompt
        )
        print(
            f"Initialized WLLM detector (gamma={args.gamma}, "
            f"z_threshold={args.z_threshold}, vocab_size={len(vocab)})"
        )

    elif args.detector == "sweet":
        from sweet import SweetDetector

        detector = SweetDetector(
            vocab=vocab,
            gamma=args.gamma,
            tokenizer=tokenizer,
            z_threshold=args.z_threshold,
            entropy_threshold=args.entropy_threshold,
            hash_key=args.hash_key,
        )
        detect_fn = lambda code, prompt: run_sweet_detection(
            detector, tokenizer, model, code, prompt, args.device
        )
        print(
            f"Initialized SWEET detector (gamma={args.gamma}, "
            f"z_threshold={args.z_threshold}, entropy_threshold={args.entropy_threshold}, "
            f"vocab_size={len(vocab)})"
        )

    elif args.detector == "exp":
        from exp import EXPDetector

        detector = EXPDetector(
            vocab=vocab,
            n=args.key_length,
            k=args.block_size,
            hash_key=args.hash_key,
            detection_p_threshold=args.p_threshold,
        )
        detect_fn = lambda code, prompt: run_exp_detection(
            detector, tokenizer, code, prompt, args.n_runs
        )
        print(
            f"Initialized EXP-edit detector (key_length={args.key_length}, "
            f"block_size={args.block_size}, n_runs={args.n_runs}, "
            f"p_threshold={args.p_threshold}, vocab_size={len(vocab)})"
        )

    # Choose metric functions based on detector type
    is_zscore = args.detector in ("wllm", "sweet")

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

        result = detect_fn(code, prompt)
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

            result = detect_fn(attacked_code, attacked_prompt)
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
        if is_zscore:
            all_metrics[attack_name] = compute_metrics_zscore(results)
        else:
            all_metrics[attack_name] = compute_metrics_pvalue(results)

        if attack_name != "original":
            if is_zscore:
                paired_deltas[attack_name] = compute_paired_deltas_zscore(
                    results_by_attack["original"], results
                )
            else:
                paired_deltas[attack_name] = compute_paired_deltas_pvalue(
                    results_by_attack["original"], results
                )

    # Print results
    if is_zscore:
        detector_label = "WLLM" if args.detector == "wllm" else "SWEET"
        print_results_table_zscore(all_metrics, paired_deltas, detector_label)
    else:
        print_results_table_pvalue(all_metrics, paired_deltas)

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
            "detector": args.detector,
            "model": args.model,
            "gamma": args.gamma if is_zscore else None,
            "z_threshold": args.z_threshold if is_zscore else None,
            "entropy_threshold": args.entropy_threshold if args.detector == "sweet" else None,
            "key_length": args.key_length if args.detector == "exp" else None,
            "block_size": args.block_size if args.detector == "exp" else None,
            "n_runs": args.n_runs if args.detector == "exp" else None,
            "p_threshold": args.p_threshold if args.detector == "exp" else None,
            "n_generations": len(generations),
            "attacks": attack_names,
            "tokenizer": "fallback_byte_level" if is_fallback else args.model,
        },
        "metrics": all_metrics,
        "paired_deltas": paired_deltas,
        "raw_results": {
            k: [
                {key: val for key, val in r.items() if key != "fake_key_scores"}
                for r in v
            ]
            for k, v in results_by_attack.items()
        },
    }

    # Remove None values from config
    output["config"] = {k: v for k, v in output["config"].items() if v is not None}

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
