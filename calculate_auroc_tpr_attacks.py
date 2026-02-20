#!/usr/bin/env python3
"""
Calculate AUROC and TPR for each attack in the robustness results.

This script bridges between:
  - test_watermark_detection.py output (robustness_results_*.json)
  - calculate_auroc_tpr.py logic (AUROC/TPR computation)

It reads attacked code z-scores from the robustness results file and
human code z-scores from the human baseline file, then computes AUROC
and TPR@FPR for each attack.

Usage:
    python calculate_auroc_tpr_attacks.py \
        --task humaneval \
        --human_fname outputs_sweet_human/evaluation_results.json \
        --robustness_fname outputs/robustness_results_sweet.json \
        --output_fname outputs/auroc_robustness_sweet.json
"""

import argparse
import json
import os

import numpy as np
import sklearn.metrics as metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate AUROC and TPR for watermark robustness results"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="humaneval",
        help="Task name used in human results file (default: humaneval)",
    )
    parser.add_argument(
        "--human_fname",
        type=str,
        required=True,
        help="Path to human code detection results (evaluation_results.json)",
    )
    parser.add_argument(
        "--robustness_fname",
        type=str,
        required=True,
        help="Path to robustness results from test_watermark_detection.py",
    )
    parser.add_argument(
        "--output_fname",
        type=str,
        default=None,
        help="Path to save output (default: derived from robustness_fname)",
    )
    return parser.parse_args()


def get_roc_auc(human_z, machine_z):
    """Compute AUROC from human (negative) and machine (positive) z-scores."""
    baseline_z_scores = np.array(human_z)
    watermark_z_scores = np.array(machine_z)
    all_scores = np.concatenate([baseline_z_scores, watermark_z_scores])

    baseline_labels = np.zeros_like(baseline_z_scores)
    watermarked_labels = np.ones_like(watermark_z_scores)
    all_labels = np.concatenate([baseline_labels, watermarked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc, fpr, tpr, thresholds


def get_tpr(fpr, tpr, error_rate):
    """Get TPR at a specific FPR threshold."""
    value = None
    for f, t in zip(fpr, tpr):
        if f <= error_rate:
            value = t
        else:
            assert value is not None
            return value

    assert value == 1.0
    return value


def main():
    args = parse_args()

    # Load human baseline z-scores
    with open(args.human_fname) as f:
        human_results = json.load(f)

    human_z = [
        r["z_score"]
        for r in human_results[args.task]["watermark_detection"]["raw_detection_results"]
    ]
    print(f"Loaded {len(human_z)} human z-scores from {args.human_fname}")

    # Load robustness results
    with open(args.robustness_fname) as f:
        robustness = json.load(f)

    raw_results = robustness["raw_results"]
    config = robustness["config"]
    print(f"Loaded robustness results from {args.robustness_fname}")
    print(f"Detector: {config['detector']}, Attacks: {len(raw_results) - 1}")

    # Compute AUROC and TPR for each attack
    auroc_results = {}
    attack_order = ["original"] + sorted(k for k in raw_results if k != "original")

    print(f"\n{'='*85}")
    print(f"AUROC and TPR Results ({config['detector'].upper()})")
    print(f"{'='*85}")
    print(
        f"{'Attack':<28} | {'N_m':>4} | {'N_h':>4} | {'AUROC':>7} | "
        f"{'TPR@0%':>7} | {'TPR@1%':>7} | {'TPR@5%':>7}"
    )
    print("-" * 85)

    for attack_name in attack_order:
        results = raw_results[attack_name]
        machine_z = [r["z_score"] for r in results]

        n_machine = len(machine_z)
        n_human = len(human_z)

        # Handle length mismatch by truncating to the shorter one
        n_min = min(n_machine, n_human)
        if n_machine != n_human:
            print(
                f"  Warning: {attack_name} has {n_machine} machine samples "
                f"vs {n_human} human samples. Using first {n_min} of each."
            )
            machine_z_use = machine_z[:n_min]
            human_z_use = human_z[:n_min]
        else:
            machine_z_use = machine_z
            human_z_use = human_z

        roc_auc, fpr, tpr, _ = get_roc_auc(human_z_use, machine_z_use)
        tpr_0 = get_tpr(fpr, tpr, 0.0)
        tpr_1 = get_tpr(fpr, tpr, 0.01)
        tpr_5 = get_tpr(fpr, tpr, 0.05)

        auroc_results[attack_name] = {
            "n_machine_samples": n_machine,
            "n_human_samples": n_human,
            "roc_auc": round(roc_auc, 4),
            "TPR (FPR = 0%)": round(tpr_0, 4),
            "TPR (FPR < 1%)": round(tpr_1, 4),
            "TPR (FPR < 5%)": round(tpr_5, 4),
        }

        # Also include detection metrics from robustness results
        if attack_name in robustness.get("metrics", {}):
            auroc_results[attack_name]["mean_z_score"] = robustness["metrics"][attack_name].get("mean_z_score")
            auroc_results[attack_name]["detection_rate_z4"] = robustness["metrics"][attack_name].get("detection_rate")

        print(
            f"{attack_name:<28} | {n_machine:>4} | {n_human:>4} | "
            f"{roc_auc:>7.4f} | {tpr_0:>7.4f} | {tpr_1:>7.4f} | {tpr_5:>7.4f}"
        )

    print(f"{'='*85}")

    # Compute delta relative to original
    if "original" in auroc_results:
        orig_auc = auroc_results["original"]["roc_auc"]
        print(f"\n{'Attack':<28} | {'Delta AUROC':>11}")
        print("-" * 45)
        for attack_name in attack_order:
            if attack_name == "original":
                print(f"{attack_name:<28} |         ---")
            else:
                delta = auroc_results[attack_name]["roc_auc"] - orig_auc
                print(f"{attack_name:<28} | {delta:>+11.4f}")
                auroc_results[attack_name]["delta_auroc"] = round(delta, 4)
        print("-" * 45)

    # Save results
    output = {
        "config": config,
        "human_baseline": args.human_fname,
        "robustness_source": args.robustness_fname,
        "auroc_results": auroc_results,
    }

    if args.output_fname is None:
        base = os.path.splitext(args.robustness_fname)[0]
        args.output_fname = f"{base}_auroc.json"

    os.makedirs(os.path.dirname(args.output_fname) or ".", exist_ok=True)
    with open(args.output_fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output_fname}")


if __name__ == "__main__":
    main()
