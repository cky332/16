"""
Cross-Experiment Result Summarizer
====================================
Collects and compares results across all failure case experiments:
- Multi-language (MultiPL-E)
- New tasks (CoNaLa, APPS, DS-1000 sub-libs)
- Robustness (code transformations)
- Parameter sensitivity sweeps

Produces a unified comparison table and identifies key failure patterns.
"""

import json
import os
import glob
import argparse
from collections import defaultdict

import numpy as np


def load_eval_results(path):
    """Load evaluation_results.json if it exists."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def extract_metrics(results, task=None):
    """Extract key metrics from evaluation results."""
    if results is None:
        return None

    # Auto-detect task name
    if task is None:
        task = list(results.keys())[0]

    task_results = results.get(task, {})
    wd = task_results.get('watermark_detection', {})

    if not wd or 'raw_detection_results' not in wd:
        return None

    raw = wd['raw_detection_results']
    z_scores = [r['z_score'] for r in raw]
    scored = [r['num_tokens_scored'] for r in raw]
    gen_lens = [r['num_tokens_generated'] for r in raw]

    wfs = [r.get('watermarking_fraction', 0) for r in raw]
    gfs = [r.get('green_fraction', 0) for r in raw]

    detected = sum(1 for z in z_scores if z >= 4)
    total = len(z_scores)

    metrics = {
        'task': task,
        'total_samples': total,
        'detected': detected,
        'detection_rate': detected / total if total > 0 else 0,
        'mean_z_score': np.mean(z_scores),
        'mean_scored_tokens': np.mean(scored),
        'mean_gen_len': np.mean(gen_lens),
        'mean_watermarking_fraction': np.mean(wfs) if wfs else 0,
        'mean_green_fraction': np.mean(gfs) if gfs else 0,
        'pct_below_16_scored': sum(1 for s in scored if s < 16) / total if total > 0 else 0,
    }

    # Add pass@k if available
    if 'pass@1' in task_results:
        metrics['pass@1'] = task_results['pass@1']

    # Add AUROC if available
    if 'roc_auc' in wd:
        metrics['roc_auc'] = wd['roc_auc']

    return metrics


def scan_output_directories(base_dir='.'):
    """Scan for all output directories with evaluation results."""
    results = []

    # Pattern: outputs*/evaluation_results.json
    for eval_file in glob.glob(os.path.join(base_dir, 'outputs*/evaluation_results.json')):
        dir_name = os.path.dirname(eval_file)
        data = load_eval_results(eval_file)
        if data:
            for task_name in data.keys():
                metrics = extract_metrics(data, task_name)
                if metrics:
                    metrics['experiment'] = os.path.basename(dir_name)
                    results.append(metrics)

    # Also scan nested dirs: outputs/sweet_*/evaluation_results.json
    for eval_file in glob.glob(os.path.join(base_dir, 'outputs/*/evaluation_results.json')):
        dir_name = os.path.dirname(eval_file)
        data = load_eval_results(eval_file)
        if data:
            for task_name in data.keys():
                metrics = extract_metrics(data, task_name)
                if metrics:
                    metrics['experiment'] = os.path.basename(dir_name)
                    results.append(metrics)

    return results


def print_comparison_table(all_results):
    """Print a formatted comparison table of all experiments."""
    if not all_results:
        print("No results found.")
        return

    print("\n" + "=" * 120)
    print("CROSS-EXPERIMENT COMPARISON TABLE")
    print("=" * 120)

    header = (f"{'Experiment':<30} {'Task':<25} {'Samples':<8} {'Det.':<6} "
              f"{'Rate':<7} {'Avg_z':<8} {'Avg_Scored':<12} {'WF':<8} {'%T<16':<8} {'pass@1':<8}")
    print(header)
    print("-" * 120)

    for m in sorted(all_results, key=lambda x: (x['experiment'], x['task'])):
        pass_str = f"{m.get('pass@1', 'N/A'):.3f}" if isinstance(m.get('pass@1'), float) else 'N/A'
        print(f"{m['experiment']:<30} {m['task']:<25} {m['total_samples']:<8} {m['detected']:<6} "
              f"{m['detection_rate']:<7.3f} {m['mean_z_score']:<8.2f} {m['mean_scored_tokens']:<12.1f} "
              f"{m['mean_watermarking_fraction']:<8.3f} {m['pct_below_16_scored']:<8.3f} {pass_str:<8}")


def identify_failure_patterns(all_results):
    """Identify and categorize failure patterns across experiments."""
    print("\n" + "=" * 120)
    print("IDENTIFIED FAILURE PATTERNS")
    print("=" * 120)

    patterns = defaultdict(list)

    for m in all_results:
        if m['detection_rate'] < 0.1:
            patterns['near_zero_detection'].append(m)
        if m['mean_watermarking_fraction'] < 0.1:
            patterns['very_low_wf'].append(m)
        if m['pct_below_16_scored'] > 0.8:
            patterns['mostly_undetectable'].append(m)
        if m['mean_scored_tokens'] < 5:
            patterns['extremely_short_scored'].append(m)

    for pattern_name, items in patterns.items():
        if items:
            print(f"\n--- {pattern_name} ({len(items)} experiments) ---")
            for m in items:
                print(f"  {m['experiment']}/{m['task']}: det_rate={m['detection_rate']:.3f}, "
                      f"wf={m['mean_watermarking_fraction']:.3f}, avg_scored={m['mean_scored_tokens']:.1f}")

    # Summary statistics
    print("\n" + "-" * 120)
    print("FAILURE PATTERN SUMMARY")
    print("-" * 120)

    if all_results:
        det_rates = [m['detection_rate'] for m in all_results]
        wfs = [m['mean_watermarking_fraction'] for m in all_results]
        print(f"Detection rate range: [{min(det_rates):.3f}, {max(det_rates):.3f}]")
        print(f"Watermarking fraction range: [{min(wfs):.3f}, {max(wfs):.3f}]")
        print(f"Experiments with det_rate < 10%: {sum(1 for d in det_rates if d < 0.1)}/{len(det_rates)}")
        print(f"Experiments with det_rate > 50%: {sum(1 for d in det_rates if d > 0.5)}/{len(det_rates)}")


def generate_latex_table(all_results, output_path='results_table.tex'):
    """Generate a LaTeX table for paper inclusion."""
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Watermark Detection Results Across Tasks and Settings}\n")
        f.write("\\begin{tabular}{l|ccccc}\n")
        f.write("\\toprule\n")
        f.write("Experiment & Samples & Det. Rate & Avg z & WF & \\%T<16 \\\\\n")
        f.write("\\midrule\n")

        for m in sorted(all_results, key=lambda x: -x['detection_rate']):
            exp_name = m['experiment'].replace('_', '\\_')
            f.write(f"{exp_name} & {m['total_samples']} & {m['detection_rate']:.3f} & "
                    f"{m['mean_z_score']:.2f} & {m['mean_watermarking_fraction']:.3f} & "
                    f"{m['pct_below_16_scored']:.3f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table saved to {output_path}")


def export_json_summary(all_results, output_path='results_summary.json'):
    """Export all results as JSON for programmatic analysis."""
    # Convert numpy types to Python types
    serializable = []
    for m in all_results:
        clean = {}
        for k, v in m.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            else:
                clean[k] = v
        serializable.append(clean)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"JSON summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize all failure case experiment results")
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory containing output directories')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX table')
    parser.add_argument('--json_output', type=str, default='results_summary.json')
    args = parser.parse_args()

    print("Scanning for experiment results...")
    all_results = scan_output_directories(args.base_dir)

    if not all_results:
        print("No evaluation results found. Run experiments first.")
        print("Available scripts:")
        print("  bash scripts/main/run_sweet_generation.sh     (HumanEval)")
        print("  bash scripts/main/run_sweet_multilang.sh      (Multi-language)")
        print("  bash scripts/main/run_sweet_newtasks.sh       (New tasks)")
        print("  bash scripts/main/run_sweet_robustness.sh     (Robustness)")
        print("  bash scripts/main/run_sweet_param_sweep.sh    (Parameter sweep)")
        return

    print(f"Found {len(all_results)} experiment results.")

    # Print comparison table
    print_comparison_table(all_results)

    # Identify failure patterns
    identify_failure_patterns(all_results)

    # Export
    export_json_summary(all_results, args.json_output)

    if args.latex:
        generate_latex_table(all_results)

    print("\n" + "=" * 120)
    print("SUMMARY COMPLETE")
    print("=" * 120)


if __name__ == '__main__':
    main()
