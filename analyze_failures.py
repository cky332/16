"""
Failure Case Analysis for SWEET Code Watermarking
==================================================
Systematically analyze existing evaluation results to categorize and understand
watermark detection failure patterns.

Key finding: z-score = sqrt(T) when green_fraction=1.0 and gamma=0.5,
so at least 16 scored tokens are needed to reach z=4 detection threshold.
"""

import json
import os
import argparse
from collections import defaultdict, Counter

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_detection_results(results, task='humaneval'):
    return results[task]['watermark_detection']['raw_detection_results']


def bucket_analysis(detection_results):
    """Analyze detection results by scored tokens buckets."""
    buckets = {
        '0-2': [], '3-5': [], '6-10': [], '11-15': [], '16-20': [], '21+': []
    }

    for r in detection_results:
        t = r['num_tokens_scored']
        if t <= 2:
            buckets['0-2'].append(r)
        elif t <= 5:
            buckets['3-5'].append(r)
        elif t <= 10:
            buckets['6-10'].append(r)
        elif t <= 15:
            buckets['11-15'].append(r)
        elif t <= 20:
            buckets['16-20'].append(r)
        else:
            buckets['21+'].append(r)

    print("\n" + "=" * 70)
    print("SCORED TOKENS BUCKET ANALYSIS")
    print("=" * 70)
    print(f"{'Bucket':<10} {'Count':<8} {'Detected':<10} {'Det.Rate':<10} {'Avg z':<10} {'Avg GF':<10}")
    print("-" * 70)

    for name, items in buckets.items():
        if not items:
            print(f"{name:<10} {0:<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue
        count = len(items)
        detected = sum(1 for r in items if r.get('prediction', r.get('z_score', 0) >= 4))
        det_rate = detected / count if count > 0 else 0
        avg_z = np.mean([r['z_score'] for r in items])
        avg_gf = np.mean([r['green_fraction'] for r in items])
        print(f"{name:<10} {count:<8} {detected:<10} {det_rate:<10.3f} {avg_z:<10.2f} {avg_gf:<10.3f}")

    return buckets


def generation_length_analysis(detection_results):
    """Analyze by total generated token length."""
    length_buckets = {
        '1-5': [], '6-20': [], '21-50': [], '51-100': [], '100+': []
    }

    for r in detection_results:
        n = r['num_tokens_generated']
        if n <= 5:
            length_buckets['1-5'].append(r)
        elif n <= 20:
            length_buckets['6-20'].append(r)
        elif n <= 50:
            length_buckets['21-50'].append(r)
        elif n <= 100:
            length_buckets['51-100'].append(r)
        else:
            length_buckets['100+'].append(r)

    print("\n" + "=" * 70)
    print("GENERATION LENGTH ANALYSIS")
    print("=" * 70)
    print(f"{'Length':<10} {'Count':<8} {'Detected':<10} {'Det.Rate':<10} {'Avg WF':<10} {'Avg Scored':<12}")
    print("-" * 70)

    for name, items in length_buckets.items():
        if not items:
            print(f"{name:<10} {0:<8} {'N/A':<10}")
            continue
        count = len(items)
        detected = sum(1 for r in items if r.get('prediction', r.get('z_score', 0) >= 4))
        det_rate = detected / count if count > 0 else 0
        avg_wf = np.mean([r['watermarking_fraction'] for r in items])
        avg_scored = np.mean([r['num_tokens_scored'] for r in items])
        print(f"{name:<10} {count:<8} {detected:<10} {det_rate:<10.3f} {avg_wf:<10.3f} {avg_scored:<12.1f}")

    return length_buckets


def watermarking_fraction_analysis(detection_results):
    """Analyze by watermarking fraction (what % of tokens were watermarked)."""
    wf_buckets = {
        '0-5%': [], '5-10%': [], '10-20%': [], '20-50%': [], '50%+': []
    }

    for r in detection_results:
        wf = r['watermarking_fraction']
        if wf <= 0.05:
            wf_buckets['0-5%'].append(r)
        elif wf <= 0.10:
            wf_buckets['5-10%'].append(r)
        elif wf <= 0.20:
            wf_buckets['10-20%'].append(r)
        elif wf <= 0.50:
            wf_buckets['20-50%'].append(r)
        else:
            wf_buckets['50%+'].append(r)

    print("\n" + "=" * 70)
    print("WATERMARKING FRACTION ANALYSIS")
    print("=" * 70)
    print(f"{'WF Range':<10} {'Count':<8} {'Detected':<10} {'Det.Rate':<10} {'Avg z':<10} {'Avg GenLen':<12}")
    print("-" * 70)

    for name, items in wf_buckets.items():
        if not items:
            print(f"{name:<10} {0:<8} {'N/A':<10}")
            continue
        count = len(items)
        detected = sum(1 for r in items if r.get('prediction', r.get('z_score', 0) >= 4))
        det_rate = detected / count if count > 0 else 0
        avg_z = np.mean([r['z_score'] for r in items])
        avg_gen = np.mean([r['num_tokens_generated'] for r in items])
        print(f"{name:<10} {count:<8} {detected:<10} {det_rate:<10.3f} {avg_z:<10.2f} {avg_gen:<12.1f}")

    return wf_buckets


def green_fraction_paradox(detection_results):
    """Identify samples with high green_fraction but low z-score (the paradox)."""
    print("\n" + "=" * 70)
    print("GREEN FRACTION PARADOX ANALYSIS")
    print("(Samples with green_fraction=1.0 but NOT detected)")
    print("=" * 70)

    paradox_cases = [r for r in detection_results
                     if r['green_fraction'] == 1.0 and r.get('z_score', 0) < 4]
    detected_gf1 = [r for r in detection_results
                    if r['green_fraction'] == 1.0 and r.get('z_score', 0) >= 4]
    other = [r for r in detection_results if r['green_fraction'] < 1.0]

    print(f"\nTotal samples: {len(detection_results)}")
    print(f"green_fraction=1.0 but NOT detected: {len(paradox_cases)} ({100*len(paradox_cases)/len(detection_results):.1f}%)")
    print(f"green_fraction=1.0 and detected: {len(detected_gf1)} ({100*len(detected_gf1)/len(detection_results):.1f}%)")
    print(f"green_fraction < 1.0: {len(other)} ({100*len(other)/len(detection_results):.1f}%)")

    if paradox_cases:
        scored_tokens = [r['num_tokens_scored'] for r in paradox_cases]
        print(f"\nParadox cases scored tokens: min={min(scored_tokens)}, max={max(scored_tokens)}, "
              f"mean={np.mean(scored_tokens):.1f}, median={np.median(scored_tokens):.1f}")
        print(f"Z-scores: min={min(r['z_score'] for r in paradox_cases):.2f}, "
              f"max={max(r['z_score'] for r in paradox_cases):.2f}")
        print(f"\nMathematical explanation: z = sqrt(T) when GF=1.0 and gamma=0.5")
        print(f"Need T >= 16 for z >= 4. In these {len(paradox_cases)} cases, all have T < 16.")

    return paradox_cases


def overall_statistics(detection_results, label="SWEET"):
    """Print overall statistics summary."""
    print("\n" + "=" * 70)
    print(f"OVERALL STATISTICS ({label})")
    print("=" * 70)

    total = len(detection_results)
    detected = sum(1 for r in detection_results if r.get('prediction', r.get('z_score', 0) >= 4))

    z_scores = [r['z_score'] for r in detection_results]
    gen_lens = [r['num_tokens_generated'] for r in detection_results]
    scored = [r['num_tokens_scored'] for r in detection_results]
    wfs = [r['watermarking_fraction'] for r in detection_results]
    gfs = [r['green_fraction'] for r in detection_results]

    print(f"Total samples: {total}")
    print(f"Detected: {detected} ({100*detected/total:.1f}%)")
    print(f"NOT detected: {total - detected} ({100*(total-detected)/total:.1f}%)")
    print(f"\nZ-score: mean={np.mean(z_scores):.2f}, std={np.std(z_scores):.2f}, "
          f"min={min(z_scores):.2f}, max={max(z_scores):.2f}, median={np.median(z_scores):.2f}")
    print(f"Generated tokens: mean={np.mean(gen_lens):.1f}, std={np.std(gen_lens):.1f}, "
          f"min={min(gen_lens)}, max={max(gen_lens)}")
    print(f"Scored tokens: mean={np.mean(scored):.1f}, std={np.std(scored):.1f}, "
          f"min={min(scored)}, max={max(scored)}")
    print(f"Watermarking fraction: mean={np.mean(wfs):.3f}, std={np.std(wfs):.3f}")
    print(f"Green fraction: mean={np.mean(gfs):.3f}, std={np.std(gfs):.3f}")

    # Zero-scored analysis
    zero_scored = sum(1 for s in scored if s == 0)
    print(f"\nSamples with 0 scored tokens: {zero_scored}")


def failure_case_list(detection_results, top_n=20):
    """List the most extreme failure cases."""
    print("\n" + "=" * 70)
    print(f"TOP {top_n} WORST FAILURE CASES (watermarked but lowest z-scores)")
    print("=" * 70)

    sorted_results = sorted(enumerate(detection_results), key=lambda x: x[1]['z_score'])

    print(f"{'Idx':<5} {'GenLen':<8} {'Scored':<8} {'GreenTk':<8} {'WF':<8} {'GF':<8} {'z-score':<10}")
    print("-" * 70)

    for idx, r in sorted_results[:top_n]:
        print(f"{idx:<5} {r['num_tokens_generated']:<8} {r['num_tokens_scored']:<8} "
              f"{r['num_green_tokens']:<8} {r['watermarking_fraction']:<8.3f} "
              f"{r['green_fraction']:<8.3f} {r['z_score']:<10.3f}")


def compare_human_vs_machine(human_results, machine_results, task='humaneval'):
    """Compare human code detection vs machine-generated code detection."""
    print("\n" + "=" * 70)
    print("HUMAN vs MACHINE COMPARISON")
    print("=" * 70)

    human_det = get_detection_results(human_results, task)
    machine_det = get_detection_results(machine_results, task)

    human_z = [r['z_score'] for r in human_det]
    machine_z = [r['z_score'] for r in machine_det]

    print(f"\n{'Metric':<25} {'Human':<15} {'Machine':<15}")
    print("-" * 55)
    print(f"{'Samples':<25} {len(human_z):<15} {len(machine_z):<15}")
    print(f"{'Mean z-score':<25} {np.mean(human_z):<15.3f} {np.mean(machine_z):<15.3f}")
    print(f"{'Std z-score':<25} {np.std(human_z):<15.3f} {np.std(machine_z):<15.3f}")
    print(f"{'Min z-score':<25} {min(human_z):<15.3f} {min(machine_z):<15.3f}")
    print(f"{'Max z-score':<25} {max(human_z):<15.3f} {max(machine_z):<15.3f}")
    print(f"{'Detected (z>=4)':<25} {sum(1 for z in human_z if z >= 4):<15} {sum(1 for z in machine_z if z >= 4):<15}")

    # Overlap analysis
    human_above_2 = sum(1 for z in human_z if z >= 2)
    machine_below_2 = sum(1 for z in machine_z if z < 2)
    print(f"\n{'Human z >= 2 (FP risk)':<25} {human_above_2} ({100*human_above_2/len(human_z):.1f}%)")
    print(f"{'Machine z < 2 (FN risk)':<25} {machine_below_2} ({100*machine_below_2/len(machine_z):.1f}%)")


def generate_visualizations(detection_results, output_dir='failure_analysis_plots'):
    """Generate visualization plots if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available, skipping visualizations.")
        return

    os.makedirs(output_dir, exist_ok=True)

    z_scores = [r['z_score'] for r in detection_results]
    scored_tokens = [r['num_tokens_scored'] for r in detection_results]
    gen_lens = [r['num_tokens_generated'] for r in detection_results]
    wfs = [r['watermarking_fraction'] for r in detection_results]
    gfs = [r['green_fraction'] for r in detection_results]

    # Plot 1: z-score vs scored tokens scatter
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['red' if z < 4 else 'green' for z in z_scores]
    ax.scatter(scored_tokens, z_scores, c=colors, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.axhline(y=4, color='blue', linestyle='--', label='z=4 threshold')
    ax.axvline(x=16, color='orange', linestyle='--', label='T=16 (min for detection)')

    # Theoretical curve: z = sqrt(T) when GF=1.0
    t_range = np.arange(1, max(scored_tokens) + 1)
    ax.plot(t_range, np.sqrt(t_range), 'k--', alpha=0.3, label='z=sqrt(T) theoretical')

    ax.set_xlabel('Number of Scored Tokens (T)')
    ax.set_ylabel('Z-Score')
    ax.set_title('Z-Score vs Scored Tokens (Green=Detected, Red=Not Detected)')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'zscore_vs_scored_tokens.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Watermarking fraction histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(wfs, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Watermarking Fraction')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Watermarking Fraction')
    ax.axvline(x=np.mean(wfs), color='red', linestyle='--', label=f'Mean={np.mean(wfs):.3f}')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'watermarking_fraction_dist.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: Z-score distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(z_scores, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=4, color='red', linestyle='--', label='z=4 threshold')
    ax.set_xlabel('Z-Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Z-Scores')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'zscore_dist.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 4: Detection rate by scored tokens threshold
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    thresholds = range(0, max(scored_tokens) + 1)
    det_rates = []
    sample_counts = []
    for thresh in thresholds:
        subset = [r for r in detection_results if r['num_tokens_scored'] >= thresh]
        if subset:
            det_rate = sum(1 for r in subset if r.get('z_score', 0) >= 4) / len(subset)
            det_rates.append(det_rate)
            sample_counts.append(len(subset))
        else:
            det_rates.append(0)
            sample_counts.append(0)

    ax.plot(list(thresholds), det_rates, 'b-', linewidth=2)
    ax.set_xlabel('Minimum Scored Tokens Threshold')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate vs Minimum Scored Tokens')
    ax.axvline(x=16, color='orange', linestyle='--', label='T=16')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'detection_rate_by_threshold.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 5: Generated length vs scored tokens
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(gen_lens, scored_tokens, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Generated Tokens (total)')
    ax.set_ylabel('Scored Tokens')
    ax.set_title('Total Generated vs Scored Tokens')

    # Add diagonal reference
    max_val = max(max(gen_lens), max(scored_tokens))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3, label='1:1 line')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'gen_len_vs_scored.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nVisualizations saved to {output_dir}/")


def export_csv(detection_results, output_path='failure_analysis.csv'):
    """Export all detection results to CSV for further analysis."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_idx', 'num_tokens_generated', 'num_tokens_scored',
            'num_green_tokens', 'watermarking_fraction', 'green_fraction',
            'z_score', 'p_value', 'detected'
        ])
        writer.writeheader()

        for idx, r in enumerate(detection_results):
            writer.writerow({
                'sample_idx': idx,
                'num_tokens_generated': r['num_tokens_generated'],
                'num_tokens_scored': r['num_tokens_scored'],
                'num_green_tokens': r['num_green_tokens'],
                'watermarking_fraction': r['watermarking_fraction'],
                'green_fraction': r['green_fraction'],
                'z_score': r['z_score'],
                'p_value': r['p_value'],
                'detected': 1 if r.get('prediction', r.get('z_score', 0) >= 4) else 0
            })

    print(f"\nCSV exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SWEET watermark failure cases")
    parser.add_argument('--task', type=str, default='humaneval', help='Task name')
    parser.add_argument('--sweet_results', type=str, default='outputs_sweet/evaluation_results.json')
    parser.add_argument('--baseline_results', type=str, default='outputs/evaluation_results.json')
    parser.add_argument('--human_results', type=str, default='outputs_sweet_human/evaluation_results.json')
    parser.add_argument('--output_dir', type=str, default='failure_analysis_plots')
    parser.add_argument('--csv_output', type=str, default='failure_analysis.csv')
    args = parser.parse_args()

    # Load results
    print("Loading results...")
    sweet_results = load_results(args.sweet_results)
    sweet_det = get_detection_results(sweet_results, args.task)

    # Overall statistics
    overall_statistics(sweet_det, label="SWEET watermark")

    # Bucket analyses
    bucket_analysis(sweet_det)
    generation_length_analysis(sweet_det)
    watermarking_fraction_analysis(sweet_det)

    # Paradox analysis
    green_fraction_paradox(sweet_det)

    # Failure case list
    failure_case_list(sweet_det)

    # Load and compare with baseline and human
    if os.path.exists(args.baseline_results):
        baseline_results = load_results(args.baseline_results)
        baseline_det = get_detection_results(baseline_results, args.task)
        print("\n")
        overall_statistics(baseline_det, label="Baseline (no watermark)")

    if os.path.exists(args.human_results):
        human_results = load_results(args.human_results)
        compare_human_vs_machine(human_results, sweet_results, args.task)

    # Visualizations
    generate_visualizations(sweet_det, args.output_dir)

    # Export CSV
    export_csv(sweet_det, args.csv_output)

    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
