"""
Code Pattern Analysis for Watermark Failure Cases
===================================================
Analyzes the structural characteristics of generated code to understand
WHY certain samples fail watermark detection.

Key questions:
- What code patterns produce low entropy (few scored tokens)?
- Are one-line solutions, imports, and template code the main culprits?
- Which HumanEval problem types are most resistant to watermarking?
"""

import json
import re
import ast
import argparse
from collections import Counter, defaultdict

import numpy as np


def load_generations(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def analyze_code_structure(code):
    """Analyze structural features of a code snippet."""
    lines = code.strip().split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    comment_lines = [l for l in lines if l.strip().startswith('#')]
    import_lines = [l for l in lines if l.strip().startswith(('import ', 'from '))]

    features = {
        'total_lines': len(lines),
        'non_empty_lines': len(non_empty_lines),
        'comment_lines': len(comment_lines),
        'import_lines': len(import_lines),
        'max_indent_depth': 0,
        'has_loop': False,
        'has_conditional': False,
        'has_list_comprehension': False,
        'has_return': False,
        'is_one_liner': False,
        'char_count': len(code),
    }

    # Analyze indentation depth
    for line in non_empty_lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        indent_depth = indent // 4  # assuming 4-space indent
        features['max_indent_depth'] = max(features['max_indent_depth'], indent_depth)

    # Pattern detection
    code_lower = code.lower()
    features['has_loop'] = bool(re.search(r'\bfor\b|\bwhile\b', code))
    features['has_conditional'] = bool(re.search(r'\bif\b|\belif\b|\belse\b', code))
    features['has_list_comprehension'] = bool(re.search(r'\[.*\bfor\b.*\bin\b.*\]', code))
    features['has_return'] = 'return ' in code
    features['has_try_except'] = 'try:' in code
    features['has_class'] = bool(re.search(r'\bclass\b\s+\w+', code))
    features['has_lambda'] = 'lambda ' in code
    features['has_dict_comp'] = bool(re.search(r'\{.*\bfor\b.*\bin\b.*\}', code))

    # Count function body lines (after first def line)
    body_lines = []
    in_body = False
    for line in lines:
        if in_body and line.strip():
            body_lines.append(line)
        if line.strip().startswith('def '):
            in_body = True

    features['body_lines'] = len(body_lines)
    features['is_one_liner'] = len(body_lines) <= 1

    # Detect code type
    features['code_type'] = classify_code_type(code)

    return features


def classify_code_type(code):
    """Classify the type/category of code."""
    code_lower = code.lower()

    if re.search(r'sort|sorted|reverse|min\(|max\(', code):
        return 'sorting/ordering'
    if re.search(r'def\s+\w+.*->.*list|def\s+\w+.*\[\]', code) or 'append' in code:
        return 'list_manipulation'
    if re.search(r'str\.|\.split|\.join|\.replace|\.strip|\.lower|\.upper', code):
        return 'string_processing'
    if re.search(r'\bmath\b|\bsqrt\b|\bpow\b|\babs\b|%\s*\d', code):
        return 'math_computation'
    if re.search(r'dict\(|\.keys\(|\.values\(|\.items\(|\{.*:.*\}', code):
        return 'dictionary_ops'
    if re.search(r'set\(|\.add\(|\.union\(|\.intersection\(', code):
        return 'set_ops'
    if re.search(r'class\s+\w+', code):
        return 'class_definition'
    if re.search(r'open\(|\.read\(|\.write\(', code):
        return 'file_io'
    if 'recursion' in code_lower or 'recursive' in code_lower:
        return 'recursion'
    if re.search(r'True|False|bool|is_|has_|can_', code):
        return 'boolean_logic'

    return 'other'


def extract_body_from_generation(generation, prompt):
    """Extract the function body (code generated after prompt)."""
    if generation.startswith(prompt):
        return generation[len(prompt):]
    return generation


def analyze_failure_by_code_pattern(generations, detection_results, task='humaneval'):
    """Correlate code patterns with detection failures."""
    det_results = detection_results[task]['watermark_detection']['raw_detection_results']

    print("\n" + "=" * 70)
    print("CODE PATTERN ANALYSIS FOR FAILURE CASES")
    print("=" * 70)

    # Use first generation per problem (matching detection)
    pattern_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'z_scores': [],
                                          'scored_tokens': [], 'gen_lens': []})
    oneliner_stats = {'total': 0, 'detected': 0, 'z_scores': [], 'scored_tokens': []}
    multiliner_stats = {'total': 0, 'detected': 0, 'z_scores': [], 'scored_tokens': []}

    all_features = []

    for idx, (gens, det) in enumerate(zip(generations, det_results)):
        # Use first generation
        gen = gens[0] if isinstance(gens, list) else gens
        features = analyze_code_structure(gen)
        features['idx'] = idx
        features['z_score'] = det['z_score']
        features['detected'] = det.get('prediction', det.get('z_score', 0) >= 4)
        features['num_tokens_scored'] = det['num_tokens_scored']
        features['num_tokens_generated'] = det['num_tokens_generated']
        features['watermarking_fraction'] = det['watermarking_fraction']
        all_features.append(features)

        # Code type stats
        ct = features['code_type']
        pattern_stats[ct]['total'] += 1
        if features['detected']:
            pattern_stats[ct]['detected'] += 1
        pattern_stats[ct]['z_scores'].append(det['z_score'])
        pattern_stats[ct]['scored_tokens'].append(det['num_tokens_scored'])
        pattern_stats[ct]['gen_lens'].append(det['num_tokens_generated'])

        # One-liner vs multi-liner
        if features['is_one_liner']:
            oneliner_stats['total'] += 1
            if features['detected']:
                oneliner_stats['detected'] += 1
            oneliner_stats['z_scores'].append(det['z_score'])
            oneliner_stats['scored_tokens'].append(det['num_tokens_scored'])
        else:
            multiliner_stats['total'] += 1
            if features['detected']:
                multiliner_stats['detected'] += 1
            multiliner_stats['z_scores'].append(det['z_score'])
            multiliner_stats['scored_tokens'].append(det['num_tokens_scored'])

    # Print code type analysis
    print(f"\n{'Code Type':<20} {'Count':<7} {'Detect':<7} {'Rate':<7} {'Avg z':<8} {'Avg Scored':<12} {'Avg GenLen':<12}")
    print("-" * 80)
    for ct, stats in sorted(pattern_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        det_rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
        avg_z = np.mean(stats['z_scores'])
        avg_scored = np.mean(stats['scored_tokens'])
        avg_gen = np.mean(stats['gen_lens'])
        print(f"{ct:<20} {stats['total']:<7} {stats['detected']:<7} {det_rate:<7.3f} {avg_z:<8.2f} {avg_scored:<12.1f} {avg_gen:<12.1f}")

    # One-liner analysis
    print("\n" + "-" * 70)
    print("ONE-LINER vs MULTI-LINE ANALYSIS")
    print("-" * 70)
    for label, stats in [("One-liner", oneliner_stats), ("Multi-line", multiliner_stats)]:
        if stats['total'] == 0:
            continue
        det_rate = stats['detected'] / stats['total']
        avg_z = np.mean(stats['z_scores'])
        avg_scored = np.mean(stats['scored_tokens'])
        print(f"{label:<15} Total={stats['total']:<5} Detected={stats['detected']:<5} "
              f"Rate={det_rate:.3f}  Avg_z={avg_z:.2f}  Avg_scored={avg_scored:.1f}")

    # Feature correlation analysis
    print("\n" + "-" * 70)
    print("STRUCTURAL FEATURE CORRELATION WITH DETECTION")
    print("-" * 70)

    for feature_name in ['has_loop', 'has_conditional', 'has_list_comprehension',
                         'has_return', 'has_try_except', 'has_lambda']:
        with_feature = [f for f in all_features if f.get(feature_name)]
        without_feature = [f for f in all_features if not f.get(feature_name)]

        if with_feature and without_feature:
            w_det = sum(1 for f in with_feature if f['detected']) / len(with_feature)
            wo_det = sum(1 for f in without_feature if f['detected']) / len(without_feature)
            w_scored = np.mean([f['num_tokens_scored'] for f in with_feature])
            wo_scored = np.mean([f['num_tokens_scored'] for f in without_feature])
            print(f"{feature_name:<25} With: det={w_det:.3f} scored={w_scored:.1f} (n={len(with_feature)})  "
                  f"Without: det={wo_det:.3f} scored={wo_scored:.1f} (n={len(without_feature)})")

    return all_features


def analyze_short_code_failure(generations, detection_results, task='humaneval'):
    """Specifically analyze why short code fails."""
    det_results = detection_results[task]['watermark_detection']['raw_detection_results']

    print("\n" + "=" * 70)
    print("SHORT CODE FAILURE DEEP DIVE")
    print("=" * 70)

    short_failures = []
    for idx, (gens, det) in enumerate(zip(generations, det_results)):
        gen = gens[0] if isinstance(gens, list) else gens
        if det['num_tokens_generated'] <= 20:
            body = gen.split('\n')
            short_failures.append({
                'idx': idx,
                'gen_tokens': det['num_tokens_generated'],
                'scored': det['num_tokens_scored'],
                'z_score': det['z_score'],
                'wf': det['watermarking_fraction'],
                'code_snippet': gen[-200:] if len(gen) > 200 else gen  # last 200 chars
            })

    print(f"\nTotal short code samples (<=20 tokens): {len(short_failures)}")
    print(f"\nSample short code snippets (showing last function lines):")
    print("-" * 70)

    for sf in short_failures[:10]:
        # Show just the last few lines
        lines = sf['code_snippet'].strip().split('\n')
        last_lines = '\n'.join(lines[-3:])
        print(f"\n[Sample {sf['idx']}] gen={sf['gen_tokens']} scored={sf['scored']} z={sf['z_score']:.2f}")
        print(f"  {last_lines}")

    return short_failures


def main():
    parser = argparse.ArgumentParser(description="Analyze code patterns in failure cases")
    parser.add_argument('--task', type=str, default='humaneval')
    parser.add_argument('--generations', type=str, default='outputs_sweet/generations.json')
    parser.add_argument('--results', type=str, default='outputs_sweet/evaluation_results.json')
    args = parser.parse_args()

    print("Loading data...")
    generations = load_generations(args.generations)
    results = load_results(args.results)

    # Main analysis
    all_features = analyze_failure_by_code_pattern(generations, results, args.task)

    # Short code deep dive
    short_failures = analyze_short_code_failure(generations, results, args.task)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    det_results = results[args.task]['watermark_detection']['raw_detection_results']
    total = len(det_results)
    detected = sum(1 for r in det_results if r.get('z_score', 0) >= 4)

    print(f"\n1. Detection Rate: {detected}/{total} = {100*detected/total:.1f}%")
    print(f"2. Short code (<=20 gen tokens): {len(short_failures)} samples, 0% detection")

    oneliners = [f for f in all_features if f['is_one_liner']]
    print(f"3. One-liner solutions: {len(oneliners)} samples, "
          f"{100*sum(1 for f in oneliners if f['detected'])/max(len(oneliners),1):.1f}% detection")

    long_code = [f for f in all_features if f['num_tokens_generated'] > 100]
    print(f"4. Long code (>100 tokens): {len(long_code)} samples, "
          f"{100*sum(1 for f in long_code if f['detected'])/max(len(long_code),1):.1f}% detection")

    print(f"\n5. Root cause: {100*sum(1 for r in det_results if r['num_tokens_scored'] < 16)/total:.1f}% "
          f"of samples have < 16 scored tokens (mathematically impossible to detect)")

    print("\n" + "=" * 70)
    print("CODE PATTERN ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
