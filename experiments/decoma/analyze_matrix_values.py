#!/usr/bin/env python3
"""
Analyze the distribution of values in the pattern matrix
to determine optimal minimum_scale threshold.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datasets import load_from_disk
import re

from caw.attackers.decoma.utils import preprocess_from_list
from caw.attackers.decoma.decoma import DeCoMa, DetectionConfig, PatternMatrix, PatternExtractor


def extract_code_from_message(message):
    code_snippets = []
    for msg in message:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            pattern = r"<code>(.*?)</code>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                code_snippets.append(match.group(1).strip())
    return "\n\n".join(code_snippets)


def main():
    print("Loading dataset...")
    raw_dataset = load_from_disk('../../llm_ft/data_prepare/datas/sqa_print_frq0.05')
    train_ds = raw_dataset["train"]

    # Extract code
    train_code_ls = []
    for row in train_ds:
        code = extract_code_from_message(row["messages"])
        train_code_ls.append(code)

    print(f"Processing {len(train_code_ls)} samples...")
    train_expressions = preprocess_from_list(train_code_ls)

    # Build pattern matrix
    config = DetectionConfig(z_score_threshold=4.0, detect_type="col", minimum_scale=2.0)
    extractor = PatternExtractor(config)

    pattern_sets = []
    for sample in train_expressions:
        if sample.is_valid():
            pattern_set = extractor.extract(sample)
            pattern_sets.append(pattern_set)

    matrix_builder = PatternMatrix(pattern_sets)
    matrix = matrix_builder.get_matrix()

    print(f"\nMatrix shape: {matrix.shape}")

    # Analyze value distribution
    all_values = matrix.values.flatten()
    non_zero_values = all_values[all_values > 0]

    print(f"\nValue Statistics:")
    print(f"  Total values: {len(all_values)}")
    print(f"  Non-zero values: {len(non_zero_values)} ({len(non_zero_values)/len(all_values):.2%})")
    print(f"  Zero values: {len(all_values) - len(non_zero_values)} ({(len(all_values) - len(non_zero_values))/len(all_values):.2%})")

    print(f"\nNon-zero Value Distribution:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(non_zero_values, p)
        print(f"  {p}th percentile: {val:.4f}")

    print(f"\n  Mean: {non_zero_values.mean():.4f}")
    print(f"  Std: {non_zero_values.std():.4f}")
    print(f"  Min: {non_zero_values.min():.4f}")
    print(f"  Max: {non_zero_values.max():.4f}")

    # Analyze per-sample maximum values
    print(f"\nPer-sample Maximum Values:")
    max_per_sample = matrix.max(axis=1)
    for p in percentiles:
        val = np.percentile(max_per_sample, p)
        print(f"  {p}th percentile: {val:.4f}")

    # Analyze how many samples have values above different thresholds
    print(f"\nSamples with max value above threshold:")
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for thresh in thresholds:
        count = (max_per_sample > thresh).sum()
        pct = count / len(max_per_sample) * 100
        print(f"  > {thresh}: {count} samples ({pct:.1f}%)")

    # Plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of non-zero values
    axes[0, 0].hist(non_zero_values, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='current threshold=2.0')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Non-zero Values')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 10)

    # 2. Histogram of non-zero values (log scale)
    axes[0, 1].hist(non_zero_values, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(2.0, color='red', linestyle='--', linewidth=2, label='current threshold=2.0')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].set_title('Distribution of Non-zero Values (Log Scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 10)

    # 3. Per-sample maximum values
    axes[1, 0].hist(max_per_sample, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='current threshold=2.0')
    axes[1, 0].set_xlabel('Maximum value per sample')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Per-Sample Maximum Values')
    axes[1, 0].legend()

    # 4. CDF of maximum values
    sorted_max = np.sort(max_per_sample)
    cdf = np.arange(1, len(sorted_max) + 1) / len(sorted_max)
    axes[1, 1].plot(sorted_max, cdf, linewidth=2)
    axes[1, 1].axvline(2.0, color='red', linestyle='--', linewidth=2, label='current threshold=2.0')
    axes[1, 1].axhline(0.9, color='green', linestyle=':', linewidth=1, label='90% samples')
    axes[1, 1].set_xlabel('Maximum value per sample')
    axes[1, 1].set_ylabel('Cumulative probability')
    axes[1, 1].set_title('CDF of Per-Sample Maximum Values')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('matrix_value_distribution.png', dpi=150)
    print(f"\nPlot saved to matrix_value_distribution.png")

    # Find optimal threshold for 10% detection rate
    print(f"\n" + "="*80)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*80)
    target_detection_rate = 0.10
    target_samples = int(len(matrix) * target_detection_rate)

    # Find threshold where approximately 10% of samples have max > threshold
    optimal_threshold = np.percentile(max_per_sample, 90)
    samples_above = (max_per_sample > optimal_threshold).sum()

    print(f"Target detection rate: {target_detection_rate:.1%} ({target_samples} samples)")
    print(f"Optimal threshold (90th percentile of max values): {optimal_threshold:.2f}")
    print(f"Samples with max > {optimal_threshold:.2f}: {samples_above} ({samples_above/len(matrix):.1%})")


if __name__ == "__main__":
    main()
