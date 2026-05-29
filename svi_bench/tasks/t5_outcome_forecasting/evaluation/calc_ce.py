"""
Calibration Error (CE) Calculation Script with Deep Analysis

Calculates the Calibration Error for model predictions.
CE measures how well the predicted probabilities match the actual accuracy.

Formula: CE = sum_i (n_i / N) * |acc(i) - conf(i)|
Where:
- N is total number of samples
- n_i is number of samples in bin i
- acc(i) is empirical accuracy in bin i
- conf(i) is average predicted confidence in bin i

Usage:
  python calc_ce.py --results results.json
  python calc_ce.py --results results.json --num_bins 10
"""

import argparse
import json
from collections import defaultdict
import statistics


def deep_analysis(results_file: str):
    """Perform deep analysis to check for data issues."""

    with open(results_file, 'r') as f:
        results = json.load(f)

    print("=" * 70)
    print("DEEP ANALYSIS - CHECKING FOR DATA ISSUES")
    print("=" * 70)

    # 1. Check probability sums
    print("\n1. PROBABILITY SUM CHECK")
    print("-" * 50)
    prob_sums = []
    for entry in results:
        if entry.get('error') is not None:
            continue
        option_probs = entry.get('option_probabilities', {})
        if option_probs:
            prob_sum = sum(option_probs.values())
            prob_sums.append(prob_sum)

    print(f"   Min sum: {min(prob_sums):.6f}")
    print(f"   Max sum: {max(prob_sums):.6f}")
    print(f"   Mean sum: {statistics.mean(prob_sums):.6f}")
    print(f"   Std sum: {statistics.stdev(prob_sums):.6f}")
    print(f"   -> All sums should be ~1.0 if softmax is applied correctly")

    # 2. Distribution of ground truth option probabilities
    print("\n2. GROUND TRUTH OPTION PROBABILITY DISTRIBUTION")
    print("-" * 50)
    gt_probs = []
    wrong_pred_gt_probs = []
    correct_pred_gt_probs = []

    for entry in results:
        if entry.get('error') is not None:
            continue
        option_probs = entry.get('option_probabilities', {})
        ground_truth = entry.get('ground_truth', '')
        prediction = entry.get('prediction', '')

        if ground_truth in option_probs:
            gt_prob = option_probs[ground_truth]
            gt_probs.append(gt_prob)

            if prediction == ground_truth:
                correct_pred_gt_probs.append(gt_prob)
            else:
                wrong_pred_gt_probs.append(gt_prob)

    print(f"   All questions:")
    print(f"     Min GT prob: {min(gt_probs):.4f}")
    print(f"     Max GT prob: {max(gt_probs):.4f}")
    print(f"     Mean GT prob: {statistics.mean(gt_probs):.4f}")
    print(f"     Median GT prob: {statistics.median(gt_probs):.4f}")

    if correct_pred_gt_probs:
        print(f"\n   Correctly predicted questions ({len(correct_pred_gt_probs)}):")
        print(f"     Mean GT prob: {statistics.mean(correct_pred_gt_probs):.4f}")
        print(f"     Median GT prob: {statistics.median(correct_pred_gt_probs):.4f}")

    if wrong_pred_gt_probs:
        print(f"\n   Incorrectly predicted questions ({len(wrong_pred_gt_probs)}):")
        print(f"     Mean GT prob: {statistics.mean(wrong_pred_gt_probs):.4f}")
        print(f"     Median GT prob: {statistics.median(wrong_pred_gt_probs):.4f}")

    # 3. Model accuracy
    print("\n3. MODEL ACCURACY")
    print("-" * 50)
    correct = sum(1 for e in results if e.get('error') is None and e.get('prediction') == e.get('ground_truth'))
    total = sum(1 for e in results if e.get('error') is None)
    print(f"   Correct: {correct} / {total} = {100*correct/total:.2f}%")

    # 4. Distribution of predicted option probabilities
    print("\n4. PREDICTED OPTION PROBABILITY DISTRIBUTION")
    print("-" * 50)
    pred_probs = []
    correct_pred_probs = []
    wrong_pred_probs = []

    for entry in results:
        if entry.get('error') is not None:
            continue
        option_probs = entry.get('option_probabilities', {})
        prediction = entry.get('prediction', '')
        ground_truth = entry.get('ground_truth', '')

        if prediction in option_probs:
            pred_prob = option_probs[prediction]
            pred_probs.append(pred_prob)

            if prediction == ground_truth:
                correct_pred_probs.append(pred_prob)
            else:
                wrong_pred_probs.append(pred_prob)

    print(f"   All predictions:")
    print(f"     Min predicted prob: {min(pred_probs):.4f}")
    print(f"     Max predicted prob: {max(pred_probs):.4f}")
    print(f"     Mean predicted prob: {statistics.mean(pred_probs):.4f}")

    if correct_pred_probs:
        print(f"\n   Correct predictions ({len(correct_pred_probs)}):")
        print(f"     Mean predicted prob: {statistics.mean(correct_pred_probs):.4f}")

    if wrong_pred_probs:
        print(f"\n   Wrong predictions ({len(wrong_pred_probs)}):")
        print(f"     Mean predicted prob: {statistics.mean(wrong_pred_probs):.4f}")

    # 5. Check if high-prob predictions are always correct
    print("\n5. HIGH CONFIDENCE PREDICTION ANALYSIS")
    print("-" * 50)
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        high_conf_correct = sum(1 for e in results
                               if e.get('error') is None
                               and e.get('prediction') == e.get('ground_truth')
                               and e.get('option_probabilities', {}).get(e.get('prediction', ''), 0) >= threshold)
        high_conf_total = sum(1 for e in results
                             if e.get('error') is None
                             and e.get('option_probabilities', {}).get(e.get('prediction', ''), 0) >= threshold)
        if high_conf_total > 0:
            print(f"   Predictions with conf >= {threshold}: {high_conf_correct}/{high_conf_total} = {100*high_conf_correct/high_conf_total:.2f}%")
        else:
            print(f"   Predictions with conf >= {threshold}: 0 samples")

    # 6. Histogram of all option probabilities
    print("\n6. HISTOGRAM OF ALL OPTION PROBABILITIES")
    print("-" * 50)
    all_probs = []
    for entry in results:
        if entry.get('error') is not None:
            continue
        option_probs = entry.get('option_probabilities', {})
        all_probs.extend(option_probs.values())

    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in bins:
        count = sum(1 for p in all_probs if low <= p < high)
        pct = 100 * count / len(all_probs)
        bar = "█" * int(pct / 2)
        print(f"   [{low:.1f}-{high:.1f}): {count:6d} ({pct:5.1f}%) {bar}")

    # 7. Check for number of options per question
    print("\n7. NUMBER OF OPTIONS PER QUESTION")
    print("-" * 50)
    option_counts = defaultdict(int)
    for entry in results:
        if entry.get('error') is not None:
            continue
        option_probs = entry.get('option_probabilities', {})
        option_counts[len(option_probs)] += 1

    for n_opts, count in sorted(option_counts.items()):
        print(f"   {n_opts} options: {count} questions")

    # 8. Sample some entries for manual inspection
    print("\n8. SAMPLE ENTRIES FOR INSPECTION")
    print("-" * 50)
    sample_indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, len(results)-1]
    for idx in sample_indices[:3]:
        entry = results[idx]
        print(f"\n   Entry {idx}: {entry.get('id', 'N/A')}")
        print(f"     Ground Truth: {entry.get('ground_truth')}")
        print(f"     Prediction: {entry.get('prediction')}")
        print(f"     Option Probs: {entry.get('option_probabilities')}")
        if entry.get('option_probabilities'):
            print(f"     Sum: {sum(entry.get('option_probabilities', {}).values()):.4f}")


def calculate_calibration_error(results_file: str, num_bins: int = 5):
    """
    Calculate calibration error from model results.

    For each question, we look at all option probabilities and put each option
    into a bin based on its probability. We record whether that option was the
    ground truth (1 if correct, 0 otherwise).

    Args:
        results_file: Path to the JSON results file
        num_bins: Number of bins (default 5: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0])
    """

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Initialize bins
    # Bin boundaries: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]

    # Each bin stores: list of (predicted_prob, is_correct) tuples
    bins = defaultdict(list)

    # Process each question
    total_samples = 0
    for entry in results:
        if entry.get('error') is not None:
            continue  # Skip entries with errors

        option_probs = entry.get('option_probabilities', {})
        ground_truth = entry.get('ground_truth', '')

        if not option_probs:
            continue

        # For each option and its probability
        for option, prob in option_probs.items():
            # Determine which bin this probability falls into
            # Bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
            bin_idx = min(int(prob * num_bins), num_bins - 1)

            # Is this option the correct answer?
            is_correct = 1 if option == ground_truth else 0

            bins[bin_idx].append((prob, is_correct))
            total_samples += 1

    # Calculate statistics for each bin
    print("\n" + "=" * 70)
    print("CALIBRATION ERROR CALCULATION")
    print("=" * 70)
    print(f"\nTotal samples (option-level): {total_samples}")
    print(f"Number of questions processed: {len(results)}")
    print("\n" + "-" * 70)
    print("BIN-WISE BREAKDOWN")
    print("-" * 70)

    bin_stats = []
    for i in range(num_bins):
        bin_data = bins[i]
        n_i = len(bin_data)

        if n_i > 0:
            # Average predicted confidence in this bin
            avg_conf = sum(prob for prob, _ in bin_data) / n_i
            # Empirical accuracy in this bin (proportion of correct answers)
            avg_acc = sum(is_correct for _, is_correct in bin_data) / n_i
        else:
            avg_conf = 0
            avg_acc = 0

        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        bin_stats.append({
            'bin_idx': i,
            'bin_range': f"[{bin_lower:.1f}, {bin_upper:.1f})",
            'n_samples': n_i,
            'avg_confidence': avg_conf,
            'avg_accuracy': avg_acc,
            'abs_diff': abs(avg_acc - avg_conf)
        })

        print(f"\nBin {i+1}: [{bin_lower:.1f}, {bin_upper:.1f})")
        print(f"  Samples: {n_i} ({100*n_i/total_samples:.2f}%)")
        print(f"  Average Confidence: {avg_conf:.4f}")
        print(f"  Average Accuracy (Ground Truth Rate): {avg_acc:.4f}")
        print(f"  |Accuracy - Confidence|: {abs(avg_acc - avg_conf):.4f}")

    # Calculate Calibration Error
    # CE = sum_i (n_i / N) * |acc(i) - conf(i)|
    ce = 0
    for stat in bin_stats:
        weight = stat['n_samples'] / total_samples
        ce += weight * stat['abs_diff']

    # Also calculate simple average across bins (unweighted)
    simple_avg_ce = sum(stat['abs_diff'] for stat in bin_stats) / num_bins

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Simple average CE across bins (unweighted) - PRIMARY METRIC
    print(f"\n*** CALIBRATION ERROR (CE): {simple_avg_ce:.4f} ***")
    print(f"  Formula: CE = (1/B) * sum_i |acc(i) - conf(i)|")
    print(f"  This is the simple average of |accuracy - confidence| across {num_bins} bins")

    # Also show weighted version for reference
    print(f"\nWeighted CE (for reference): {ce:.4f}")
    print(f"  Formula: CE = sum_i (n_i / N) * |acc(i) - conf(i)|")

    # Additional: Expected Calibration Error interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print(f"A lower CE indicates better calibration.")
    print(f"CE = 0 means perfect calibration (predicted prob = actual accuracy)")
    print(f"CE = 1 means worst calibration")

    return simple_avg_ce, ce, bin_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration Error Calculation")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of bins (default: 5)")
    args = parser.parse_args()

    deep_analysis(args.results)
    ce, weighted_ce, bin_stats = calculate_calibration_error(args.results, num_bins=args.num_bins)
