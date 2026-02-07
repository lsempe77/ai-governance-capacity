"""
Calculate Inter-Rater Reliability and Validation Metrics

Run this after manual coding is complete.
Expects: Two CSV files with manual scores from two coders.
"""
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_coding_sheet(filepath):
    """Load a completed coding sheet."""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy_id = row['Policy_ID']
            data[policy_id] = {
                'title': row['Title'],
                'jurisdiction': row['Jurisdiction'],
                'auto': {
                    'clarity': int(row['AUTO_Clarity']),
                    'resources': int(row['AUTO_Resources']),
                    'authority': int(row['AUTO_Authority']),
                    'accountability': int(row['AUTO_Accountability']),
                    'coherence': int(row['AUTO_Coherence']),
                },
                'manual': {
                    'clarity': int(row['MANUAL_Clarity']) if row['MANUAL_Clarity'] else None,
                    'resources': int(row['MANUAL_Resources']) if row['MANUAL_Resources'] else None,
                    'authority': int(row['MANUAL_Authority']) if row['MANUAL_Authority'] else None,
                    'accountability': int(row['MANUAL_Accountability']) if row['MANUAL_Accountability'] else None,
                    'coherence': int(row['MANUAL_Coherence']) if row['MANUAL_Coherence'] else None,
                }
            }
    return data


def cohens_kappa(ratings1, ratings2):
    """
    Calculate Cohen's kappa for ordinal ratings.
    Using linear weights for ordinal data.
    """
    # Filter out None values
    valid = [(r1, r2) for r1, r2 in zip(ratings1, ratings2) if r1 is not None and r2 is not None]
    if not valid:
        return None
    
    ratings1 = [v[0] for v in valid]
    ratings2 = [v[1] for v in valid]
    
    n = len(ratings1)
    categories = list(range(5))  # 0-4
    
    # Build confusion matrix
    matrix = np.zeros((5, 5))
    for r1, r2 in zip(ratings1, ratings2):
        matrix[int(r1), int(r2)] += 1
    
    # Calculate observed agreement (weighted)
    weights = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            weights[i, j] = 1 - abs(i - j) / 4  # Linear weights
    
    po = np.sum(weights * matrix) / n
    
    # Calculate expected agreement
    row_marginals = matrix.sum(axis=1) / n
    col_marginals = matrix.sum(axis=0) / n
    pe = np.sum(weights * np.outer(row_marginals, col_marginals))
    
    if pe == 1:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)
    return kappa


def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient."""
    valid = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(valid) < 3:
        return None
    x = [v[0] for v in valid]
    y = [v[1] for v in valid]
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = np.sqrt(sum((xi - mean_x)**2 for xi in x) * sum((yi - mean_y)**2 for yi in y))
    
    if denominator == 0:
        return None
    return numerator / denominator


def mean_absolute_error(x, y):
    """Calculate mean absolute error."""
    valid = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not valid:
        return None
    x = [v[0] for v in valid]
    y = [v[1] for v in valid]
    return np.mean([abs(a - b) for a, b in zip(x, y)])


def analyze_single_coder(filepath):
    """Analyze a single coder's results against automated scores."""
    data = load_coding_sheet(filepath)
    
    dimensions = ['clarity', 'resources', 'authority', 'accountability', 'coherence']
    
    print("=" * 70)
    print("VALIDATION ANALYSIS: Automated vs Manual Scores")
    print("=" * 70)
    
    # Overall statistics
    auto_totals = []
    manual_totals = []
    
    for policy_id, info in data.items():
        if all(info['manual'][d] is not None for d in dimensions):
            auto_total = sum(info['auto'][d] for d in dimensions)
            manual_total = sum(info['manual'][d] for d in dimensions)
            auto_totals.append(auto_total)
            manual_totals.append(manual_total)
    
    print(f"\nPolicies with complete manual scores: {len(auto_totals)}")
    
    if auto_totals:
        # Per-dimension analysis
        print("\n" + "-" * 70)
        print("DIMENSION-LEVEL ANALYSIS")
        print("-" * 70)
        print(f"{'Dimension':<20} {'Correlation':>12} {'MAE':>10} {'Mean Diff':>12}")
        print("-" * 70)
        
        for dim in dimensions:
            auto = [data[p]['auto'][dim] for p in data]
            manual = [data[p]['manual'][dim] for p in data]
            
            r = pearson_correlation(auto, manual)
            mae = mean_absolute_error(auto, manual)
            
            valid = [(a, m) for a, m in zip(auto, manual) if m is not None]
            if valid:
                mean_diff = np.mean([m - a for a, m in valid])
            else:
                mean_diff = None
            
            r_str = f"{r:.3f}" if r is not None else "N/A"
            mae_str = f"{mae:.2f}" if mae is not None else "N/A"
            diff_str = f"{mean_diff:+.2f}" if mean_diff is not None else "N/A"
            
            print(f"{dim.title():<20} {r_str:>12} {mae_str:>10} {diff_str:>12}")
        
        # Overall
        print("-" * 70)
        r_total = pearson_correlation(auto_totals, manual_totals)
        mae_total = mean_absolute_error(auto_totals, manual_totals)
        mean_diff_total = np.mean([m - a for a, m in zip(auto_totals, manual_totals)])
        
        print(f"{'TOTAL (0-20)':<20} {r_total:.3f}" if r_total else "N/A")
        print(f"{'MAE':<20} {mae_total:.2f}")
        print(f"{'Mean Difference':<20} {mean_diff_total:+.2f}")
        
        # Interpretation
        print("\n" + "-" * 70)
        print("INTERPRETATION")
        print("-" * 70)
        
        if r_total and r_total > 0.7:
            print("✓ Strong correlation (r > 0.7): Automated scores are valid")
        elif r_total and r_total > 0.5:
            print("~ Moderate correlation (0.5 < r < 0.7): Automated scores need refinement")
        else:
            print("✗ Weak correlation (r < 0.5): Methodology needs revision")
        
        if mae_total < 2:
            print("✓ Low MAE (< 2 points): Scores are reasonably accurate")
        else:
            print("✗ High MAE (≥ 2 points): Significant scoring errors")
        
        if abs(mean_diff_total) < 1:
            print("✓ Low systematic bias: No consistent over/under-scoring")
        elif mean_diff_total > 0:
            print(f"~ Systematic underscoring by automated system (+{mean_diff_total:.1f} points)")
        else:
            print(f"~ Systematic overscoring by automated system ({mean_diff_total:.1f} points)")
    
    return data


def analyze_two_coders(filepath1, filepath2):
    """Analyze inter-rater reliability between two coders."""
    data1 = load_coding_sheet(filepath1)
    data2 = load_coding_sheet(filepath2)
    
    dimensions = ['clarity', 'resources', 'authority', 'accountability', 'coherence']
    
    print("\n" + "=" * 70)
    print("INTER-RATER RELIABILITY: Coder 1 vs Coder 2")
    print("=" * 70)
    
    print(f"\n{'Dimension':<20} {'Cohen κ':>12} {'Interpretation':>20}")
    print("-" * 70)
    
    for dim in dimensions:
        ratings1 = [data1[p]['manual'][dim] for p in data1 if p in data2]
        ratings2 = [data2[p]['manual'][dim] for p in data2 if p in data1]
        
        kappa = cohens_kappa(ratings1, ratings2)
        
        if kappa is not None:
            if kappa > 0.8:
                interp = "Excellent"
            elif kappa > 0.6:
                interp = "Substantial"
            elif kappa > 0.4:
                interp = "Moderate"
            elif kappa > 0.2:
                interp = "Fair"
            else:
                interp = "Poor"
            print(f"{dim.title():<20} {kappa:>12.3f} {interp:>20}")
        else:
            print(f"{dim.title():<20} {'N/A':>12} {'Insufficient data':>20}")
    
    # Identify discrepancies
    print("\n" + "-" * 70)
    print("MAJOR DISCREPANCIES (difference ≥ 2 points)")
    print("-" * 70)
    
    for policy_id in data1:
        if policy_id not in data2:
            continue
        
        for dim in dimensions:
            r1 = data1[policy_id]['manual'][dim]
            r2 = data2[policy_id]['manual'][dim]
            
            if r1 is not None and r2 is not None and abs(r1 - r2) >= 2:
                print(f"{policy_id}: {dim.title()} - Coder1={r1}, Coder2={r2}, Diff={abs(r1-r2)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate validation metrics')
    parser.add_argument('--coder1', default='data/analysis/validation/validation_coding_sheet.csv',
                       help='Path to first coder CSV')
    parser.add_argument('--coder2', help='Path to second coder CSV (optional)')
    args = parser.parse_args()
    
    # Analyze single coder vs automated
    analyze_single_coder(args.coder1)
    
    # If two coders, analyze inter-rater reliability
    if args.coder2:
        analyze_two_coders(args.coder1, args.coder2)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. If inter-rater κ < 0.6: Revise coding scheme, retrain coders")
    print("2. If automated-manual r < 0.7: Revise extraction patterns")
    print("3. If systematic bias detected: Adjust scoring thresholds")
    print("4. Document all discrepancies and resolution decisions")


if __name__ == '__main__':
    main()
