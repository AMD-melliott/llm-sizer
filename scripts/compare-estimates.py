#!/usr/bin/env python3
"""
Compare actual memory measurements with calculator estimates

Usage:
    python compare-estimates.py memory-profile.json \
        --calc-weights 13.5 \
        --calc-kv 0.4 \
        --calc-activations 0.9 \
        --calc-overhead 0.5
"""

import argparse
import json
import sys


def load_profile(filepath: str) -> dict:
    """Load memory profile JSON"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}", file=sys.stderr)
        sys.exit(1)


def compare_values(actual: float, calculated: float, component: str) -> dict:
    """Compare actual vs calculated values and compute metrics"""
    diff = actual - calculated
    if calculated > 0:
        percent_diff = (diff / calculated) * 100
    else:
        percent_diff = float('inf') if actual > 0 else 0

    match = "✓" if abs(percent_diff) < 15 else "⚠" if abs(percent_diff) < 30 else "✗"

    return {
        'component': component,
        'actual_gb': round(actual, 2),
        'calculated_gb': round(calculated, 2),
        'difference_gb': round(diff, 2),
        'percent_diff': round(percent_diff, 1),
        'match': match
    }


def print_comparison_table(comparisons: list):
    """Print formatted comparison table"""

    print("\n" + "="*80)
    print("MEMORY ESTIMATE COMPARISON")
    print("="*80)
    print(f"{'Component':<20} {'Actual':<12} {'Calculated':<12} {'Diff':<12} {'% Diff':<10} {'Match':<5}")
    print("-"*80)

    for comp in comparisons:
        print(
            f"{comp['component']:<20} "
            f"{comp['actual_gb']:>10.2f} GB "
            f"{comp['calculated_gb']:>10.2f} GB "
            f"{comp['difference_gb']:>+10.2f} GB "
            f"{comp['percent_diff']:>8.1f}% "
            f"{comp['match']:>5}"
        )

    print("-"*80)

    # Overall accuracy
    total_actual = sum(c['actual_gb'] for c in comparisons if c['component'] != 'Total')
    total_calc = sum(c['calculated_gb'] for c in comparisons if c['component'] != 'Total')
    total_comp = compare_values(total_actual, total_calc, 'TOTAL')

    print(
        f"{'TOTAL':<20} "
        f"{total_comp['actual_gb']:>10.2f} GB "
        f"{total_comp['calculated_gb']:>10.2f} GB "
        f"{total_comp['difference_gb']:>+10.2f} GB "
        f"{total_comp['percent_diff']:>8.1f}% "
        f"{total_comp['match']:>5}"
    )
    print("="*80)


def print_analysis(comparisons: list):
    """Print analysis and recommendations"""

    print("\nANALYSIS")
    print("-"*80)

    # Check each component
    issues = []
    for comp in comparisons:
        if comp['component'] == 'Total':
            continue

        percent_diff = abs(comp['percent_diff'])

        if percent_diff > 30:
            issues.append(f"❌ {comp['component']}: Large discrepancy ({comp['percent_diff']:+.1f}%)")
        elif percent_diff > 15:
            issues.append(f"⚠️  {comp['component']}: Moderate discrepancy ({comp['percent_diff']:+.1f}%)")
        else:
            print(f"✅ {comp['component']}: Good match ({comp['percent_diff']:+.1f}%)")

    if issues:
        print("\nIssues detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ All components match well!")

    # Recommendations
    print("\nRECOMMENDATIONS")
    print("-"*80)

    for comp in comparisons:
        if comp['component'] == 'Total':
            continue

        percent_diff = comp['percent_diff']

        if abs(percent_diff) > 15:
            component = comp['component']

            if component == 'Model Weights':
                if percent_diff > 0:
                    print(f"• Model Weights: Actual higher than calculated")
                    print(f"  → Check quantization settings match (FP16, INT8, etc.)")
                    print(f"  → Verify parameter count is correct")
                else:
                    print(f"• Model Weights: Actual lower than calculated")
                    print(f"  → Model might be using better compression")
                    print(f"  → Check for weight sharing or quantization")

            elif component == 'KV Cache':
                if percent_diff > 0:
                    print(f"• KV Cache: Actual higher than calculated")
                    print(f"  → Verify sequence length matches (actual vs requested)")
                    print(f"  → Check if model uses GQA/MQA (affects KV size)")
                else:
                    print(f"• KV Cache: Actual lower than calculated")
                    print(f"  → Possible PagedAttention or KV cache compression")
                    print(f"  → Check framework optimizations")

            elif component == 'Activations':
                if percent_diff > 0:
                    print(f"• Activations: Actual higher than calculated")
                    print(f"  → Batch size or hidden dims may be different")
                    print(f"  → Framework may use more temporary buffers")
                else:
                    print(f"• Activations: Actual lower than calculated")
                    print(f"  → Framework using activation checkpointing")
                    print(f"  → More efficient memory management")

            elif component == 'Framework Overhead':
                if percent_diff > 0:
                    print(f"• Framework Overhead: Actual higher than calculated")
                    print(f"  → Increase overhead percentage in calculator")
                else:
                    print(f"• Framework Overhead: May be overestimated")
                    print(f"  → This can indicate other components are overestimated")


def main():
    parser = argparse.ArgumentParser(
        description="Compare actual memory profile with calculator estimates"
    )
    parser.add_argument(
        "profile",
        help="Path to memory-profile.json from profile-model-memory.py"
    )
    parser.add_argument(
        "--calc-weights",
        type=float,
        required=True,
        help="Calculated model weights (GB)"
    )
    parser.add_argument(
        "--calc-kv",
        type=float,
        required=True,
        help="Calculated KV cache (GB)"
    )
    parser.add_argument(
        "--calc-activations",
        type=float,
        required=True,
        help="Calculated activations (GB)"
    )
    parser.add_argument(
        "--calc-overhead",
        type=float,
        required=True,
        help="Calculated framework overhead (GB)"
    )

    args = parser.parse_args()

    # Load actual measurements
    profile = load_profile(args.profile)
    actual = profile.get('memory_breakdown', {})

    # Compare each component
    comparisons = [
        compare_values(
            actual.get('model_weights_gb', 0),
            args.calc_weights,
            'Model Weights'
        ),
        compare_values(
            actual.get('kv_cache_gb', 0),
            args.calc_kv,
            'KV Cache'
        ),
        compare_values(
            actual.get('activations_gb', 0),
            args.calc_activations,
            'Activations'
        ),
        compare_values(
            actual.get('framework_overhead_gb', 0),
            args.calc_overhead,
            'Framework Overhead'
        ),
    ]

    # Print results
    print_comparison_table(comparisons)
    print_analysis(comparisons)

    # Show model info
    if 'model_info' in profile:
        info = profile['model_info']
        print("\nMODEL INFO")
        print("-"*80)
        print(f"Model: {info.get('model_name', 'unknown')}")
        print(f"Parameters: {info.get('num_parameters', 0):,}")
        print(f"Data Type: {info.get('dtype', 'unknown')}")
        print(f"Sequence Length: {info.get('total_sequence_length', 0)}")
        print(f"Batch Size: {info.get('batch_size', 1)}")

    print()


if __name__ == "__main__":
    main()
