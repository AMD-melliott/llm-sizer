#!/usr/bin/env python3
"""
Compare Batch Profile Results with Calculator Estimates (v3.0)

This script analyzes batch profiling results from batch-profile-bench-v3.py
and compares actual measurements against LLM-sizer calculator estimates.

Helps identify systematic biases and calibration needs in calculator formulas.

Usage:
    # Analyze batch results CSV
    python compare-batch-estimates-v3.py results/memory-profiles/batch-bench-results-v3-*.csv

    # With custom threshold
    python compare-batch-estimates-v3.py results.csv --threshold 20

    # Generate detailed report
    python compare-batch-estimates-v3.py results.csv --detailed --output report.txt
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics


# Import calculator formulas
try:
    sys.path.insert(0, str(Path(__file__).parent / 'lib'))
    from calculator_formulas import calculate_expected_memory, calculate_proposed_memory
    CALCULATOR_AVAILABLE = True
except ImportError:
    CALCULATOR_AVAILABLE = False
    print("⚠️  Warning: calculator_formulas module not available")
    print("   Place calculator_formulas.py in scripts/lib/")
    sys.exit(1)

# Import model loader for architecture params
try:
    from model_loader import get_model_info
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    print("⚠️  Warning: model_loader module not available")


@dataclass
class ProfileComparison:
    """Comparison between actual measurement and calculator estimate"""
    model_id: str
    input_len: int
    output_len: int
    batch_size: int
    tensor_parallel_size: int

    # Actual measurements
    actual_total_gb: float
    actual_weights_gb: float
    actual_kv_cache_gb: float
    actual_activations_gb: float
    actual_overhead_gb: float

    # v3.0 specific actual measurements
    actual_kv_allocated_gb: Optional[float] = None
    actual_kv_used_gb: Optional[float] = None
    actual_graph_capture_gb: Optional[float] = None

    # Calculator estimates
    calc_total_gb: float = 0.0
    calc_weights_gb: float = 0.0
    calc_kv_cache_gb: float = 0.0
    calc_activations_gb: float = 0.0
    calc_overhead_gb: float = 0.0

    # Errors (actual - calculated)
    error_total_gb: float = 0.0
    error_weights_gb: float = 0.0
    error_kv_cache_gb: float = 0.0
    error_activations_gb: float = 0.0
    error_overhead_gb: float = 0.0

    # Percent errors
    error_total_pct: float = 0.0
    error_weights_pct: float = 0.0
    error_kv_cache_pct: float = 0.0
    error_activations_pct: float = 0.0
    error_overhead_pct: float = 0.0

    # Metadata
    weight_dtype: str = 'float16'
    kv_cache_dtype: str = 'float16'
    vllm_version: str = 'unknown'
    success: bool = True


def load_batch_results(csv_path: Path) -> List[Dict]:
    """Load batch profiling results from CSV"""
    results = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only process successful profiles
            if row.get('success', '').lower() == 'true':
                results.append(row)

    return results


def get_model_architecture(model_id: str) -> Optional[Dict]:
    """Get model architecture parameters from models.json"""
    if not MODEL_LOADER_AVAILABLE:
        return None

    try:
        info = get_model_info(model_id)
        if info:
            return {
                'model_params': info.get('parameters', 0),
                'num_layers': info.get('num_layers', 32),
                'num_heads': info.get('num_attention_heads', 32),
                'head_dim': info.get('head_dim', 128),
                'hidden_size': info.get('hidden_size', 4096)
            }
    except Exception as e:
        print(f"  Warning: Could not load architecture for {model_id}: {e}")

    return None


def calculate_estimates(
    model_id: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    weight_dtype: str,
    kv_cache_dtype: str,
    tensor_parallel_size: int,
    use_proposed: bool = False
) -> Dict[str, float]:
    """Calculate memory estimates using calculator formulas"""

    # Get model architecture
    arch = get_model_architecture(model_id)
    if not arch:
        return {}

    # Choose formula version
    calc_func = calculate_proposed_memory if use_proposed else calculate_expected_memory

    try:
        result = calc_func(
            model_params=arch['model_params'],
            num_layers=arch['num_layers'],
            hidden_size=arch['hidden_size'],
            num_heads=arch['num_heads'],
            input_len=input_len,
            output_len=output_len,
            batch_size=batch_size,
            weight_dtype=weight_dtype,
            kv_cache_dtype=kv_cache_dtype,
            num_gpus=tensor_parallel_size
        )

        return {
            'total_gb': result['total_gb'],
            'weights_gb': result['weights_gb'],
            'kv_cache_gb': result['kv_cache_gb'],
            'overhead_gb': result.get('overhead_gb', 0),
            'activations_gb': result.get('activations_gb', 0)
        }
    except Exception as e:
        print(f"  Warning: Calculator failed for {model_id}: {e}")
        return {}


def compare_result(row: Dict, use_proposed: bool = False) -> Optional[ProfileComparison]:
    """Compare a single result row with calculator estimates"""

    # Extract values from row
    try:
        model_id = row['model_id']
        input_len = int(row['input_len'])
        output_len = int(row['output_len'])
        batch_size = int(row['batch_size'])
        tensor_parallel_size = int(row.get('tensor_parallel_size', 1)) if row.get('tensor_parallel_size') else 1

        # Actual measurements
        actual_total = float(row.get('total_memory_gb', 0) or 0)
        actual_weights = float(row.get('weights_gb', 0) or 0)
        actual_kv = float(row.get('kv_cache_used_gb', 0) or row.get('kv_cache_gb', 0) or 0)
        actual_activations = float(row.get('activations_gb', 0) or 0)
        actual_overhead = float(row.get('overhead_gb', 0) or 0)

        # v3.0 specific measurements
        actual_kv_allocated = float(row.get('kv_cache_allocated_gb', 0) or 0) if row.get('kv_cache_allocated_gb') else None
        actual_kv_used = float(row.get('kv_cache_used_gb', 0) or 0) if row.get('kv_cache_used_gb') else None
        actual_graph_capture = float(row.get('graph_capture_gb', 0) or 0) if row.get('graph_capture_gb') else None

        # Dtypes
        weight_dtype = row.get('weight_dtype', 'float16')
        kv_cache_dtype = row.get('kv_cache_dtype', weight_dtype)

    except (ValueError, KeyError) as e:
        print(f"  Warning: Could not parse row: {e}")
        return None

    # Skip if no actual measurements
    if actual_total == 0:
        return None

    # Calculate estimates
    estimates = calculate_estimates(
        model_id=model_id,
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        weight_dtype=weight_dtype,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=tensor_parallel_size,
        use_proposed=use_proposed
    )

    if not estimates:
        return None

    # Create comparison
    comp = ProfileComparison(
        model_id=model_id,
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,

        actual_total_gb=actual_total,
        actual_weights_gb=actual_weights,
        actual_kv_cache_gb=actual_kv,
        actual_activations_gb=actual_activations,
        actual_overhead_gb=actual_overhead,

        actual_kv_allocated_gb=actual_kv_allocated,
        actual_kv_used_gb=actual_kv_used,
        actual_graph_capture_gb=actual_graph_capture,

        calc_total_gb=estimates['total_gb'],
        calc_weights_gb=estimates['weights_gb'],
        calc_kv_cache_gb=estimates['kv_cache_gb'],
        calc_activations_gb=estimates.get('activations_gb', 0),
        calc_overhead_gb=estimates['overhead_gb'],

        weight_dtype=weight_dtype,
        kv_cache_dtype=kv_cache_dtype,
        vllm_version=row.get('engine_version', 'unknown')
    )

    # Calculate errors
    comp.error_total_gb = actual_total - comp.calc_total_gb
    comp.error_weights_gb = actual_weights - comp.calc_weights_gb
    comp.error_kv_cache_gb = actual_kv - comp.calc_kv_cache_gb
    comp.error_activations_gb = actual_activations - comp.calc_activations_gb
    comp.error_overhead_gb = actual_overhead - comp.calc_overhead_gb

    # Calculate percent errors
    comp.error_total_pct = (comp.error_total_gb / comp.calc_total_gb * 100) if comp.calc_total_gb > 0 else 0
    comp.error_weights_pct = (comp.error_weights_gb / comp.calc_weights_gb * 100) if comp.calc_weights_gb > 0 else 0
    comp.error_kv_cache_pct = (comp.error_kv_cache_gb / comp.calc_kv_cache_gb * 100) if comp.calc_kv_cache_gb > 0 else 0
    comp.error_activations_pct = (comp.error_activations_gb / comp.calc_activations_gb * 100) if comp.calc_activations_gb > 0 else 0
    comp.error_overhead_pct = (comp.error_overhead_gb / comp.calc_overhead_gb * 100) if comp.calc_overhead_gb > 0 else 0

    return comp


def print_summary_statistics(comparisons: List[ProfileComparison], threshold: float = 15.0):
    """Print summary statistics across all comparisons"""

    if not comparisons:
        print("No valid comparisons to analyze")
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall stats
    print(f"\nTotal Profiles Analyzed: {len(comparisons)}")
    print(f"Accuracy Threshold: ±{threshold}%")

    # Calculate statistics for each component
    components = [
        ('Total Memory', 'error_total_pct', 'error_total_gb'),
        ('Model Weights', 'error_weights_pct', 'error_weights_gb'),
        ('KV Cache', 'error_kv_cache_pct', 'error_kv_cache_gb'),
        ('Activations', 'error_activations_pct', 'error_activations_gb'),
        ('Overhead', 'error_overhead_pct', 'error_overhead_gb')
    ]

    print(f"\n{'Component':<20} {'Mean Err':<12} {'Std Dev':<12} {'Min Err':<12} {'Max Err':<12} {'Within':<10}")
    print("-" * 80)

    for comp_name, pct_field, gb_field in components:
        errors_pct = [getattr(c, pct_field) for c in comparisons]
        errors_gb = [getattr(c, gb_field) for c in comparisons]

        if errors_pct:
            mean_err = statistics.mean(errors_pct)
            std_err = statistics.stdev(errors_pct) if len(errors_pct) > 1 else 0
            min_err = min(errors_pct)
            max_err = max(errors_pct)
            within_threshold = sum(1 for e in errors_pct if abs(e) <= threshold)
            pct_within = (within_threshold / len(errors_pct)) * 100

            # Format with color indicators
            status = "✅" if abs(mean_err) <= threshold else "⚠️" if abs(mean_err) <= threshold * 2 else "❌"

            print(
                f"{comp_name:<20} "
                f"{mean_err:>+10.1f}% "
                f"{std_err:>10.1f}% "
                f"{min_err:>+10.1f}% "
                f"{max_err:>+10.1f}% "
                f"{pct_within:>7.0f}% {status}"
            )

    print("-" * 80)


def print_detailed_analysis(comparisons: List[ProfileComparison], threshold: float = 15.0):
    """Print detailed per-profile analysis"""

    print("\n" + "=" * 80)
    print("DETAILED PROFILE ANALYSIS")
    print("=" * 80)

    # Group by model
    by_model = defaultdict(list)
    for comp in comparisons:
        model_name = comp.model_id.split('/')[-1]
        by_model[model_name].append(comp)

    for model_name, model_comps in sorted(by_model.items()):
        print(f"\n{'─' * 80}")
        print(f"Model: {model_name} ({len(model_comps)} configurations)")
        print(f"{'─' * 80}")

        # Header
        print(f"{'Config':<30} {'Actual':<12} {'Calc':<12} {'Error':<12} {'Status':<8}")
        print("-" * 80)

        for comp in sorted(model_comps, key=lambda c: (c.input_len, c.output_len, c.batch_size)):
            config_str = f"in:{comp.input_len} out:{comp.output_len} bs:{comp.batch_size}"
            status = "✅ PASS" if abs(comp.error_total_pct) <= threshold else \
                     "⚠️ WARN" if abs(comp.error_total_pct) <= threshold * 2 else \
                     "❌ FAIL"

            print(
                f"{config_str:<30} "
                f"{comp.actual_total_gb:>10.2f} GB "
                f"{comp.calc_total_gb:>10.2f} GB "
                f"{comp.error_total_pct:>+10.1f}% "
                f"{status:<8}"
            )

            # Show component breakdown for failures
            if abs(comp.error_total_pct) > threshold:
                if abs(comp.error_weights_pct) > threshold:
                    print(f"  └─ Weights: {comp.error_weights_pct:+.1f}% ({comp.actual_weights_gb:.2f} vs {comp.calc_weights_gb:.2f} GB)")
                if abs(comp.error_kv_cache_pct) > threshold:
                    print(f"  └─ KV Cache: {comp.error_kv_cache_pct:+.1f}% ({comp.actual_kv_cache_gb:.2f} vs {comp.calc_kv_cache_gb:.2f} GB)")
                if abs(comp.error_activations_pct) > threshold:
                    print(f"  └─ Activations: {comp.error_activations_pct:+.1f}% ({comp.actual_activations_gb:.2f} vs {comp.calc_activations_gb:.2f} GB)")
                if abs(comp.error_overhead_pct) > threshold:
                    print(f"  └─ Overhead: {comp.error_overhead_pct:+.1f}% ({comp.actual_overhead_gb:.2f} vs {comp.calc_overhead_gb:.2f} GB)")


def print_pattern_analysis(comparisons: List[ProfileComparison]):
    """Analyze patterns in errors across different parameters"""

    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    # Group by batch size
    by_batch = defaultdict(list)
    for comp in comparisons:
        by_batch[comp.batch_size].append(comp.error_total_pct)

    print("\nError by Batch Size:")
    print(f"{'Batch Size':<15} {'Mean Error':<15} {'Std Dev':<15} {'Count':<10}")
    print("-" * 55)
    for bs in sorted(by_batch.keys()):
        errors = by_batch[bs]
        print(
            f"{bs:<15} "
            f"{statistics.mean(errors):>+13.1f}% "
            f"{statistics.stdev(errors) if len(errors) > 1 else 0:>13.1f}% "
            f"{len(errors):<10}"
        )

    # Group by sequence length
    by_seqlen = defaultdict(list)
    for comp in comparisons:
        total_len = comp.input_len + comp.output_len
        by_seqlen[total_len].append(comp.error_total_pct)

    print("\nError by Total Sequence Length:")
    print(f"{'Seq Length':<15} {'Mean Error':<15} {'Std Dev':<15} {'Count':<10}")
    print("-" * 55)
    for seqlen in sorted(by_seqlen.keys()):
        errors = by_seqlen[seqlen]
        print(
            f"{seqlen:<15} "
            f"{statistics.mean(errors):>+13.1f}% "
            f"{statistics.stdev(errors) if len(errors) > 1 else 0:>13.1f}% "
            f"{len(errors):<10}"
        )

    # KV cache dtype analysis
    by_kv_dtype = defaultdict(list)
    for comp in comparisons:
        by_kv_dtype[comp.kv_cache_dtype].append(comp.error_kv_cache_pct)

    if len(by_kv_dtype) > 1:
        print("\nKV Cache Error by Dtype:")
        print(f"{'KV Dtype':<15} {'Mean Error':<15} {'Std Dev':<15} {'Count':<10}")
        print("-" * 55)
        for dtype in sorted(by_kv_dtype.keys()):
            errors = by_kv_dtype[dtype]
            print(
                f"{dtype:<15} "
                f"{statistics.mean(errors):>+13.1f}% "
                f"{statistics.stdev(errors) if len(errors) > 1 else 0:>13.1f}% "
                f"{len(errors):<10}"
            )


def print_recommendations(comparisons: List[ProfileComparison], threshold: float = 15.0):
    """Generate recommendations based on error patterns"""

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR CALCULATOR ADJUSTMENTS")
    print("=" * 80)

    # Calculate component-level biases
    total_errors = [c.error_total_pct for c in comparisons]
    weights_errors = [c.error_weights_pct for c in comparisons]
    kv_errors = [c.error_kv_cache_pct for c in comparisons]
    act_errors = [c.error_activations_pct for c in comparisons]
    overhead_errors = [c.error_overhead_pct for c in comparisons]

    mean_total = statistics.mean(total_errors)
    mean_weights = statistics.mean(weights_errors)
    mean_kv = statistics.mean(kv_errors)
    mean_act = statistics.mean(act_errors)
    mean_overhead = statistics.mean(overhead_errors)

    print("\nSystematic Biases Detected:")
    print("-" * 80)

    # Total memory
    if abs(mean_total) > threshold:
        direction = "UNDERESTIMATES" if mean_total > 0 else "OVERESTIMATES"
        print(f"\n❌ Calculator {direction} total memory by {abs(mean_total):.1f}% on average")
        print(f"   Component breakdown:")

    # Component-specific recommendations
    recommendations = []

    if abs(mean_weights) > threshold:
        direction = "higher" if mean_weights > 0 else "lower"
        recommendations.append(
            f"• Model Weights: Actual is {abs(mean_weights):.1f}% {direction} than calculated\n"
            f"  → Check parameter count calculations\n"
            f"  → Verify dtype conversion factors\n"
            f"  → Consider model-specific optimizations"
        )

    if abs(mean_kv) > threshold:
        direction = "higher" if mean_kv > 0 else "lower"
        recommendations.append(
            f"• KV Cache: Actual is {abs(mean_kv):.1f}% {direction} than calculated\n"
            f"  → Review KV cache formula (2 * layers * hidden * seq * batch)\n"
            f"  → Check if GQA/MQA affects size\n"
            f"  → Consider PagedAttention overhead/savings"
        )

    if abs(mean_act) > threshold:
        direction = "higher" if mean_act > 0 else "lower"
        recommendations.append(
            f"• Activations: Actual is {abs(mean_act):.1f}% {direction} than calculated\n"
            f"  → Adjust activation memory formula\n"
            f"  → Check batch size scaling factor\n"
            f"  → Review hidden dimension multipliers"
        )

    if abs(mean_overhead) > threshold:
        direction = "higher" if mean_overhead > 0 else "lower"
        adjustment = "increase" if mean_overhead > 0 else "decrease"
        recommendations.append(
            f"• Framework Overhead: Actual is {abs(mean_overhead):.1f}% {direction} than calculated\n"
            f"  → {adjustment.capitalize()} overhead percentage\n"
            f"  → Current formula may need calibration\n"
            f"  → Consider vLLM version-specific overhead"
        )

    if recommendations:
        print("\nSpecific Adjustments Needed:")
        for rec in recommendations:
            print(f"\n{rec}")
    else:
        print("\n✅ No major adjustments needed - calculator is well-calibrated!")

    # Show best and worst predictions
    comparisons_sorted = sorted(comparisons, key=lambda c: abs(c.error_total_pct))

    print("\n" + "-" * 80)
    print("\nBest Predictions (lowest error):")
    for comp in comparisons_sorted[:3]:
        model_name = comp.model_id.split('/')[-1]
        print(f"  ✅ {model_name} (in:{comp.input_len} out:{comp.output_len} bs:{comp.batch_size}): {comp.error_total_pct:+.1f}%")

    print("\nWorst Predictions (highest error):")
    for comp in comparisons_sorted[-3:]:
        model_name = comp.model_id.split('/')[-1]
        print(f"  ❌ {model_name} (in:{comp.input_len} out:{comp.output_len} bs:{comp.batch_size}): {comp.error_total_pct:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare batch profile results with calculator estimates (v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'csv_file',
        type=Path,
        help='CSV file from batch-profile-bench-v3.py'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=15.0,
        help='Error threshold percentage for pass/fail (default: 15%%)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed per-profile analysis'
    )
    parser.add_argument(
        '--use-proposed',
        action='store_true',
        help='Use proposed calculator formulas instead of current'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save report to file'
    )

    args = parser.parse_args()

    # Check if calculator is available
    if not CALCULATOR_AVAILABLE:
        sys.exit(1)

    # Load results
    print(f"📊 Loading batch results from: {args.csv_file}")
    results = load_batch_results(args.csv_file)
    print(f"   Found {len(results)} successful profiles")

    # Compare each result
    print(f"\n🔍 Comparing with calculator estimates...")
    comparisons = []

    for i, row in enumerate(results, 1):
        print(f"\r   Processing {i}/{len(results)}...", end='', flush=True)
        comp = compare_result(row, use_proposed=args.use_proposed)
        if comp:
            comparisons.append(comp)

    print(f"\r   ✓ Processed {len(comparisons)} valid comparisons")

    if not comparisons:
        print("\n❌ No valid comparisons could be made")
        print("   Check that:")
        print("   - CSV file contains successful profiles")
        print("   - models.json has architecture data for these models")
        print("   - Calculator formulas are accessible")
        sys.exit(1)

    # Redirect output to file if requested
    original_stdout = sys.stdout
    if args.output:
        sys.stdout = open(args.output, 'w')
        print(f"Batch Profile vs Calculator Comparison Report")
        print(f"Generated from: {args.csv_file}")
        print(f"Using: {'Proposed' if args.use_proposed else 'Current'} calculator formulas")
        print(f"Timestamp: {__import__('datetime').datetime.now()}")

    try:
        # Print analyses
        print_summary_statistics(comparisons, args.threshold)

        if args.detailed:
            print_detailed_analysis(comparisons, args.threshold)

        print_pattern_analysis(comparisons)
        print_recommendations(comparisons, args.threshold)

        print("\n" + "=" * 80)

    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"\n📝 Report saved to: {args.output}")


if __name__ == '__main__':
    main()
