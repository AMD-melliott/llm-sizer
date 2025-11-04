#!/usr/bin/env python3
"""
Validation Results Analysis Tool

Analyzes validation CSV to identify patterns, trends, and calculator accuracy issues.

Usage:
    python scripts/analyze-validation.py validation-results.csv
    python scripts/analyze-validation.py validation-results.csv --detailed
    python scripts/analyze-validation.py validation-results.csv --format markdown > report.md
"""

import sys
import csv
from collections import defaultdict
from typing import Dict, List, Any
import statistics


def load_validation_csv(filepath: str) -> List[Dict[str, Any]]:
    """Load validation CSV and convert numeric fields"""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key in ['params_b', 'gpus', 'batch', 'input_len', 'output_len', 'seq_len']:
                    row[key] = float(row[key]) if row[key] else 0
                elif '_gb' in key or '_pct' in key or 'latency_ms' in key:
                    row[key] = float(row[key]) if row[key] else 0.0
            results.append(row)
    return results


def categorize_by_size(params_b: float) -> str:
    """Categorize model by parameter size"""
    if params_b < 10:
        return "Small (< 10B)"
    elif params_b < 20:
        return "Medium (10-20B)"
    elif params_b < 40:
        return "Large (20-40B)"
    elif params_b < 80:
        return "XLarge (40-80B)"
    else:
        return "XXLarge (80B+)"


def analyze_overall(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze overall statistics"""
    total = len(results)
    
    # Count by match quality
    quality_counts = defaultdict(int)
    for r in results:
        quality = r['match_quality'].split()[0]  # Get "Excellent", "Good", etc.
        quality_counts[quality] += 1
    
    # Calculate average errors
    total_errors = [abs(r['diff_total_pct']) for r in results]
    weights_errors = [abs(r['diff_weights_pct']) for r in results]
    kv_errors = [abs(r['diff_kv_pct']) for r in results]
    activations_errors = [abs(r['diff_activations_pct']) for r in results]
    overhead_errors = [abs(r['diff_overhead_pct']) for r in results]
    
    return {
        'total_profiles': total,
        'quality_counts': dict(quality_counts),
        'avg_total_error': statistics.mean(total_errors),
        'median_total_error': statistics.median(total_errors),
        'max_total_error': max(total_errors),
        'avg_weights_error': statistics.mean(weights_errors),
        'avg_kv_error': statistics.mean(kv_errors),
        'avg_activations_error': statistics.mean(activations_errors),
        'avg_overhead_error': statistics.mean(overhead_errors),
        'worst_profile': max(results, key=lambda r: abs(r['diff_total_pct'])),
        'best_profile': min(results, key=lambda r: abs(r['diff_total_pct'])),
    }


def analyze_by_size(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze errors by model size category"""
    by_size = defaultdict(list)
    
    for r in results:
        category = categorize_by_size(r['params_b'])
        by_size[category].append(r)
    
    analysis = {}
    for category, profiles in by_size.items():
        if not profiles:
            continue
        
        total_errors = [abs(p['diff_total_pct']) for p in profiles]
        weights_errors = [abs(p['diff_weights_pct']) for p in profiles]
        kv_errors = [abs(p['diff_kv_pct']) for p in profiles]
        
        analysis[category] = {
            'count': len(profiles),
            'avg_params': statistics.mean([p['params_b'] for p in profiles]),
            'avg_total_error': statistics.mean(total_errors),
            'avg_weights_error': statistics.mean(weights_errors),
            'avg_kv_error': statistics.mean(kv_errors),
            'avg_activations_error': statistics.mean([abs(p['diff_activations_pct']) for p in profiles]),
        }
    
    return analysis


def analyze_by_gpus(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Analyze errors by GPU count"""
    by_gpus = defaultdict(list)
    
    for r in results:
        by_gpus[int(r['gpus'])].append(r)
    
    analysis = {}
    for gpu_count, profiles in sorted(by_gpus.items()):
        if not profiles:
            continue
        
        total_errors = [abs(p['diff_total_pct']) for p in profiles]
        weights_errors = [abs(p['diff_weights_pct']) for p in profiles]
        
        analysis[gpu_count] = {
            'count': len(profiles),
            'avg_total_error': statistics.mean(total_errors),
            'avg_weights_error': statistics.mean(weights_errors),
            'avg_kv_error': statistics.mean([abs(p['diff_kv_pct']) for p in profiles]),
            'avg_activations_error': statistics.mean([abs(p['diff_activations_pct']) for p in profiles]),
        }
    
    return analysis


def analyze_by_component(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze which components have the worst errors"""
    components = {
        'weights': [],
        'kv_cache': [],
        'activations': [],
        'overhead': [],
    }
    
    for r in results:
        components['weights'].append({
            'profile': r['profile_file'],
            'model': r['model_name'],
            'error_pct': r['diff_weights_pct'],
            'profiled': r['profiled_weights_gb'],
            'calculated': r['calculator_weights_gb'],
        })
        components['kv_cache'].append({
            'profile': r['profile_file'],
            'model': r['model_name'],
            'error_pct': r['diff_kv_pct'],
            'profiled': r['profiled_kv_gb'],
            'calculated': r['calculator_kv_gb'],
        })
        components['activations'].append({
            'profile': r['profile_file'],
            'model': r['model_name'],
            'error_pct': r['diff_activations_pct'],
            'profiled': r['profiled_activations_gb'],
            'calculated': r['calculator_activations_gb'],
        })
        components['overhead'].append({
            'profile': r['profile_file'],
            'model': r['model_name'],
            'error_pct': r['diff_overhead_pct'],
            'profiled': r['profiled_overhead_gb'],
            'calculated': r['calculator_overhead_gb'],
        })
    
    # Find worst performers for each component
    analysis = {}
    for comp_name, comp_data in components.items():
        avg_error = statistics.mean([abs(c['error_pct']) for c in comp_data])
        worst = max(comp_data, key=lambda c: abs(c['error_pct']))
        
        # Determine if calculator over or under estimates
        over_count = sum(1 for c in comp_data if c['error_pct'] < 0)  # negative = profiled < calc
        under_count = len(comp_data) - over_count
        bias = "OVERESTIMATES" if over_count > under_count else "UNDERESTIMATES"
        
        analysis[comp_name] = {
            'avg_error_pct': avg_error,
            'bias': bias,
            'over_count': over_count,
            'under_count': under_count,
            'worst_case': worst,
        }
    
    return analysis


def identify_issues(results: List[Dict[str, Any]], overall: Dict[str, Any], 
                   by_component: Dict[str, Any]) -> List[str]:
    """Identify top issues based on analysis"""
    issues = []
    
    # Issue 1: Overall accuracy
    if overall['avg_total_error'] > 20:
        issues.append(
            f"CRITICAL: Average total error is {overall['avg_total_error']:.1f}%, "
            f"far exceeding acceptable threshold of 10-15%"
        )
    elif overall['avg_total_error'] > 10:
        issues.append(
            f"WARNING: Average total error is {overall['avg_total_error']:.1f}%, "
            f"above target threshold of 10%"
        )
    
    # Issue 2: Component-specific issues
    for comp_name, comp_data in by_component.items():
        if comp_data['avg_error_pct'] > 30:
            issues.append(
                f"CRITICAL: {comp_name.upper()} calculation {comp_data['bias'].lower()} "
                f"by {comp_data['avg_error_pct']:.1f}% on average "
                f"({comp_data['over_count']} over, {comp_data['under_count']} under)"
            )
    
    # Issue 3: Multi-GPU accuracy
    multi_gpu = [r for r in results if r['gpus'] > 1]
    if multi_gpu:
        multi_gpu_errors = [abs(r['diff_total_pct']) for r in multi_gpu]
        single_gpu = [r for r in results if r['gpus'] == 1]
        if single_gpu:
            single_gpu_errors = [abs(r['diff_total_pct']) for r in single_gpu]
            if statistics.mean(multi_gpu_errors) > statistics.mean(single_gpu_errors) * 1.5:
                issues.append(
                    f"WARNING: Multi-GPU configurations have {statistics.mean(multi_gpu_errors):.1f}% avg error "
                    f"vs {statistics.mean(single_gpu_errors):.1f}% for single-GPU (potential TP scaling issue)"
                )
    
    # Issue 4: Large model accuracy
    large_models = [r for r in results if r['params_b'] > 40]
    if large_models:
        large_errors = [abs(r['diff_total_pct']) for r in large_models]
        if statistics.mean(large_errors) > 25:
            issues.append(
                f"WARNING: Large models (>40B) have {statistics.mean(large_errors):.1f}% avg error, "
                f"suggesting scaling formula issues"
            )
    
    return issues


def format_text_report(results: List[Dict[str, Any]], detailed: bool = False) -> str:
    """Format analysis as text report"""
    lines = []
    
    # Overall statistics
    overall = analyze_overall(results)
    lines.append("=" * 100)
    lines.append("VALIDATION ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append("")
    
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"  Total Profiles Analyzed:  {overall['total_profiles']}")
    lines.append(f"  Average Total Error:      {overall['avg_total_error']:.1f}%")
    lines.append(f"  Median Total Error:       {overall['median_total_error']:.1f}%")
    lines.append(f"  Maximum Total Error:      {overall['max_total_error']:.1f}%")
    lines.append("")
    
    lines.append("  Match Quality Distribution:")
    for quality, count in sorted(overall['quality_counts'].items()):
        pct = (count / overall['total_profiles']) * 100
        lines.append(f"    {quality:<12} {count:>3} ({pct:>5.1f}%)")
    lines.append("")
    
    # Component analysis
    by_component = analyze_by_component(results)
    lines.append("COMPONENT-LEVEL ANALYSIS")
    lines.append("-" * 100)
    lines.append(f"  {'Component':<15} {'Avg Error':<12} {'Bias':<18} {'Over/Under'}")
    lines.append("  " + "-" * 96)
    
    for comp_name in ['weights', 'kv_cache', 'activations', 'overhead']:
        comp_data = by_component[comp_name]
        lines.append(
            f"  {comp_name.replace('_', ' ').title():<15} "
            f"{comp_data['avg_error_pct']:>10.1f}% "
            f"{comp_data['bias']:<18} "
            f"{comp_data['over_count']}/{comp_data['under_count']}"
        )
    lines.append("")
    
    # Worst cases per component
    if detailed:
        lines.append("  Worst Cases by Component:")
        for comp_name, comp_data in by_component.items():
            worst = comp_data['worst_case']
            lines.append(f"    {comp_name.replace('_', ' ').title()}:")
            lines.append(f"      Model: {worst['model']}")
            lines.append(f"      Error: {worst['error_pct']:+.1f}%")
            lines.append(
                f"      Values: {worst['profiled']:.2f} GB (profiled) vs "
                f"{worst['calculated']:.2f} GB (calculated)"
            )
        lines.append("")
    
    # By model size
    by_size = analyze_by_size(results)
    lines.append("ANALYSIS BY MODEL SIZE")
    lines.append("-" * 100)
    lines.append(
        f"  {'Category':<20} {'Count':<8} {'Avg Params':<12} "
        f"{'Total Err':<12} {'Weights Err':<12} {'KV Err'}"
    )
    lines.append("  " + "-" * 96)
    
    for category in sorted(by_size.keys()):
        data = by_size[category]
        lines.append(
            f"  {category:<20} {data['count']:<8} "
            f"{data['avg_params']:>10.1f}B "
            f"{data['avg_total_error']:>10.1f}% "
            f"{data['avg_weights_error']:>10.1f}% "
            f"{data['avg_kv_error']:>10.1f}%"
        )
    lines.append("")
    
    # By GPU count
    by_gpus = analyze_by_gpus(results)
    lines.append("ANALYSIS BY GPU COUNT")
    lines.append("-" * 100)
    lines.append(
        f"  {'GPUs':<8} {'Count':<8} {'Total Err':<12} "
        f"{'Weights Err':<12} {'KV Err':<12} {'Act Err'}"
    )
    lines.append("  " + "-" * 96)
    
    for gpu_count, data in sorted(by_gpus.items()):
        lines.append(
            f"  {gpu_count:<8} {data['count']:<8} "
            f"{data['avg_total_error']:>10.1f}% "
            f"{data['avg_weights_error']:>10.1f}% "
            f"{data['avg_kv_error']:>10.1f}% "
            f"{data['avg_activations_error']:>10.1f}%"
        )
    lines.append("")
    
    # Top issues
    issues = identify_issues(results, overall, by_component)
    lines.append("IDENTIFIED ISSUES")
    lines.append("-" * 100)
    if issues:
        for i, issue in enumerate(issues, 1):
            lines.append(f"  {i}. {issue}")
    else:
        lines.append("  ✓ No major issues identified. Calculator accuracy is good!")
    lines.append("")
    
    # Best and worst profiles
    lines.append("BEST & WORST PROFILES")
    lines.append("-" * 100)
    
    worst = overall['worst_profile']
    lines.append(f"  Worst Match: {worst['model_name']}")
    lines.append(f"    File: {worst['profile_file']}")
    lines.append(f"    Error: {worst['diff_total_pct']:+.1f}%")
    lines.append(
        f"    Memory: {worst['profiled_total_gb']:.2f} GB (profiled) vs "
        f"{worst['calculator_total_gb']:.2f} GB (calculated)"
    )
    lines.append("")
    
    best = overall['best_profile']
    lines.append(f"  Best Match: {best['model_name']}")
    lines.append(f"    File: {best['profile_file']}")
    lines.append(f"    Error: {best['diff_total_pct']:+.1f}%")
    lines.append(
        f"    Memory: {best['profiled_total_gb']:.2f} GB (profiled) vs "
        f"{best['calculator_total_gb']:.2f} GB (calculated)"
    )
    lines.append("")
    
    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 100)
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if by_component['weights']['bias'] == 'OVERESTIMATES' and by_component['weights']['avg_error_pct'] > 20:
        recommendations.append(
            "1. WEIGHTS: Calculator overestimates model weights significantly. "
            "Check if weight calculation is multiplying by GPU count incorrectly for tensor parallelism."
        )
    
    if by_component['kv_cache']['avg_error_pct'] > 30:
        if by_component['kv_cache']['bias'] == 'UNDERESTIMATES':
            recommendations.append(
                "2. KV CACHE: Calculator underestimates KV cache. "
                "Verify the KV cache formula accounts for batch size, sequence length, and attention mechanism correctly."
            )
        else:
            recommendations.append(
                "2. KV CACHE: Calculator overestimates KV cache. "
                "Check if formula accounts for multi-GPU distribution or PagedAttention optimizations."
            )
    
    if by_component['activations']['avg_error_pct'] > 30:
        recommendations.append(
            "3. ACTIVATIONS: Activation memory calculation has high error. "
            "Review formula for batch size, sequence length, and multi-GPU scaling."
        )
    
    if len([r for r in results if r['gpus'] > 1]) > 0:
        multi_gpu_weight_errors = [
            abs(r['diff_weights_pct']) for r in results if r['gpus'] > 1
        ]
        if statistics.mean(multi_gpu_weight_errors) > 30:
            recommendations.append(
                "4. MULTI-GPU: Multi-GPU tensor parallelism calculations appear incorrect. "
                "Weights should be DIVIDED across GPUs, not multiplied."
            )
    
    if overall['avg_total_error'] < 10:
        recommendations.append(
            "✓ Overall accuracy is excellent! Continue gathering more data for edge cases."
        )
    
    for rec in recommendations:
        lines.append(f"  {rec}")
        lines.append("")
    
    lines.append("=" * 100)
    
    return '\n'.join(lines)


def format_markdown_report(results: List[Dict[str, Any]]) -> str:
    """Format analysis as Markdown report"""
    overall = analyze_overall(results)
    by_component = analyze_by_component(results)
    by_size = analyze_by_size(results)
    by_gpus = analyze_by_gpus(results)
    issues = identify_issues(results, overall, by_component)
    
    lines = [
        "# Calculator Validation Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"- **Total Profiles**: {overall['total_profiles']}",
        f"- **Average Error**: {overall['avg_total_error']:.1f}%",
        f"- **Median Error**: {overall['median_total_error']:.1f}%",
        f"- **Max Error**: {overall['max_total_error']:.1f}%",
        "",
        "### Match Quality Distribution",
        "",
    ]
    
    for quality, count in sorted(overall['quality_counts'].items()):
        pct = (count / overall['total_profiles']) * 100
        lines.append(f"- **{quality}**: {count} ({pct:.1f}%)")
    
    lines.extend([
        "",
        "## Component Analysis",
        "",
        "| Component | Avg Error | Bias | Over/Under |",
        "|-----------|-----------|------|------------|",
    ])
    
    for comp_name in ['weights', 'kv_cache', 'activations', 'overhead']:
        comp_data = by_component[comp_name]
        lines.append(
            f"| {comp_name.replace('_', ' ').title()} | "
            f"{comp_data['avg_error_pct']:.1f}% | "
            f"{comp_data['bias']} | "
            f"{comp_data['over_count']}/{comp_data['under_count']} |"
        )
    
    lines.extend([
        "",
        "## Analysis by Model Size",
        "",
        "| Category | Count | Avg Params | Total Error | Weights Error | KV Error |",
        "|----------|-------|------------|-------------|---------------|----------|",
    ])
    
    for category in sorted(by_size.keys()):
        data = by_size[category]
        lines.append(
            f"| {category} | {data['count']} | "
            f"{data['avg_params']:.1f}B | "
            f"{data['avg_total_error']:.1f}% | "
            f"{data['avg_weights_error']:.1f}% | "
            f"{data['avg_kv_error']:.1f}% |"
        )
    
    lines.extend([
        "",
        "## Analysis by GPU Count",
        "",
        "| GPUs | Count | Total Error | Weights Error | KV Error | Activations Error |",
        "|------|-------|-------------|---------------|----------|-------------------|",
    ])
    
    for gpu_count, data in sorted(by_gpus.items()):
        lines.append(
            f"| {gpu_count} | {data['count']} | "
            f"{data['avg_total_error']:.1f}% | "
            f"{data['avg_weights_error']:.1f}% | "
            f"{data['avg_kv_error']:.1f}% | "
            f"{data['avg_activations_error']:.1f}% |"
        )
    
    lines.extend([
        "",
        "## Identified Issues",
        "",
    ])
    
    if issues:
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. {issue}")
    else:
        lines.append("✓ No major issues identified.")
    
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    
    # Same recommendation logic as text format
    if by_component['weights']['bias'] == 'OVERESTIMATES' and by_component['weights']['avg_error_pct'] > 20:
        lines.append(
            "1. **WEIGHTS**: Calculator overestimates model weights. "
            "Check tensor parallelism weight distribution logic."
        )
    
    if by_component['kv_cache']['avg_error_pct'] > 30:
        lines.append(
            "2. **KV CACHE**: Calculator has significant KV cache error. "
            "Review formula for batch size and sequence length handling."
        )
    
    if by_component['activations']['avg_error_pct'] > 30:
        lines.append(
            "3. **ACTIVATIONS**: Activation memory calculation needs review."
        )
    
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    csv_file = sys.argv[1]
    detailed = '--detailed' in sys.argv
    format_type = 'text'
    
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            format_type = sys.argv[idx + 1]
    
    try:
        results = load_validation_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    if len(results) == 0:
        print("Error: No results found in CSV")
        sys.exit(1)
    
    if format_type == 'markdown':
        print(format_markdown_report(results))
    else:
        print(format_text_report(results, detailed))


if __name__ == '__main__':
    main()
