#!/usr/bin/env python3
"""
Quick Validation Script for Profiling Results

This script checks profiling JSON files for common issues:
- Contaminated baselines (> 5 GB)
- Impossible values (weights > total)
- Component sum validation
- Clean vs contaminated run detection

Usage:
    python scripts/validate-profiling-results.py results/priority-profiles/*.json
    python scripts/validate-profiling-results.py results/test-cleanup/*.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


class ResultValidator:
    """Validate profiling results for data quality issues"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []
    
    def validate_file(self, filepath: Path) -> Dict[str, Any]:
        """Validate a single profiling result file"""
        
        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            return {
                'file': filepath.name,
                'status': 'error',
                'message': f'Failed to load JSON: {e}'
            }
        
        # Extract key metrics
        mem_meas = data.get('memory_measurements', {})
        mem_breakdown = data.get('memory_breakdown', {})
        
        baseline_gb = mem_meas.get('baseline', {}).get('total_gb', 0)
        peak_gb = mem_meas.get('peak', {}).get('total_gb', 0)
        increase_gb = mem_meas.get('memory_increase_gb', 0)
        
        weights_gb = mem_breakdown.get('model_weights_gb', 0)
        kv_gb = mem_breakdown.get('kv_cache_gb', 0)
        activations_gb = mem_breakdown.get('activations_gb', 0)
        overhead_gb = mem_breakdown.get('framework_overhead_gb', 0)
        
        # Validation checks
        issues = []
        warnings = []
        
        # Check 1: Contaminated baseline
        if baseline_gb > 5.0:
            issues.append(f"CONTAMINATED BASELINE: {baseline_gb:.2f} GB (should be < 5 GB)")
        elif baseline_gb > 2.0:
            warnings.append(f"High baseline: {baseline_gb:.2f} GB (expected < 2 GB)")
        
        # Check 2: Impossible weight values
        if weights_gb > peak_gb and peak_gb > 0:
            issues.append(f"IMPOSSIBLE: Weights ({weights_gb:.2f} GB) > Total ({peak_gb:.2f} GB)")
        
        # Check 3: Negative increase
        if increase_gb < -10:  # Allow small negatives for measurement noise
            issues.append(f"INVALID: Negative memory increase ({increase_gb:.2f} GB)")
        
        # Check 4: Component sum validation
        component_sum = weights_gb + kv_gb + activations_gb + overhead_gb
        if peak_gb > 0:
            diff_pct = abs(component_sum - peak_gb) / peak_gb * 100
            if diff_pct > 20:
                warnings.append(f"Component sum ({component_sum:.2f} GB) differs from total ({peak_gb:.2f} GB) by {diff_pct:.1f}%")
        
        # Check 5: Very low peak (model likely cleaned up before measurement)
        if peak_gb < 5.0 and weights_gb > 10.0:
            issues.append(f"INVALID: Peak too low ({peak_gb:.2f} GB) for model size ({weights_gb:.2f} GB)")
        
        # Determine status
        if issues:
            status = 'failed'
            self.issues.extend(issues)
        elif warnings:
            status = 'warning'
            self.warnings.extend(warnings)
        else:
            status = 'ok'
            self.successes.append(filepath.name)
        
        return {
            'file': filepath.name,
            'status': status,
            'baseline_gb': baseline_gb,
            'peak_gb': peak_gb,
            'increase_gb': increase_gb,
            'weights_gb': weights_gb,
            'component_sum_gb': component_sum,
            'issues': issues,
            'warnings': warnings
        }
    
    def print_result(self, result: Dict[str, Any]):
        """Print formatted validation result"""
        
        file = result['file']
        status = result['status']
        
        # Status emoji
        if status == 'ok':
            emoji = '✅'
        elif status == 'warning':
            emoji = '⚠️'
        else:
            emoji = '❌'
        
        print(f"\n{emoji} {file}")
        print(f"   Baseline: {result['baseline_gb']:.2f} GB")
        print(f"   Peak: {result['peak_gb']:.2f} GB")
        print(f"   Increase: {result['increase_gb']:.2f} GB")
        print(f"   Weights: {result['weights_gb']:.2f} GB")
        print(f"   Component Sum: {result['component_sum_gb']:.2f} GB")
        
        if result['issues']:
            for issue in result['issues']:
                print(f"   ❌ {issue}")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"   ⚠️  {warning}")
        
        if status == 'ok':
            print(f"   ✓ All validation checks passed")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of all validations"""
        
        total = len(results)
        ok = sum(1 for r in results if r['status'] == 'ok')
        warnings = sum(1 for r in results if r['status'] == 'warning')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Total Files: {total}")
        print(f"✅ Passed: {ok} ({ok/total*100:.1f}%)")
        print(f"⚠️  Warnings: {warnings} ({warnings/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        
        if failed == 0 and warnings == 0:
            print("\n🎉 All validations passed! Results are reliable.")
        elif failed == 0:
            print("\n⚠️  Some warnings detected. Review but likely usable.")
        else:
            print("\n❌ Critical issues detected. Results are unreliable.")
            print("\nRecommended actions:")
            print("1. Ensure GPU memory cleanup between runs")
            print("2. Kill vLLM processes before profiling: docker exec <container> pkill -9 -f vllm")
            print("3. Clear CUDA cache: docker exec <container> python3 -c 'import torch; torch.cuda.empty_cache()'")
            print("4. Re-run profiling with cleanup fixes applied")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate-profiling-results.py <json_files...>")
        print("\nExample:")
        print("  python validate-profiling-results.py results/priority-profiles/*.json")
        sys.exit(1)
    
    # Collect files
    files = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists():
            files.append(path)
        else:
            print(f"Warning: File not found: {arg}")
    
    if not files:
        print("Error: No valid files provided")
        sys.exit(1)
    
    print(f"Validating {len(files)} profiling result(s)...\n")
    
    # Validate each file
    validator = ResultValidator()
    results = []
    
    for filepath in sorted(files):
        result = validator.validate_file(filepath)
        results.append(result)
        validator.print_result(result)
    
    # Print summary
    validator.print_summary(results)
    
    # Exit code
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == '__main__':
    main()
