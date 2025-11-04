#!/usr/bin/env python3
"""
Config Migration Script - v1.0 → v2.1 Format

Migrates old YAML config files from the legacy format (with explicit model
architecture parameters) to the new format (using hf_model_id only).

Usage:
    # Dry run (shows what would be changed)
    python scripts/migrate-configs.py --dry-run

    # Migrate all configs (creates backups)
    python scripts/migrate-configs.py

    # Migrate specific file
    python scripts/migrate-configs.py --file scripts/configs/models-example.yaml

    # Skip backups (dangerous!)
    python scripts/migrate-configs.py --no-backup
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import model loader to validate models exist
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
try:
    from model_loader import validate_model_in_db, load_models_db
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    print("⚠️  Warning: model_loader not available, model validation will be skipped")


# Fields to remove (architecture parameters that should come from models.json)
LEGACY_FIELDS = {
    'model_params',
    'num_layers',
    'num_heads',
    'head_dim',
    'hidden_size',
    'intermediate_size',
    'vocab_size',
    'num_kv_heads',
    'num_experts',
    'experts_per_token',
}

# Fields to keep (test configuration)
KEEP_FIELDS = {
    'hf_model_id',          # NEW: replaces model_id
    'container_name',
    'input_lengths',
    'output_lengths',
    'batch_sizes',
    'dtype',
    'tensor_parallel_size',
    'quantization',
    'kv_cache_dtype',
    'trust_remote_code',
    'gpu_ids',
}


def migrate_model_config(model_data: Dict[str, Any], validate: bool = True) -> tuple[Dict[str, Any], List[str]]:
    """
    Migrate a single model configuration from old to new format

    Args:
        model_data: Original model configuration
        validate: Whether to validate model exists in models.json

    Returns:
        (migrated_config, warnings)
    """
    warnings = []
    migrated = {}

    # Handle model_id → hf_model_id rename
    if 'hf_model_id' in model_data:
        hf_model_id = model_data['hf_model_id']
    elif 'model_id' in model_data:
        hf_model_id = model_data['model_id']
        warnings.append(f"Renamed 'model_id' → 'hf_model_id' for {hf_model_id}")
    else:
        warnings.append("ERROR: No model_id or hf_model_id found")
        return model_data, warnings

    migrated['hf_model_id'] = hf_model_id

    # Validate model exists in models.json
    if validate and MODEL_LOADER_AVAILABLE:
        if not validate_model_in_db(hf_model_id, verbose=False):
            warnings.append(f"⚠️  Model NOT FOUND in models.json: {hf_model_id}")

    # Copy over keep fields
    for field in KEEP_FIELDS:
        if field in model_data and field != 'hf_model_id':  # Already added
            migrated[field] = model_data[field]

    # Track removed fields
    removed_fields = []
    for field in model_data:
        if field in LEGACY_FIELDS:
            removed_fields.append(field)
        elif field not in KEEP_FIELDS and field != 'model_id':
            # Unknown field - keep it with a warning
            warnings.append(f"Unknown field kept: {field}")
            migrated[field] = model_data[field]

    if removed_fields:
        warnings.append(f"Removed architecture fields: {', '.join(removed_fields)}")

    return migrated, warnings


def migrate_config_file(
    input_file: Path,
    output_file: Path = None,
    dry_run: bool = False,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Migrate entire config file

    Args:
        input_file: Path to input YAML file
        output_file: Path to output file (defaults to input_file)
        dry_run: If True, don't write changes
        validate: Whether to validate models exist

    Returns:
        Migration statistics
    """
    if output_file is None:
        output_file = input_file

    # Read original file
    with open(input_file, 'r') as f:
        config = yaml.safe_load(f)

    if 'models' not in config:
        return {
            'file': str(input_file),
            'status': 'skipped',
            'reason': 'No models key found'
        }

    # Migrate each model
    migrated_models = []
    all_warnings = []
    models_changed = 0

    for i, model_data in enumerate(config['models']):
        migrated, warnings = migrate_model_config(model_data, validate=validate)

        # Check if anything changed
        if migrated != model_data:
            models_changed += 1

        migrated_models.append(migrated)
        if warnings:
            all_warnings.extend([f"  Model {i+1}: {w}" for w in warnings])

    # Create new config
    new_config = config.copy()
    new_config['models'] = migrated_models

    # Write output
    stats = {
        'file': str(input_file),
        'total_models': len(migrated_models),
        'models_changed': models_changed,
        'warnings': all_warnings,
        'status': 'success'
    }

    if not dry_run:
        with open(output_file, 'w') as f:
            yaml.dump(
                new_config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
        stats['written'] = str(output_file)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate config files from v1.0 to v2.1 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--file', '-f',
        type=Path,
        help='Specific file to migrate (default: all files in scripts/configs/)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be changed without writing files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files (default: creates .bak files)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validating models exist in models.json'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for migrated files (default: overwrite in place)'
    )

    args = parser.parse_args()

    # Find files to migrate
    if args.file:
        files = [args.file]
    else:
        configs_dir = Path(__file__).parent / 'configs'
        files = sorted(configs_dir.glob('*.yaml'))

    if not files:
        print("No YAML files found to migrate")
        sys.exit(1)

    print("=" * 80)
    print("CONFIG MIGRATION: v1.0 → v2.1")
    print("=" * 80)
    print(f"Files to process: {len(files)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Validate models: {not args.no_validate}")
    print(f"Create backups: {not args.no_backup}")
    print()

    # Process each file
    all_stats = []
    for input_file in files:
        print(f"\n{'='*80}")
        print(f"Processing: {input_file.name}")
        print(f"{'='*80}")

        # Create backup
        if not args.dry_run and not args.no_backup:
            backup_file = input_file.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak')
            print(f"  Creating backup: {backup_file.name}")
            backup_file.write_text(input_file.read_text())

        # Determine output file
        if args.output_dir:
            output_file = args.output_dir / input_file.name
            args.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_file = input_file

        # Migrate
        try:
            stats = migrate_config_file(
                input_file,
                output_file,
                dry_run=args.dry_run,
                validate=not args.no_validate
            )
            all_stats.append(stats)

            # Print results
            if stats['status'] == 'skipped':
                print(f"  ⏭️  Skipped: {stats.get('reason', 'unknown')}")
            else:
                print(f"  ✓ Models: {stats['total_models']}")
                print(f"  ✓ Changed: {stats['models_changed']}")

                if stats['warnings']:
                    print(f"\n  Warnings:")
                    for warning in stats['warnings']:
                        print(f"    {warning}")

                if not args.dry_run:
                    print(f"\n  ✓ Written to: {stats['written']}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({
                'file': str(input_file),
                'status': 'error',
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*80}")
    print("MIGRATION SUMMARY")
    print(f"{'='*80}")

    total_files = len(all_stats)
    successful = sum(1 for s in all_stats if s['status'] == 'success')
    errors = sum(1 for s in all_stats if s['status'] == 'error')
    skipped = sum(1 for s in all_stats if s['status'] == 'skipped')

    print(f"Total files: {total_files}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Errors: {errors}")

    if args.dry_run:
        print(f"\n💡 This was a DRY RUN - no files were modified")
        print(f"   Remove --dry-run to apply changes")
    else:
        print(f"\n✅ Migration complete!")
        if not args.no_backup:
            print(f"   Backups created with .bak extension")

    # Exit with error if any migrations failed
    sys.exit(1 if errors > 0 else 0)


if __name__ == '__main__':
    main()
