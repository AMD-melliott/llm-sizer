#!/usr/bin/env python3
"""
Model Database Loader

Loads model architecture data from src/data/models.json to provide a single
source of truth for model parameters used in profiling.

This module eliminates data duplication across config files by centralizing
model specifications in the main application's models.json database.

Usage:
    from lib.model_loader import get_model_info, load_models_db

    # Get specific model info
    model = get_model_info("meta-llama/Llama-3.2-1B-Instruct")
    print(f"Layers: {model['num_layers']}")

    # Or load entire database
    models = load_models_db()
    for hf_id, info in models.items():
        print(f"{hf_id}: {info['parameters_billions']}B params")
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any


def get_models_json_path() -> Path:
    """
    Get absolute path to src/data/models.json

    Handles both scenarios:
    - Running from scripts/ directory (development)
    - Running from container with cloned repo

    Returns:
        Path to models.json file
    """
    # Try relative to this file first (scripts/lib/model_loader.py -> src/data/models.json)
    models_file = Path(__file__).parent.parent.parent / 'src' / 'data' / 'models.json'

    if models_file.exists():
        return models_file

    # Try from current working directory (container scenario)
    models_file = Path.cwd() / 'src' / 'data' / 'models.json'

    if models_file.exists():
        return models_file

    # Try common container paths
    for base in ['/app/llm-sizer', '/workspace/llm-sizer', '/app', '/code']:
        models_file = Path(base) / 'src' / 'data' / 'models.json'
        if models_file.exists():
            return models_file

    raise FileNotFoundError(
        "Could not find src/data/models.json. "
        "Ensure llm-sizer repository is properly cloned/mounted. "
        f"Searched: {models_file.parent}"
    )


def load_models_db() -> Dict[str, Dict[str, Any]]:
    """
    Load models.json and index by hf_model_id for fast lookup

    Returns:
        Dictionary mapping HuggingFace model IDs to model info:
        {
            "meta-llama/Llama-3.2-1B-Instruct": {
                "id": "Llama-3.2-1B-Instruct",
                "name": "Llama 3.2 1B Instruct",
                "hf_model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "hidden_size": 2048,
                "num_layers": 16,
                "num_heads": 32,
                "parameters_billions": 1.2358144,
                ...
            },
            ...
        }

    Raises:
        FileNotFoundError: If models.json cannot be found
        json.JSONDecodeError: If models.json is invalid
    """
    models_file = get_models_json_path()

    with open(models_file, 'r') as f:
        data = json.load(f)

    if 'models' not in data or not isinstance(data['models'], list):
        raise ValueError(
            f"Invalid models.json format: expected {{'models': [...]}} structure"
        )

    # Index by hf_model_id for O(1) lookup
    models_by_hf_id = {}

    for model in data['models']:
        if 'hf_model_id' not in model:
            # Skip models without HuggingFace ID (e.g., proprietary models)
            continue

        hf_id = model['hf_model_id']

        if hf_id in models_by_hf_id:
            print(f"WARNING: Duplicate hf_model_id in models.json: {hf_id}")
            print(f"  Keeping first occurrence, ignoring duplicate")
            continue

        models_by_hf_id[hf_id] = model

    return models_by_hf_id


def get_model_info(hf_model_id: str, required_fields: Optional[list] = None) -> Optional[Dict[str, Any]]:
    """
    Get model architecture info by HuggingFace model ID

    Args:
        hf_model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        required_fields: Optional list of required fields to validate

    Returns:
        Model info dictionary, or None if not found

    Example:
        >>> model = get_model_info("meta-llama/Llama-3.2-1B-Instruct")
        >>> print(f"{model['parameters_billions']}B parameters")
        1.2358144B parameters

        >>> # With validation
        >>> model = get_model_info(
        ...     "meta-llama/Llama-3.2-1B-Instruct",
        ...     required_fields=['num_layers', 'hidden_size', 'num_heads']
        ... )
    """
    models = load_models_db()
    model = models.get(hf_model_id)

    if model is None:
        return None

    # Validate required fields if specified
    if required_fields:
        missing_fields = [field for field in required_fields if field not in model]
        if missing_fields:
            print(f"WARNING: Model {hf_model_id} is missing fields: {missing_fields}")
            print(f"  Available fields: {list(model.keys())}")

    return model


def get_profiling_params(hf_model_id: str) -> Dict[str, Any]:
    """
    Extract profiling-relevant parameters from model info

    Converts models.json format to the parameters needed by profile-vllm-bench-enhanced.py

    Args:
        hf_model_id: HuggingFace model identifier

    Returns:
        Dictionary with profiling parameters:
        {
            'model_params': float,      # Total parameters (not billions)
            'num_layers': int,
            'num_heads': int,
            'hidden_size': int,
            'head_dim': int,            # Calculated or from model
            'vocab_size': int,
            'intermediate_size': int,
            'num_kv_heads': int,        # May be None for MHA models
        }

    Raises:
        ValueError: If model not found or missing critical fields
    """
    model = get_model_info(hf_model_id)

    if model is None:
        raise ValueError(
            f"Model not found in models.json: {hf_model_id}\n"
            f"Available models: {list(load_models_db().keys())[:5]}... (showing first 5)"
        )

    # Extract required fields
    required = ['parameters_billions', 'num_layers', 'num_heads', 'hidden_size']
    missing = [f for f in required if f not in model or model[f] is None]

    if missing:
        raise ValueError(
            f"Model {hf_model_id} is missing required fields: {missing}\n"
            f"Available fields: {list(model.keys())}"
        )

    # Convert billions to absolute count
    model_params = model['parameters_billions'] * 1e9

    # Calculate head_dim if not provided
    # Standard transformer: head_dim = hidden_size / num_heads
    if 'head_dim' in model and model['head_dim']:
        head_dim = model['head_dim']
    else:
        head_dim = model['hidden_size'] // model['num_heads']

    return {
        'model_params': model_params,
        'num_layers': model['num_layers'],
        'num_heads': model['num_heads'],
        'hidden_size': model['hidden_size'],
        'head_dim': head_dim,
        'vocab_size': model.get('vocab_size'),
        'intermediate_size': model.get('intermediate_size'),
        'num_kv_heads': model.get('num_kv_heads'),  # For GQA models
    }


def search_models(query: str, limit: int = 10) -> list:
    """
    Search for models by name or ID

    Args:
        query: Search string (case-insensitive)
        limit: Maximum number of results

    Returns:
        List of matching model info dictionaries

    Example:
        >>> models = search_models("llama")
        >>> for m in models:
        ...     print(f"{m['hf_model_id']} - {m['parameters_billions']}B")
    """
    models = load_models_db()
    query_lower = query.lower()

    matches = []
    for hf_id, model_info in models.items():
        if (query_lower in hf_id.lower() or
            query_lower in model_info.get('name', '').lower() or
            query_lower in model_info.get('id', '').lower()):
            matches.append(model_info)
            if len(matches) >= limit:
                break

    return matches


def validate_model_in_db(hf_model_id: str, verbose: bool = True) -> bool:
    """
    Check if a model exists in the database with all required fields

    Args:
        hf_model_id: HuggingFace model identifier
        verbose: Print detailed information about validation

    Returns:
        True if model exists and has all required fields, False otherwise
    """
    try:
        params = get_profiling_params(hf_model_id)

        if verbose:
            print(f"✓ Model validated: {hf_model_id}")
            print(f"  Parameters: {params['model_params']/1e9:.1f}B")
            print(f"  Layers: {params['num_layers']}")
            print(f"  Hidden size: {params['hidden_size']}")
            print(f"  Attention heads: {params['num_heads']}")
            if params['num_kv_heads']:
                print(f"  KV heads: {params['num_kv_heads']} (GQA)")

        return True

    except ValueError as e:
        if verbose:
            print(f"✗ Model validation failed: {hf_model_id}")
            print(f"  Error: {e}")

        return False


if __name__ == '__main__':
    """
    Command-line interface for testing and searching models

    Usage:
        python scripts/lib/model_loader.py                      # List all models
        python scripts/lib/model_loader.py search llama         # Search for models
        python scripts/lib/model_loader.py validate <model_id>  # Validate model
    """
    import sys

    if len(sys.argv) == 1:
        # List all models
        print("Available models in database:")
        print("=" * 80)

        models = load_models_db()
        for hf_id, info in sorted(models.items()):
            params_b = info.get('parameters_billions', 0)
            name = info.get('name', 'Unknown')
            print(f"{hf_id:50} {params_b:6.1f}B  {name}")

        print(f"\nTotal: {len(models)} models")

    elif len(sys.argv) >= 2:
        command = sys.argv[1]

        if command == 'search':
            query = sys.argv[2] if len(sys.argv) > 2 else ""
            print(f"Searching for: {query}")
            print("=" * 80)

            matches = search_models(query)
            for model in matches:
                print(f"{model['hf_model_id']:50} {model['parameters_billions']:6.1f}B")

            print(f"\nFound: {len(matches)} matches")

        elif command == 'validate':
            if len(sys.argv) < 3:
                print("Usage: python model_loader.py validate <hf_model_id>")
                sys.exit(1)

            hf_model_id = sys.argv[2]
            validate_model_in_db(hf_model_id, verbose=True)

        else:
            print(f"Unknown command: {command}")
            print("Available commands: search, validate")
            sys.exit(1)
