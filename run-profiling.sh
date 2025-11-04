#/bin/bash

python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/batch-scaling-study.yaml \
  --results-dir results/batch-scaling-study

python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/moe-study.yaml \
  --results-dir results/moe-study
  
python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/multi-gpu-study.yaml \
  --results-dir results/multi-gpu-study
  
python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/quantization-study.yaml \
  --results-dir results/quantization-study
  
python scripts/batch-profile-bench-enhanced.py \
  --config scripts/configs/sequence-scaling-study.yaml \
  --results-dir results/sequence-scaling-study
