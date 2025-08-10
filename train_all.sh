#!/bin/bash

for dataset in moons circles blobs classification gaussian; do
  for seed in {1..15}; do
    python train.py --dataset $dataset --seed $seed --output "model_${dataset}_${seed}.pkl"
  done
done