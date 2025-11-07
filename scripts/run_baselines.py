#!/usr/bin/env python3
# scripts/run_baselines.py
import subprocess, time

SEEDS = [42, 101, 2025, 7, 99]
for seed in SEEDS:
    print(f"\n=== Running seed {seed} ===")
    t0 = time.time()
    subprocess.run(["python", "scripts/train_xgb.py", "--seed", str(seed), "--subsample", "0.05"], check=True)
    print(f"Seed {seed} done in {time.time()-t0:.1f}s")

print("\nAll baseline runs completed.")
