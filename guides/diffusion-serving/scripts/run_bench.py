"""Thin launcher for the upstream vllm-omni diffusion benchmark client.

Usage: python run_bench.py --seed-arrivals N -- <diffusion_benchmark_serving.py args>

Why it exists: the upstream script draws Poisson inter-arrival times from the
UNSEEDED global `random` module (diffusion_benchmark_serving.py, iter_requests),
while the request-profile mix uses its own `random.Random(42)` instance. Seeding
the global RNG here makes arrival times deterministic per repetition without
touching the profile mix — so every arm replays the identical request sequence
AND the identical arrival timeline for a given (num_prompts, rate, seed).
"""

import os
import random
import runpy
import sys


def main() -> None:
    args = sys.argv[1:]
    seed = None
    if args and args[0] == "--seed-arrivals":
        seed = int(args[1])
        args = args[2:]
    if args and args[0] == "--":
        args = args[1:]
    if seed is not None:
        random.seed(seed)

    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusion_benchmark_serving.py")
    sys.argv = [target] + args
    # Make the sibling `from backends import ...` resolve regardless of cwd.
    sys.path.insert(0, os.path.dirname(target))
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
