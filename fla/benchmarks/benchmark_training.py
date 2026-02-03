#!/usr/bin/env python3
"""Benchmark training components.

Run benchmarks:
    python benchmarks/benchmark_training.py

Results are printed to stdout and saved to benchmarks/results/.
"""

import time
import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Ensure FLA is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    iterations: int
    total_time_s: float
    mean_time_ms: float
    std_time_ms: float
    throughput: float  # iterations per second
    device: str
    extra: dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


def benchmark_lora_forward(batch_sizes: list[int] = [1, 4, 16, 32], iterations: int = 100):
    """Benchmark LoRA forward pass."""
    from fla.training import LoRAConfig, LoRALinear

    results = []
    config = LoRAConfig(rank=16, alpha=16.0)

    for batch_size in batch_sizes:
        rngs = nnx.Rngs(42)
        layer = LoRALinear(
            in_features=1024,
            out_features=256,
            config=config,
            rngs=rngs,
        )

        # Warmup
        x = jax.random.normal(jax.random.key(0), (batch_size, 32, 1024))
        _ = layer(x)
        jax.block_until_ready(_)

        # Benchmark
        times = []
        for i in range(iterations):
            x = jax.random.normal(jax.random.key(i), (batch_size, 32, 1024))
            start = time.perf_counter()
            out = layer(x)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - start)

        times_ms = [t * 1000 for t in times]
        result = BenchmarkResult(
            name=f"lora_forward_bs{batch_size}",
            iterations=iterations,
            total_time_s=sum(times),
            mean_time_ms=sum(times_ms) / len(times_ms),
            std_time_ms=(sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
            throughput=iterations / sum(times),
            device=str(jax.devices()[0]),
            extra={"batch_size": batch_size, "in_features": 1024, "out_features": 256, "rank": 16},
        )
        results.append(result)
        print(f"  LoRA forward (bs={batch_size}): {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")

    return results


def benchmark_knowledge_insulation(batch_sizes: list[int] = [1, 4, 16, 32], iterations: int = 100):
    """Benchmark knowledge insulation operations."""
    from fla.training import KnowledgeInsulationConfig, apply_knowledge_insulation

    results = []

    for mode in ["full", "soft"]:
        config = KnowledgeInsulationConfig(mode=mode, gradient_scale=0.1)

        for batch_size in batch_sizes:
            # Warmup
            tokens = jnp.ones((batch_size, 100, 1024))
            _ = apply_knowledge_insulation(tokens, config)
            jax.block_until_ready(_)

            # Benchmark
            times = []
            for i in range(iterations):
                tokens = jax.random.normal(jax.random.key(i), (batch_size, 100, 1024))
                start = time.perf_counter()
                out = apply_knowledge_insulation(tokens, config)
                jax.block_until_ready(out)
                times.append(time.perf_counter() - start)

            times_ms = [t * 1000 for t in times]
            result = BenchmarkResult(
                name=f"ki_{mode}_bs{batch_size}",
                iterations=iterations,
                total_time_s=sum(times),
                mean_time_ms=sum(times_ms) / len(times_ms),
                std_time_ms=(sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
                throughput=iterations / sum(times),
                device=str(jax.devices()[0]),
                extra={"batch_size": batch_size, "mode": mode, "seq_len": 100, "dim": 1024},
            )
            results.append(result)
            print(f"  KI {mode} (bs={batch_size}): {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")

    return results


def benchmark_discrete_state_encoder(batch_sizes: list[int] = [1, 4, 16, 32], iterations: int = 100):
    """Benchmark discrete state encoder."""
    from fla.training import DiscreteStateEncoder

    results = []

    for batch_size in batch_sizes:
        rngs = nnx.Rngs(42)
        encoder = DiscreteStateEncoder(
            state_dim=14,
            num_bins=256,
            embedding_dim=1024,
            rngs=rngs,
        )

        # Warmup
        state = jax.random.uniform(jax.random.key(0), (batch_size, 14), minval=-1, maxval=1)
        _ = encoder(state)
        jax.block_until_ready(_)

        # Benchmark
        times = []
        for i in range(iterations):
            state = jax.random.uniform(jax.random.key(i), (batch_size, 14), minval=-1, maxval=1)
            start = time.perf_counter()
            out = encoder(state)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - start)

        times_ms = [t * 1000 for t in times]
        result = BenchmarkResult(
            name=f"discrete_encoder_bs{batch_size}",
            iterations=iterations,
            total_time_s=sum(times),
            mean_time_ms=sum(times_ms) / len(times_ms),
            std_time_ms=(sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
            throughput=iterations / sum(times),
            device=str(jax.devices()[0]),
            extra={"batch_size": batch_size, "state_dim": 14, "num_bins": 256},
        )
        results.append(result)
        print(f"  Discrete encoder (bs={batch_size}): {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")

    return results


def benchmark_reinflow_gae(trajectory_lengths: list[int] = [10, 50, 100, 500], iterations: int = 100):
    """Benchmark GAE computation."""
    from fla.training.reinflow import compute_gae, compute_returns

    results = []

    for traj_len in trajectory_lengths:
        batch_size = 4

        # Create trajectory data
        rewards = [jnp.ones((batch_size,)) for _ in range(traj_len)]
        values = [jnp.ones((batch_size,)) * 0.5 for _ in range(traj_len + 1)]
        dones = [jnp.zeros((batch_size,), dtype=jnp.bool_) for _ in range(traj_len)]

        # Warmup
        _ = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        # Benchmark
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
            # Force computation
            _ = [jax.block_until_ready(a) for a in advantages]
            times.append(time.perf_counter() - start)

        times_ms = [t * 1000 for t in times]
        result = BenchmarkResult(
            name=f"gae_len{traj_len}",
            iterations=iterations,
            total_time_s=sum(times),
            mean_time_ms=sum(times_ms) / len(times_ms),
            std_time_ms=(sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
            throughput=iterations / sum(times),
            device=str(jax.devices()[0]),
            extra={"trajectory_length": traj_len, "batch_size": batch_size},
        )
        results.append(result)
        print(f"  GAE (len={traj_len}): {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")

    return results


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("FLA Training Benchmarks")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"JAX version: {jax.__version__}")
    print()

    all_results = []

    print("LoRA Forward Pass:")
    all_results.extend(benchmark_lora_forward())
    print()

    print("Knowledge Insulation:")
    all_results.extend(benchmark_knowledge_insulation())
    print()

    print("Discrete State Encoder:")
    all_results.extend(benchmark_discrete_state_encoder())
    print()

    print("GAE Computation:")
    all_results.extend(benchmark_reinflow_gae())
    print()

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"benchmark_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    print(f"Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
