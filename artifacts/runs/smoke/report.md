# Switchback results

Generated from raw request measurements. Do not edit numerical values by hand.

Source commit: `066f1b844a5720749b7421f8493441a03728d092`. Baseline: `hf_ar`.

```json
{
  "benchmark_source_sha256": "4ce0021dc593662d2bb6399faf54578ddd9d6bf44765e777f08cc8031932926c",
  "calibration_sha256": "b3f6a4c569380df8c123f5dde09ae610adf269892a79c71318dd8f457573e286",
  "config_sha256": "0135db34c3099794b5a797a66c88142dabc7a9983a93cf23d782490dd8914c02",
  "files": {
    "evidence.json": "878f25befa128a24b3f1eef0c6d5f9e2eb3289450e4cd68f40711d8286153dc3",
    "requests.jsonl": "8ac45904c079e75749a2a870aaca8d42e5e6084afa6b4553f3c8e24c0517fc50"
  },
  "hardware": {
    "capability": [
      12,
      1
    ],
    "device": "NVIDIA GB10",
    "driver_version": "580.142",
    "hostname": "gigi-spark",
    "machine": "aarch64",
    "resident_allocated_bytes": 9293824000,
    "resident_reserved_bytes": 9317646336,
    "total_memory_bytes": 130663165952
  },
  "model_revisions": {
    "draft": "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
    "target": "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
  },
  "software": {
    "attn_implementation": "sdpa",
    "cuda_build": "13.0",
    "deterministic_knobs": {
      "allow_tf32_cudnn": false,
      "allow_tf32_matmul": false,
      "cublas_workspace_config": ":4096:8",
      "cudnn_benchmark": false,
      "float32_matmul_precision": "highest"
    },
    "dtype": "bfloat16",
    "huggingface-hub": "0.36.2",
    "numpy": "2.5.3",
    "python": "3.12.3",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.14.0+cu130",
    "torch_native_overrides": {
      "disabled_dsl": [
        "triton"
      ],
      "disabled_ops": [
        "bmm"
      ],
      "probed": true,
      "reason": "CPython development headers absent (/usr/include/python3.12/Python.h); install python3-dev to restore the stock Triton routing",
      "triton_overrides_available": false
    },
    "transformers": "4.57.1",
    "triton": "3.8.0"
  },
  "workload_sha256": "eb932ebe192c0cc1f6524444ccb4458bcea49594a357bb2624d025aeb4e3d5c7"
}
```

Complete logical requests: 64. Required engines: 8.

Greedy token equality was checked for every paired request. Sampled outputs were not required to match.

Oracle, cache, and numerical gates: passed according to the linked validation evidence. Finite-model tests do not prove universal GPU equivalence.

Validation commands: /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_oracle.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_sampled_speculation.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_cache.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_speculation_paths.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_cached_engine_cpu.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_sampling.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/property -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/report -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_cached_engine_gpu.py -q -m gpu; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_speculation_gpu.py -q -m gpu.

Evidence references: artifacts/environment.json; artifacts/calibration.json; artifacts/conformance.json; artifacts/controller_check.json; artifacts/traces/greedy_g4.jsonl; artifacts/profile/m4_profile.json.

## smoke / ALL DATASETS IN THIS COHORT / greedy / fixed_length_32

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 8 | 8 | 1699.885 | 1735.576 | 52.553 | 53.037 | 18.761 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 8 | 8 | 1733.320 | 1742.316 | 54.286 | 54.210 | 18.483 | 0.985x | [0.980, 0.991] | 0.0% |
| fixed_1 | 8 | 8 | 1300.126 | 1321.286 | 49.917 | 40.272 | 24.940 | 1.330x | [1.301, 1.360] | 0.0% |
| fixed_2 | 8 | 8 | 1086.819 | 1147.365 | 51.712 | 33.172 | 29.476 | 1.572x | [1.528, 1.618] | 0.0% |
| fixed_4 | 8 | 8 | 921.649 | 1034.726 | 51.175 | 27.726 | 33.964 | 1.815x | [1.727, 1.912] | 0.0% |
| fixed_8 | 8 | 8 | 922.773 | 1174.971 | 50.233 | 27.805 | 33.458 | 1.800x | [1.635, 1.976] | 0.0% |
| hf_dynamic | 8 | 8 | 841.597 | 1172.096 | 153.969 | 22.206 | 36.024 | 1.951x | [1.715, 2.177] | 0.0% |
| adaptive | 8 | 8 | 919.336 | 1200.318 | 51.256 | 27.583 | 32.760 | 1.763x | [1.601, 1.937] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8916.549 | 9200.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8938.236 | 9200.000 |
| fixed_1 | 75.0% | 0.906 | 1.816 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| fixed_2 | 75.0% | 0.849 | 2.462 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| fixed_4 | 62.5% | 0.757 | 3.413 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| fixed_8 | 75.0% | 0.592 | 4.571 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| hf_dynamic | 62.5% | N/A | N/A | N/A | N/A | 8953.280 | 9200.000 |
| adaptive | 75.0% | 0.587 | 4.197 | 0.000 | 0.000 | 8975.220 | 9200.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## smoke / gsm8k / greedy / fixed_length_32

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 4 | 4 | 1699.885 | 1734.197 | 53.711 | 53.037 | 18.721 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 4 | 4 | 1737.619 | 1742.980 | 55.415 | 54.264 | 18.417 | 0.984x | [0.976, 0.995] | 0.0% |
| fixed_1 | 4 | 4 | 1300.126 | 1323.166 | 53.616 | 40.272 | 24.743 | 1.322x | [1.287, 1.372] | 0.0% |
| fixed_2 | 4 | 4 | 1099.199 | 1161.110 | 57.102 | 33.549 | 28.757 | 1.537x | [1.474, 1.591] | 0.0% |
| fixed_4 | 4 | 4 | 941.810 | 1021.458 | 56.814 | 28.373 | 34.082 | 1.826x | [1.670, 1.984] | 0.0% |
| fixed_8 | 4 | 4 | 1014.116 | 1175.590 | 53.413 | 30.746 | 31.612 | 1.711x | [1.447, 2.022] | 0.0% |
| hf_dynamic | 4 | 4 | 976.124 | 1215.501 | 149.312 | 26.661 | 32.342 | 1.756x | [1.477, 2.077] | 0.0% |
| adaptive | 4 | 4 | 1047.544 | 1201.541 | 55.576 | 31.611 | 30.880 | 1.671x | [1.416, 1.970] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8916.549 | 9200.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8938.236 | 9200.000 |
| fixed_1 | 50.0% | 0.905 | 1.803 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| fixed_2 | 50.0% | 0.833 | 2.415 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| fixed_4 | 50.0% | 0.765 | 3.459 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| fixed_8 | 50.0% | 0.547 | 4.414 | 0.000 | 0.000 | 8975.220 | 9200.000 |
| hf_dynamic | 50.0% | N/A | N/A | N/A | N/A | 8953.280 | 9200.000 |
| adaptive | 50.0% | 0.542 | 4.000 | 0.000 | 0.000 | 8975.220 | 9200.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## smoke / mbpp / greedy / fixed_length_32

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 4 | 4 | 1697.203 | 1723.576 | 49.558 | 53.103 | 18.802 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 4 | 4 | 1724.797 | 1734.194 | 51.774 | 53.992 | 18.549 | 0.987x | [0.982, 0.993] | 0.0% |
| fixed_1 | 4 | 4 | 1273.005 | 1311.598 | 48.450 | 39.322 | 25.141 | 1.338x | [1.294, 1.382] | 0.0% |
| fixed_2 | 4 | 4 | 1072.833 | 1089.850 | 49.024 | 32.851 | 30.231 | 1.609x | [1.571, 1.676] | 0.0% |
| fixed_4 | 4 | 4 | 921.649 | 1022.727 | 49.342 | 27.726 | 33.846 | 1.803x | [1.683, 1.895] | 0.0% |
| fixed_8 | 4 | 4 | 922.773 | 958.264 | 47.591 | 27.805 | 35.534 | 1.895x | [1.787, 2.046] | 0.0% |
| hf_dynamic | 4 | 4 | 783.679 | 854.160 | 153.969 | 20.302 | 40.652 | 2.168x | [2.032, 2.312] | 0.0% |
| adaptive | 4 | 4 | 909.087 | 1002.332 | 48.716 | 27.331 | 34.884 | 1.860x | [1.726, 1.981] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8908.839 | 8974.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8914.642 | 8976.000 |
| fixed_1 | 100.0% | 0.906 | 1.829 | 0.000 | 0.000 | 8929.495 | 8976.000 |
| fixed_2 | 100.0% | 0.865 | 2.510 | 0.000 | 0.000 | 8929.495 | 8976.000 |
| fixed_4 | 75.0% | 0.750 | 3.368 | 0.000 | 0.000 | 8929.495 | 8976.000 |
| fixed_8 | 100.0% | 0.643 | 4.741 | 0.000 | 0.000 | 8929.685 | 8976.000 |
| hf_dynamic | 75.0% | N/A | N/A | N/A | N/A | 8941.221 | 8976.000 |
| adaptive | 100.0% | 0.639 | 4.414 | 0.000 | 0.000 | 8929.685 | 8974.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## Greedy conformance

Greedy speculation is exact in real arithmetic. In BF16 it is not: a target-only step computes its logits in a width-1 forward while a verification step computes them inside a wider one, and when the top two logits are within that noise the argmax flips. The `Greedy match` column above is the measured fraction of requests whose token ids were identical to the baseline's.

Every mismatch in this run was checked against a near-tie bound of 0.5 logits between the two tokens the engines actually chose. A mismatch with no recorded evidence, or one beyond that bound, is refused rather than reported. See docs/decisions/0004-bf16-greedy-conformance.md.

## Limits

These are sequential, batch-one warm-request results on the recorded configuration. They do not establish multi-user serving throughput, capability accuracy, or performance on other GPUs. Fixed-length conditions suppress EOS and must be read separately from natural stopping. Hashes check file integrity; they cannot prove honest acquisition. Consult SPEC.md and the acquisition code for measurement boundaries.
