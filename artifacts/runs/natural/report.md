# Switchback results

Generated from raw request measurements. Do not edit numerical values by hand.

Source commit: `c2d2224c6fe140e90eb3ceec9a6943c563cfa4c0`. Baseline: `hf_ar`.

```json
{
  "benchmark_source_sha256": "877d3f0c856a3dd393c1b3366a15321061835a0531915b2051653b72a42c56c0",
  "calibration_sha256": "69983f0c01b7bbe3e7141ba6d3c0a3ea595892e901ce5b81252b6f33b71338e7",
  "config_sha256": "c6af76d6a219c70f5760430fad3b1e7348851e80a51bf9048a08dc1f45a3847d",
  "files": {
    "adjudication.jsonl": "263d7ecee200881895eef8cb9bee09cd9dd7c2d234e4496a030b8cab6173ee26",
    "evidence.json": "03845163fdfdcef9de3eee3ebd13801bf45a59968213023251f1e1d2eb94fe6d",
    "requests.jsonl": "170e4b3d444f98b9e9def9cb44bd07ec28a1be74ae274d67bdb71a2d8e339363"
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
  "workload_sha256": "4e8ce7dddef416562c7bf1c5c2f492bbc0f1e96921a7fb00e4ec66a7f3bb37bb"
}
```

Complete logical requests: 3456. Required engines: 9.

Greedy token equality was checked for every paired request. Sampled outputs were not required to match.

Oracle, cache, and numerical gates: passed according to the linked validation evidence. Finite-model tests do not prove universal GPU equivalence.

Validation commands: /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_oracle.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_sampled_speculation.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_cache.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_speculation_paths.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_cached_engine_cpu.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_sampling.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/property -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/report -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_cached_engine_gpu.py -q -m gpu; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_speculation_gpu.py -q -m gpu.

Evidence references: artifacts/environment.json; artifacts/calibration.json; artifacts/conformance.json; artifacts/workloads/natural.json; artifacts/runs/primary/manifest.json.

## natural_real / ALL DATASETS IN THIS COHORT / greedy / natural_stop_256

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 384 | 128 | 6826.531 | 13615.048 | 49.932 | 51.747 | 19.339 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 384 | 128 | 6911.487 | 13596.753 | 49.116 | 51.742 | 19.366 | 1.001x | [0.999, 1.004] | 0.0% |
| fixed_1 | 384 | 128 | 5178.095 | 10181.592 | 48.913 | 38.528 | 25.878 | 1.341x | [1.318, 1.376] | 0.0% |
| fixed_2 | 384 | 128 | 4224.634 | 8338.186 | 49.211 | 31.534 | 31.561 | 1.618x | [1.588, 1.661] | 0.0% |
| fixed_4 | 384 | 128 | 3439.902 | 7390.943 | 49.055 | 26.805 | 36.863 | 1.879x | [1.832, 1.939] | 0.0% |
| fixed_8 | 384 | 128 | 3236.482 | 7233.519 | 48.768 | 25.837 | 37.809 | 1.935x | [1.868, 2.010] | 0.0% |
| hf_dynamic | 384 | 128 | 3028.816 | 7110.343 | 151.465 | 23.444 | 39.557 | 2.037x | [1.964, 2.121] | 0.0% |
| adaptive | 384 | 128 | 3305.936 | 7441.719 | 48.831 | 26.574 | 37.142 | 1.911x | [1.850, 1.980] | 0.0% |
| adaptive_no_bypass | 384 | 128 | 3337.847 | 7377.668 | 48.809 | 26.629 | 37.147 | 1.905x | [1.847, 1.972] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8954.545 | 9446.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8964.022 | 9446.000 |
| fixed_1 | 72.7% | 0.895 | 1.876 | 0.000 | 0.000 | 9050.732 | 9446.000 |
| fixed_2 | 72.7% | 0.852 | 2.654 | 0.000 | 0.000 | 9051.312 | 9446.000 |
| fixed_4 | 72.7% | 0.767 | 3.930 | 0.000 | 0.000 | 9052.475 | 9446.000 |
| fixed_8 | 74.2% | 0.630 | 5.702 | 0.000 | 0.000 | 9053.243 | 9446.000 |
| hf_dynamic | 71.1% | N/A | N/A | N/A | N/A | 9023.715 | 9446.000 |
| adaptive | 74.2% | 0.682 | 4.654 | 0.000 | 0.000 | 9052.250 | 9446.000 |
| adaptive_no_bypass | 73.7% | 0.686 | 4.613 | 0.000 | 0.000 | 9052.443 | 9446.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## natural_real / gsm8k / greedy / natural_stop_256

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 192 | 64 | 12268.470 | 13644.208 | 52.596 | 51.866 | 19.269 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 192 | 64 | 12237.519 | 13645.687 | 51.550 | 51.879 | 19.292 | 1.001x | [1.000, 1.003] | 0.0% |
| fixed_1 | 192 | 64 | 9170.260 | 10235.161 | 51.022 | 38.722 | 25.837 | 1.332x | [1.315, 1.348] | 0.0% |
| fixed_2 | 192 | 64 | 7431.012 | 8411.934 | 51.272 | 31.404 | 31.734 | 1.637x | [1.611, 1.659] | 0.0% |
| fixed_4 | 192 | 64 | 6249.115 | 7467.606 | 51.598 | 26.502 | 37.289 | 1.921x | [1.884, 1.956] | 0.0% |
| fixed_8 | 192 | 64 | 5837.717 | 7390.922 | 51.508 | 25.244 | 38.674 | 2.008x | [1.951, 2.063] | 0.0% |
| hf_dynamic | 192 | 64 | 5614.513 | 7277.470 | 155.019 | 24.010 | 39.998 | 2.071x | [2.004, 2.133] | 0.0% |
| adaptive | 192 | 64 | 6031.644 | 7502.610 | 51.417 | 26.452 | 37.770 | 1.958x | [1.914, 2.001] | 0.0% |
| adaptive_no_bypass | 192 | 64 | 6027.892 | 7476.489 | 51.446 | 26.526 | 37.700 | 1.940x | [1.899, 1.978] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8954.545 | 9446.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8964.022 | 9446.000 |
| fixed_1 | 60.9% | 0.902 | 1.889 | 0.000 | 0.000 | 9050.732 | 9446.000 |
| fixed_2 | 60.9% | 0.865 | 2.698 | 0.000 | 0.000 | 9051.312 | 9446.000 |
| fixed_4 | 59.4% | 0.782 | 4.036 | 0.000 | 0.000 | 9052.475 | 9446.000 |
| fixed_8 | 62.5% | 0.649 | 5.953 | 0.000 | 0.000 | 9053.243 | 9446.000 |
| hf_dynamic | 53.1% | N/A | N/A | N/A | N/A | 9023.715 | 9446.000 |
| adaptive | 62.5% | 0.708 | 4.731 | 0.000 | 0.000 | 9052.250 | 9446.000 |
| adaptive_no_bypass | 61.5% | 0.709 | 4.685 | 0.000 | 0.000 | 9052.443 | 9446.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## natural_real / mbpp / greedy / natural_stop_256

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 192 | 64 | 1883.011 | 8523.396 | 48.095 | 51.347 | 19.577 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 192 | 64 | 1899.179 | 8541.428 | 47.010 | 51.299 | 19.621 | 1.001x | [0.996, 1.005] | 0.0% |
| fixed_1 | 192 | 64 | 1359.156 | 6312.941 | 46.836 | 38.213 | 26.019 | 1.349x | [1.308, 1.420] | 0.0% |
| fixed_2 | 192 | 64 | 1131.772 | 5260.955 | 47.017 | 31.777 | 30.984 | 1.600x | [1.543, 1.687] | 0.0% |
| fixed_4 | 192 | 64 | 974.048 | 4763.092 | 46.860 | 27.358 | 35.474 | 1.838x | [1.751, 1.959] | 0.0% |
| fixed_8 | 192 | 64 | 1054.561 | 4743.850 | 46.806 | 26.449 | 35.125 | 1.864x | [1.746, 2.009] | 0.0% |
| hf_dynamic | 192 | 64 | 996.604 | 4508.666 | 149.114 | 21.923 | 38.121 | 2.003x | [1.872, 2.170] | 0.0% |
| adaptive | 192 | 64 | 1036.215 | 4789.888 | 47.248 | 26.785 | 35.140 | 1.866x | [1.754, 2.006] | 0.0% |
| adaptive_no_bypass | 192 | 64 | 1031.198 | 4811.990 | 46.821 | 26.899 | 35.368 | 1.870x | [1.759, 2.010] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8939.071 | 9446.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8939.495 | 9446.000 |
| fixed_1 | 84.4% | 0.871 | 1.831 | 0.000 | 0.000 | 8989.773 | 9446.000 |
| fixed_2 | 84.4% | 0.810 | 2.514 | 0.000 | 0.000 | 8990.095 | 9446.000 |
| fixed_4 | 85.9% | 0.718 | 3.605 | 0.000 | 0.000 | 8990.757 | 9446.000 |
| fixed_8 | 85.9% | 0.569 | 4.982 | 0.000 | 0.000 | 8991.960 | 9446.000 |
| hf_dynamic | 89.1% | N/A | N/A | N/A | N/A | 8995.537 | 9446.000 |
| adaptive | 85.9% | 0.607 | 4.408 | 0.000 | 0.000 | 8991.758 | 9446.000 |
| adaptive_no_bypass | 85.9% | 0.616 | 4.385 | 0.000 | 0.000 | 8991.758 | 9446.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## Greedy conformance

Greedy speculation is exact in real arithmetic. In BF16 it is not: a target-only step computes its logits in a width-1 forward while a verification step computes them inside a wider one, and when the top two logits are within that noise the argmax flips. The `Greedy match` column above is the measured fraction of requests whose token ids were identical to the baseline's.

Every mismatch in this run was checked against a near-tie bound of 0.5 logits between the two tokens the engines actually chose. A mismatch with no recorded evidence, or one beyond that bound that nothing adjudicated, is refused rather than reported. See docs/decisions/0004-bf16-greedy-conformance.md.

21 mismatch(es) exceeded that screen and were adjudicated by recomputing the position under every reproducible execution path, including a deterministic replay of the engine itself. 0 remained unexplained. The screen measures one path and the engines were in others, so a large screen gap is a question rather than a verdict. See docs/decisions/0006-adjudicating-greedy-divergences.md.

## Limits

These are sequential, batch-one warm-request results on the recorded configuration. They do not establish multi-user serving throughput, capability accuracy, or performance on other GPUs. Fixed-length conditions suppress EOS and must be read separately from natural stopping. Hashes check file integrity; they cannot prove honest acquisition. Consult SPEC.md and the acquisition code for measurement boundaries.
