# Switchback results

Generated from raw request measurements. Do not edit numerical values by hand.

Source commit: `8c7ef1a6de67bc427c7d54644adc099d82dced74`. Baseline: `hf_ar`.

```json
{
  "benchmark_source_sha256": "192411a83fdfb08daaa1ad46383f96d0dbeedd08b8f70e1288cb753054ed60f5",
  "calibration_sha256": "69983f0c01b7bbe3e7141ba6d3c0a3ea595892e901ce5b81252b6f33b71338e7",
  "config_sha256": "98412ef0665e2aca4ee8339f1050908418260f8002046dccbbec35ee795d9e5d",
  "files": {
    "adjudication.jsonl": "386b2c5c7d6a8e4afb44bf1ab098f0790401be451ca8f801996323dc7127fc42",
    "evidence.json": "69fd5aef20c7534310d70b137f1a6912f4d73aa105e7a7104b76d3af1d6e64a8",
    "requests.jsonl": "322bf6f43137254440a1003abde38334db3a4f32943afad7ba885b4b3c0c5db5"
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
  "workload_sha256": "402efe2d5f2168c9c36e52ddd6e87c4c4658f4adfcf8068f31e16cd100795d36"
}
```

Complete logical requests: 3072. Required engines: 8.

Greedy token equality was checked for every paired request. Sampled outputs were not required to match.

Oracle, cache, and numerical gates: passed according to the linked validation evidence. Finite-model tests do not prove universal GPU equivalence.

Validation commands: /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_oracle.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_sampled_speculation.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_cache.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_speculation_paths.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_cached_engine_cpu.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/unit/test_sampling.py -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/property -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/report -q; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_cached_engine_gpu.py -q -m gpu; /home/ethan/Desktop/switchback/Switchback_Build_Kit/.venv/bin/python -m pytest tests/integration/test_speculation_gpu.py -q -m gpu.

Evidence references: artifacts/environment.json; artifacts/calibration.json; artifacts/conformance.json; artifacts/controller_check.json; artifacts/workloads/primary.json; artifacts/traces/greedy_g4.jsonl.

## primary_real / ALL DATASETS IN THIS COHORT / greedy / fixed_length_256

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 384 | 128 | 13276.831 | 13362.735 | 50.323 | 51.867 | 19.346 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 384 | 128 | 13283.175 | 13388.853 | 49.102 | 51.890 | 19.332 | 0.999x | [0.999, 1.000] | 0.0% |
| fixed_1 | 384 | 128 | 9833.628 | 10251.399 | 49.280 | 38.335 | 25.973 | 1.344x | [1.339, 1.349] | 0.0% |
| fixed_2 | 384 | 128 | 7972.161 | 8762.126 | 49.256 | 31.051 | 31.869 | 1.652x | [1.638, 1.663] | 0.0% |
| fixed_4 | 384 | 128 | 6683.768 | 7977.882 | 49.080 | 25.980 | 37.612 | 1.953x | [1.923, 1.982] | 0.0% |
| fixed_8 | 384 | 128 | 6290.928 | 8745.390 | 49.210 | 24.412 | 39.358 | 2.066x | [2.004, 2.125] | 0.0% |
| hf_dynamic | 384 | 128 | 5743.743 | 8148.116 | 151.305 | 21.912 | 42.497 | 2.234x | [2.164, 2.300] | 0.0% |
| adaptive | 384 | 128 | 6561.544 | 8556.810 | 49.261 | 25.508 | 38.545 | 2.021x | [1.966, 2.070] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8956.520 | 9664.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8964.023 | 9664.000 |
| fixed_1 | 61.7% | 0.908 | 1.898 | 0.000 | 0.000 | 9050.732 | 9664.000 |
| fixed_2 | 61.7% | 0.868 | 2.707 | 0.000 | 0.000 | 9051.312 | 9664.000 |
| fixed_4 | 64.1% | 0.787 | 4.069 | 0.000 | 0.000 | 9052.475 | 9664.000 |
| fixed_8 | 63.3% | 0.661 | 6.084 | 0.000 | 0.000 | 9053.243 | 9664.000 |
| hf_dynamic | 62.5% | N/A | N/A | N/A | N/A | 9023.715 | 9664.000 |
| adaptive | 63.5% | 0.705 | 5.023 | 0.000 | 0.000 | 9052.250 | 9664.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## primary_real / gsm8k / greedy / fixed_length_256

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 192 | 64 | 13285.094 | 13387.102 | 52.280 | 51.891 | 19.390 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 192 | 64 | 13300.511 | 13394.537 | 51.629 | 51.958 | 19.375 | 0.999x | [0.998, 1.000] | 0.0% |
| fixed_1 | 192 | 64 | 9946.493 | 10227.956 | 51.238 | 38.782 | 25.790 | 1.332x | [1.327, 1.338] | 0.0% |
| fixed_2 | 192 | 64 | 8132.151 | 8659.646 | 51.484 | 31.643 | 31.486 | 1.629x | [1.615, 1.643] | 0.0% |
| fixed_4 | 192 | 64 | 6900.852 | 7850.050 | 51.652 | 26.813 | 36.766 | 1.902x | [1.875, 1.931] | 0.0% |
| fixed_8 | 192 | 64 | 6870.101 | 8722.946 | 51.192 | 26.706 | 37.021 | 1.927x | [1.871, 1.984] | 0.0% |
| hf_dynamic | 192 | 64 | 6534.438 | 8188.456 | 154.050 | 25.023 | 39.042 | 2.034x | [1.977, 2.096] | 0.0% |
| adaptive | 192 | 64 | 6896.400 | 8252.043 | 51.384 | 26.792 | 36.627 | 1.905x | [1.861, 1.952] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8956.520 | 9664.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8964.023 | 9664.000 |
| fixed_1 | 50.0% | 0.890 | 1.880 | 0.000 | 0.000 | 9050.732 | 9664.000 |
| fixed_2 | 50.0% | 0.846 | 2.667 | 0.000 | 0.000 | 9051.312 | 9664.000 |
| fixed_4 | 51.6% | 0.759 | 3.961 | 0.000 | 0.000 | 9052.475 | 9664.000 |
| fixed_8 | 51.6% | 0.611 | 5.703 | 0.000 | 0.000 | 9053.243 | 9664.000 |
| hf_dynamic | 48.4% | N/A | N/A | N/A | N/A | 9023.715 | 9664.000 |
| adaptive | 51.6% | 0.662 | 4.717 | 0.000 | 0.000 | 9052.250 | 9664.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## primary_real / mbpp / greedy / fixed_length_256

| Engine | Requests | Prompts | p50 ms | p95 ms | TTFT p50 ms | TPOT p50 ms | Tokens/s | Speedup | 95% CI | Slowdown >5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| hf_ar | 192 | 64 | 13269.608 | 13337.993 | 48.773 | 51.843 | 19.302 | 1.000x | [1.000, 1.000] | 0.0% |
| native_ar | 192 | 64 | 13264.671 | 13357.792 | 47.860 | 51.834 | 19.289 | 1.000x | [0.999, 1.001] | 0.0% |
| fixed_1 | 192 | 64 | 9751.555 | 10284.535 | 47.732 | 38.047 | 26.158 | 1.356x | [1.349, 1.362] | 0.0% |
| fixed_2 | 192 | 64 | 7835.223 | 9003.275 | 47.749 | 30.503 | 32.261 | 1.675x | [1.655, 1.691] | 0.0% |
| fixed_4 | 192 | 64 | 6470.028 | 8424.498 | 47.610 | 25.165 | 38.499 | 2.006x | [1.957, 2.048] | 0.0% |
| fixed_8 | 192 | 64 | 5730.060 | 9548.566 | 47.630 | 22.235 | 42.010 | 2.215x | [2.111, 2.304] | 0.0% |
| hf_dynamic | 192 | 64 | 5121.638 | 7941.010 | 149.415 | 19.490 | 46.622 | 2.453x | [2.350, 2.542] | 0.0% |
| adaptive | 192 | 64 | 6107.979 | 8744.133 | 47.797 | 23.740 | 40.676 | 2.144x | [2.054, 2.223] | 0.0% |

| Engine | Greedy match | Accepted/proposed | Output tokens/target call | Bypass decisions | Requests with bypass | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| hf_ar | 100.0% | N/A | N/A | N/A | N/A | 8944.564 | 9664.000 |
| native_ar | 100.0% | N/A | 1.000 | N/A | 0.000 | 8943.011 | 9664.000 |
| fixed_1 | 73.4% | 0.927 | 1.916 | 0.000 | 0.000 | 9002.990 | 9664.000 |
| fixed_2 | 73.4% | 0.890 | 2.748 | 0.000 | 0.000 | 9003.312 | 9664.000 |
| fixed_4 | 76.6% | 0.816 | 4.183 | 0.000 | 0.000 | 9004.506 | 9664.000 |
| fixed_8 | 75.0% | 0.718 | 6.520 | 0.000 | 0.000 | 9005.145 | 9664.000 |
| hf_dynamic | 76.6% | N/A | N/A | N/A | N/A | 9002.038 | 9664.000 |
| adaptive | 75.5% | 0.753 | 5.371 | 0.000 | 0.000 | 9004.849 | 9664.000 |

Speedups use per-prompt median latency, then a geometric mean. Intervals resample prompt identities 2,000 times; repeats remain inside each prompt. p95 is descriptive, especially for small strata. N/A means unavailable or undefined.

Adaptive latency improvement is supported by this interval.

## Greedy conformance

Greedy speculation is exact in real arithmetic. In BF16 it is not: a target-only step computes its logits in a width-1 forward while a verification step computes them inside a wider one, and when the top two logits are within that noise the argmax flips. The `Greedy match` column above is the measured fraction of requests whose token ids were identical to the baseline's.

Every mismatch in this run was checked against a near-tie bound of 0.5 logits between the two tokens the engines actually chose. A mismatch with no recorded evidence, or one beyond that bound that nothing adjudicated, is refused rather than reported. See docs/decisions/0004-bf16-greedy-conformance.md.

18 mismatch(es) exceeded that screen and were adjudicated by recomputing the position under every reproducible execution path, including a deterministic replay of the engine itself. 0 remained unexplained. The screen measures one path and the engines were in others, so a large screen gap is a question rather than a verdict. See docs/decisions/0006-adjudicating-greedy-divergences.md.

## Limits

These are sequential, batch-one warm-request results on the recorded configuration. They do not establish multi-user serving throughput, capability accuracy, or performance on other GPUs. Fixed-length conditions suppress EOS and must be read separately from natural stopping. Hashes check file integrity; they cannot prove honest acquisition. Consult SPEC.md and the acquisition code for measurement boundaries.
