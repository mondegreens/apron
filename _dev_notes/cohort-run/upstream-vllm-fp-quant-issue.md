# Draft upstream issue — vllm-project/vllm (for the owner to file)

Not filed. Posting is the owner's decision (AGENTS.md: no autonomous issues).

---

**Title:** [Bug]: FP-Quant checkpoints fail to load since #44122 removed `backward_hadamard_matrix`

### Your current environment

vLLM v0.29.0 (release wheel, CUDA 13.0), NVIDIA B200 (SM100), RunPod Secure.
Also present on `main` as of 2026-09-27 (`vllm/model_executor/layers/quantization/fp_quant.py`).

### 🐛 Describe the bug

Every FP-Quant checkpoint on the Hub fails at weight loading:

```
ValueError: There is no module or parameter named
'layers.0.mlp.down_proj.backward_hadamard_matrix' in Qwen3Model.
```

Reproduce (B200, since `fp_quant` needs capability 100):

```
vllm serve ISTA-DASLab/Qwen3-0.6B-FPQuant-RTN-MXFP4 --allow-deprecated-quantization
```

Cause: #44122 ("[Refactor] Remove dead code fp quant", 2026-06-03, first in
v0.23.0) removed the `backward_hadamard_matrix` parameter from
`FPQuantLinearMethod.create_weights`. The tensor is unused at inference, but
the checkpoints store it, so the loader finds a checkpoint tensor with no
parameter to load it into. v0.11.1 (e.g. e33ee23) still registered it.

Scope: we read the safetensors headers of all 32 repos returned by a Hub
search for "FPQuant" (ISTA-DASLab Qwen3 0.6B–32B and Llama 3.x; RTN, GPTQ,
QAT; MXFP4 and NVFP4) on 2026-09-27. Every one stores
`backward_hadamard_matrix`, so none loads on v0.23.0 or later.

Possible fixes: register the parameter again as an ignored buffer, or skip
`*.backward_hadamard_matrix` in weight loading for `fp_quant`.

### Evidence

- Failed boot log and record: `_dev_notes/cohort-run/records/verification-reports/1220e2bb9d9e3077ec36a3b1996d3fa982eb9268665a8054c69a161934461b3fd41d.json`
  (Apron repository), 2026-09-27 02:43 UTC, B200.
- `git log -S backward_hadamard_matrix -- vllm/model_executor/layers/quantization/fp_quant.py`
  → 96ad65b7fe (added, #24440), 2b91012650 (removed, #44122).
