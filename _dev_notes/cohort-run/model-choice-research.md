# Model choice: what people actually use (research, 2026-09-27)

Owner's question: are the modern-model picks the ones people use and talk
about, or only high download counts?  A research agent read public sources;
I verified every repo id on the Hub and every architecture against the
pinned vLLM registry.  Limits: reddit.com was read through search snippets
only; OpenRouter per-model token counts need a key (public top-10 lists and
provider counts used); LMArena, X and meetups were not checked.

## Sources that decided

- OpenRouter rankings (openrouter.ai/rankings, data through 2026-09-26):
  week #1 DeepSeek V4.1 Flash (18.6T tokens), #2 GLM 5.3 Flash (17.8T), #6
  DeepSeek V4 Flash 0731 (8.1T); month #1 GLM 5.3 Flash (58.1T), #10 GLM 5.3.
- OpenRouter provider counts (`/api/v1/models/{id}/endpoints`): GLM-5.3 40,
  GLM-5.3-Flash 33, DeepSeek-V4-Flash-0731 31, V4.1-Flash 27, gpt-oss-120b
  24, V4-Pro-0813 22, Kimi K3 18, Qwen3.8-27B 16, gemma-4-31b 15,
  DeepSeek-V3.2 14, MiniMax-M3 13, gpt-oss-20b 12, Qwen3.6-35B-A3B 10,
  MiniMax-M2.7 9, Nemotron-3-Nano 4, GLM-4.7-Flash 3, Kimi-K2-0905 1.
- Artificial Analysis Intelligence Index v4.3
  (artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3):
  GLM-5.3 and Kimi K3 lead open weights.
- Hacker News: GLM-5.3-Flash (item 49450353), Kimi K3 / GLM-5.3 (49539315),
  DeepSeek V4.1 Flash (49725800), Qwen3.8-Flash-Next (49448210, "churns
  5-10 minutes"), Meta Muse-Glimmer (49241679).
- r/LocalLLaMA (snippets): gpt-oss one year on, "one of the best local
  models"; Qwen3.8-27B "game changer" and "benchmaxxed"; Gemma 4-31B best
  general/vision at its size, weak at code.

## Decisions

| Pick | Evidence | Decision |
|---|---|---|
| gpt-oss-20b, gpt-oss-120b | 12 / 24 providers, still praised | keep |
| gemma-4-31B-it | 9.5M downloads, 15 providers | keep |
| GLM-4.7-Flash | 3 providers | replaced by meta-models/Muse-Glimmer-30B (Aug 9, Apache-2.0); measured once because it was already staged |
| Qwen3.8-27B, Qwen3.6-35B-A3B-FP8 | most-discussed local model; newest small Qwen MoE | keep |
| Nemotron-3-Nano | superseded | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (same NemotronH architecture) |
| MiniMax-M2.7 | superseded | MiniMax-M3 (moves to D: needs 8x H200) |
| DeepSeek-V4-Flash-0731 | superseded by V4.1-Flash (Sep 10) | fallback in C if the engine stays at v0.29.0 |
| GLM-5.3 | open-weight leader, 40 providers | keep |
| DeepSeek-V3.2 | superseded | DeepSeek-V4-Pro-0813 |
| Kimi-K2-Instruct-0905 | 1 provider | dropped; Kimi-K3 (2.8T) needs 8x GB300 |
| (new) GLM-5.3-Flash, DeepSeek-V4.1-Flash | #1 month, #1 week on OpenRouter | C — need vLLM v0.30.0 (Glm5Next, DeepseekV41 absent from v0.29.0's registry, present in v0.30.0's) |
| (new) Qwen3.8-Flash-Next | Qwen's newest large model | C |
| (skipped) MiMo-V2.6-Flash-RL | 6 days old, "benchmaxxed" thread | not added |
