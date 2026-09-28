---
draft: true
date: 2026-09-28
categories:
  - Findings
---

# I know you run models. Will it boot?

![Poster parody: an engineer in a hoodie with a red OOM light on the hood points at you. Text: I know you run models. On rented GPUs. At 2 a.m. Apron needs you.](img/i-know-you-run-models.svg)

<!-- Draft for the owner. Every number below comes from a stored record in
_dev_notes/cohort-run/records (digests in "Where the numbers come from") or
from the pinned vLLM source (file:line). Numbers marked [pending] wait for the
proof boots of 2026-09-28. -->

It is 2 a.m. You rented an H100 because a new model came out this week and
everyone on your timeline is running it. You downloaded 60 GB of weights on
the clock. You ran `vllm serve` with the flags from the model card. It printed
a wall of `torch.compile` output, then `Engine core initialization failed`,
then nothing useful. You asked a chatbot. It suggested `--enforce-eager`.
Different error. Forty minutes and a few dollars later you still do not know
whether this model fits on this card, and if it does, with what settings.

We know, because this week we did it to ourselves, on purpose, seven times.

This post is about Apron, an open-source tool that answers "will it fit, will
it boot, will it answer" before you rent the GPU, then checks its own answer on
a real one and keeps both. It is also about what we got wrong while pointing it
at the models people actually run this month, and how we found out.

## Highlights

- **Before renting anything**, Apron predicted the memory of current open
  models (gpt-oss-120b, GLM-4.7-Flash, Gemma 4 31B, Qwen3.6-35B-A3B,
  Muse-Glimmer-30B, Nemotron-3.5, Qwen3.8-27B) from their configs, safetensors
  headers and the pinned vLLM release's own source. Weights landed within
  about 1% of what vLLM measured on the GPU.
- **It was wrong about the startup memory peak** of four new models (off by up
  to 3×). We did not tune a constant. We recorded the GPU allocator's history
  during vLLM's profiling run, found every live tensor at the peak, and fixed the
  rule for all models. Today 25 of 25 measured points agree within 6%.
- **Two models failed to boot** with vLLM's defaults on an H100, both hybrid
  (Mamba / linear attention) models. Both failures are predictable without a
  GPU, and the planner now chooses a setting that avoids them. [pending: proof
  boots]
- **One model booted and "failed" every task while answering all of them
  correctly.** It needed a reasoning parser. Booting is not answering.
- **vLLM released v0.30 on September 22.** Apron planned on it the same week:
  one runner image per vLLM version, facts and diagnosis rules generated from
  that version's source. Two of the models we want next do not load on v0.29 at
  all.
- **All of it cost $16.61** in GPU and CPU time, including every failed boot
  and every diagnostic run. Weights were staged on network storage by a CPU pod,
  so no GPU minute was spent downloading.

## The problem nobody owns

Deploying an open model on your own GPU is a three-way bet: the checkpoint, the
engine version and the hardware. Each of the three moves.

- **The engine.** vLLM ships a release every two weeks; v0.24 came out on
  June 29 and v0.30 on September 22. Its maintainers are candid about how fast
  things change: they dropped a hand-maintained model capability table because
  it "drifts", and they now run nightly accuracy and performance checks on the
  datacenter recipes they care most about.
- **The models.** New attention layouts arrive about monthly: Mamba and gated
  delta-net hybrids, DeepSeek V4's compressed KV cache, GLM-5's linear plus
  sparse attention. Each one pages memory differently. "Day-0 support" is the
  most common kind of post on the vLLM blog.
- **The hardware.** Consumer, professional and datacenter cards, rented by the
  hour, with memory that is never quite the number on the box. An H100 80GB
  gives PyTorch 85,017,493,504 bytes, not 85,899,345,920.

The tools around this are good and getting better. vLLM refuses many bad
configurations before loading and tells you what to change. NVIDIA's
aiconfigurator and llm-d's planner estimate memory and performance for common
attention types; aiconfigurator now lists hybrid models too. Hugging Face shows
whether GGUF and MLX files fit your hardware. None of them is the thing that
failed us at 2 a.m., which is a joined answer: this exact checkpoint, on this
exact vLLM version, on this exact GPU, predicted before paying and then checked.

That need is on record. vLLM issue
[#29325](https://github.com/vllm-project/vllm/issues/29325) asked for a
preflight "dry run". In the discussion, a contributor from llm-d proposed
building such a planner outside vLLM and tracking the vLLM version, "since
version differences constantly affect performance and deployment". The issue
was closed as not planned in July. The planners say it themselves: aiconfigurator's README still
notes that memory estimation "needs to be studied more", and one of its issues
([#1396](https://github.com/ai-dynamo/aiconfigurator/issues/1396)) documents a
sweep that skipped the KV budget check, over-predicted throughput 4.6× and
recommended a configuration that did not deploy. That is not a knock on them.
It is what every estimate looks like until it is checked against a boot.

## What Apron does

Apron is a small loop with a strict rule about what it is allowed to call true.

1. **Plan, GPU-free.** Read the checkpoint's config and safetensors headers from
   the Hub (headers only, no weights), and the facts of the pinned vLLM version:
   which architectures it registers, which tensors it loads and which it skips,
   how it pages KV cache for each layer type, its default batch sizes. Compute
   weights, KV cache per sequence, the startup activation peak and the CUDA
   graph cost per GPU.
2. **Say "unknown" out loud.** If a model uses an attention layout the
   calculator does not model, the answer is "unknown", not an estimate borrowed
   from a different layout.
3. **Boot and measure.** Rent the GPU, attach pre-staged weights, boot the exact
   plan, read vLLM's own memory accounting, run a small task suite and a serving
   benchmark, tear the pod down.
4. **Keep prediction beside measurement.** Every record is content-addressed
   and stores what was predicted next to what was measured, with the engine
   image digest and the hardware it ran on.
5. **Prove fixes.** When a boot fails, the error is fingerprinted and matched to
   a correction rule. A rule starts as a *hypothesis*; it becomes a proven rule
   only when a second boot with the correction succeeds.

Every number Apron shows is labelled by how it is known: *derived* (computed
from inputs), *proven constraint* (the pinned engine's source enforces it),
*predicted* (the calculator modelled it) or *measured* (the exact execution
observed it). A prediction is never presented as a vLLM test.

## What we ran this week

We picked models by what people actually use and talk about in September 2026
(OpenRouter rankings, provider counts, Hacker News and r/LocalLLaMA threads),
not by download counts. All on one H100 80GB per model, RunPod Secure Cloud,
US datacenters.

| Model | Layout | vLLM | Boot | Tasks | p99 TTFT (SLO 2 s) | Weights, predicted → measured |
|---|---|---|---|---|---|---|
| gpt-oss-120b | MoE, sliding + full attention | v0.29 | ok | 3/3 | 2.29 s, **miss** | 60.77 → 61.43 GiB |
| GLM-4.7-Flash | MoE, MLA | v0.29 | ok | 3/3 | 1.32 s | 55.77 → 55.87 GiB |
| Gemma 4 31B | dense, sliding + global, vision | v0.29 | ok | 3/3 | 0.20 s | 58.46 → 58.99 GiB |
| Qwen3.6-35B-A3B FP8 | MoE, gated delta-net hybrid, vision | v0.30 | ok | 3/3 | 0.14 s | 34.09 → 34.23 GiB |
| Muse-Glimmer-30B | dense, sliding + full, vision | v0.30 | ok | 0/3 → [pending] | 0.19 s | 55.46 → 55.83 GiB |
| Nemotron-3.5-Lightning | MoE, Mamba hybrid | v0.30 | **failed** → [pending] | — | — | — |
| Qwen3.8-27B | dense, gated delta-net hybrid, vision | v0.29 | **failed** → [pending] | — | — | — |

The tasks are deliberately tiny (three questions with exact answers). They are
there to prove the endpoint answers correctly, not to rank models. The full
predicted-versus-measured table for every boot, old and new models, is in the
repository and updates with every run.

## The bug we are proudest of finding: the startup peak

vLLM decides how much KV cache it can allocate by running one dummy batch at
startup and measuring the peak memory PyTorch allocated during it. Whatever
that peak is comes straight out of your KV budget. Apron predicts it, and until
this week its rule was simple and, on fifteen measured points from an earlier
cohort of smaller models on five GPU types, right to within 0.02 GiB: during the first, cold compile, the compiler
benchmarks the embedding kernel and briefly allocates a copy of the embedding
table plus two activation-sized outputs.

The new models broke it in both directions:

| Model | Predicted | Measured |
|---|---|---|
| GLM-4.7-Flash | 0.65 GiB | 2.04 GiB |
| Gemma 4 31B | 2.79 GiB | 1.49 GiB |
| Qwen3.6-35B-A3B | 1.01 GiB | 1.92 GiB |
| Muse-Glimmer-30B | 2.71 GiB | 1.83 GiB |

Reading the source got us part of the way. For Gemma 4 it showed that a
multimodal model's dummy run feeds pre-built embeddings, so the embedding table
never enters the compiled graph. For GLM it found MLA's prefill scratch buffer
and the fused-MoE workspace. Both explanations still left about 0.4 GiB
unaccounted for, and we did not want to paper over 0.4 GiB with a constant.

So we measured it. We wrote a small `sitecustomize` hook that loads inside
`vllm serve`, wraps `profile_run`, turns on PyTorch's allocator history and
replays it to find the exact moment of the peak and every block alive at that
moment, with its allocation stack. One H100, the exact recorded serve commands,
a cold compile cache. It reproduced vLLM's logged peaks to the byte and named
every tensor:

- **GLM-4.7-Flash**: MLA's prefill dummy (1,174,405,120 bytes, allocated at
  `mla_attention.py:783`), the fused-MoE workspace (335,544,320), and five
  query-width plus seven hidden-width buffers kept alive by the compiled graph
  around attention.
- **Gemma 4**: no compile transient at all. The peak is the forward pass: the
  MLP's gate/up output, its activation, the attention output of a global layer
  and two hidden states, plus a little left over from the video encoder.
- **The control.** The same Gemma 4 with `--language-model-only`: 2,995,739,688
  bytes measured against 2,994,733,056 predicted by the old rule. The old rule
  was right; it applied to text-only models, and Gemma 4 is not one.
- **Qwen3.6 and Muse-Glimmer (on vLLM v0.30)**: the peak is not in the language
  model. It is the vision tower, encoding the dummy batch of images vLLM sizes
  from the encoder budget: 65,536 patches for Qwen3.6, two 4,096-token images
  for Muse-Glimmer, whose rotary embeddings are computed in FP32.

The rule is now the largest of four moments, each made of named tensors whose
shapes come from the config and the engine's defaults: the compile transient
(text-only models), the forward pass, the sampler, and the vision encoder's
profiling batch. Where a count of simultaneously live buffers comes from the
probe rather than from source, the code says so. All 25 measured points pass,
the earlier ones included. Where a model's vision tower has not been
traced yet, the plan says "encoder peak not modelled" instead of adding zero.

The two diagnostic boots cost $1.40.

## New layouts, counted from the engine's own code

The next models we want to run, GLM-5.3-Flash and DeepSeek-V4.1-Flash, are the
two most-used open models on OpenRouter this month. Neither loads on vLLM v0.29,
and both page KV cache in ways that did not exist a quarter ago.

Before this week, Apron would have planned DeepSeek V4 as ordinary attention
and printed a confident number. That is the failure we care most about, so the
first fix was a guard: any config field describing an attention layout the
calculator does not read now makes the answer "unknown".

Then we taught it the layouts, from the v0.30 source:

- **DeepSeek V4 / V4.1**: the KV cache dtype is forced to an FP8 record of 584
  bytes per state regardless of what you ask for; the cache is replicated on
  every tensor-parallel rank rather than sharded; compressed, sliding-window and
  indexer caches are grouped by a packed layout rather than the usual one. We ran
  vLLM's own grouping functions verbatim on the traced specs and compared them
  with Apron's calculator: 40 of 40 cases matched.
- **Qwen3.8-Flash-Next and GLM-5.3-Flash**: linear-attention state, compressed
  index keys and ring buffers, again checked against vLLM's own code. Across
  the four models, 136 cases, 0 mismatches.
- **Host memory**: DeepSeek V4.1 keeps 189 GiB of "Engram" tables and
  Qwen3.8-Flash-Next 95 GiB of n-gram tables in pinned host RAM by default. A
  GPU-only fit check would pass them on a pod that cannot hold them. Apron now
  reports the host-RAM requirement with the plan.

None of these are measured yet; that is the next run. Until then they are
predictions, labelled as such.

## When it broke anyway

vLLM's error messages are good; most of them already tell you what to change.
What they cannot tell you is whether you could have known before paying, and
whether the change works. That is the part Apron records.

| Model | What happened | Could it be known before renting? | Fix | Proven by a second boot |
|---|---|---|---|---|
| Nemotron-3.5-Lightning (v0.30) | `max_num_seqs (1024) exceeds available Mamba cache blocks (733)` | Yes: every decoding sequence needs one Mamba state block; the blocks follow from the KV budget | planner sets `max_num_seqs` from the predicted block count (556) | [pending] |
| Qwen3.8-27B (v0.29) | an opaque `aten::new_empty ... API call failed` inside FlashAttention | Yes: before measuring CUDA graphs, vLLM allocates a KV cache for 512 sequences; for this hybrid that is 24.5 GiB, more than was left | same planner rule | [pending] |
| Muse-Glimmer-30B (v0.30) | booted, answered, scored 0/3 | Yes: its chat template has a reasoning channel and vLLM registers a parser for it | plan adds `--reasoning-parser muse_glimmer` when the template declares a reasoning channel | [pending] |

The second row is the interesting one. The real error was an out-of-memory in
the caching allocator, but the FlashAttention extension is built against an
older PyTorch stable ABI whose error macro drops the message. We found the
cause by reading the allocator and the extension source, and confirmed the
mechanism with a control: Qwen3.6, which takes the same kernel path with 35 GB
of headroom, booted fine.

Every fix starts as a hypothesis rule in the repository, with the record that
motivated it. It is promoted only by the corrected boot.

We also broke our own harness twice, and both are now tests: a network volume
sized for one group of models filled up when the next group was added, and an
SSH read that waited for a command to finish before reading its output
deadlocked on a 12.7 MB vLLM log (the SSH window is 2 MiB). The second one cost
us a healthy boot's measurement and $1.08.

## vLLM moved; so did Apron

vLLM v0.30.0 came out on September 22. By the next day Apron had a runner image
for it, built in CI from that tag's own pinned requirements and pinned by
digest; engine facts regenerated from the v0.30 source (ten new architectures
registered); and the nineteen families of diagnosis rules re-cited against the
new source, with 13 messages that no longer exist and 72 new ones. The planner
picks, per model, the newest pinned vLLM version that registers its
architecture; each plan is diagnosed by the rules of the version it runs on.

Doing this is one command for the next release. It matters because answers
expire: a flag, a default or a memory layout that was true for v0.29 may not be
true for v0.30, and a record says which one it was measured on.

## What it cost

| | |
|---|---|
| GPU and CPU time, every boot, failed or not | $16.61 |
| of which diagnostic probe boots | $1.40 |
| staging ~190 GB of weights with a CPU pod | $0.024 |
| checking staged weights on the GPU pod | 1–2 s per model |
| engine start (cold compile) | 104–507 s per model |

The budget for the whole phase is $100. Weights are staged once onto a network
volume in the same datacenter as the GPU; the GPU pod only verifies them.

## What this does not show

- **Seven models, one boot each, one GPU type.** An earlier cohort of smaller
  models adds five GPU types. That is evidence, not a statistic.
- **Some parts of the startup-peak rule come from observation.** The shapes are
  from source; how many buffers the compiled graph keeps alive at once was read
  from the probe for the models we probed. A new model family can break it, and
  the next probe is cheap.
- **The KV budget is predicted within ±0.5 GiB on 27 of 28 records.** The
  outlier, GLM's MLA CUDA graphs, is not modelled yet; the planner's safety
  buffer (2.19 GiB) is the largest error measured across all records, not a
  tuned margin.
- **The task suite is a correctness check, not a benchmark.** It proves the
  endpoint answers; it says nothing about which model is better.
- **No multi-GPU, no speculative decoding, no prefill/decode split yet.** Those
  are where the field is; they are next, not claimed.

## Where Apron fits

Apron is not a serving engine, a benchmark or a leaderboard, and it does not
replace any of them.

- **vLLM** is the engine; Apron reads its source per release and boots it.
- **aiconfigurator and llm-d's planner** produce plans; Apron can take their
  plans as candidates and keep their predictions beside a measurement.
- **vllm-project/recipes** hold verified configurations; Apron renders the ones
  it proves in the same shape.
- **InferenceMAX / InferenceX** measure performance at scale on datacenter
  racks; Apron's records are about whether a given plan fits, boots and answers
  on the GPU you can rent.

Records are published under an open data license, verify without any server,
and nothing leaves your machine unless you run `submit`.

## Apron needs YOU

If you run models, you have a failure Apron has not seen yet. Run it on your
model and your GPU and tell us where it was wrong. If you maintain a planner, an
engine adapter or a recipe collection, the records are for you to check against.
The repository, the records and every rule are at
[github.com/mondegreens/apron](https://github.com/mondegreens/apron).

## Where the numbers come from

<!-- TODO before publishing: the record digests for every row above, generated
by scripts/cohort_findings.py; the probe files in
_dev_notes/cohort-run/peak-probe/; the traces in _dev_notes/cohort-run/*-trace.md. -->
