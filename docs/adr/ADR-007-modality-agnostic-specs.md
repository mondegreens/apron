# ADR-007 — Modality-open contracts use capability signatures and execution mechanisms

**Status:** Accepted, 2026-09-06; quantization-candidate compatibility amendment accepted 2026-09-07; workload separation amended by ADR-011 on 2026-09-08; capability-signature and mechanism model replaced the original family discriminator on 2026-09-08
**Affects:** Target Architecture §1, §3, §4; Phase Plan Phase 0, Phase 4; Framework extension points and invariants; read together with ADR-011
**Evidence:** vLLM source at pinned commit `a1541f5742a29864a80087af313ad460066a1524` (2026-09-06), especially `vllm/tasks.py`, `docs/models/supported_models.md`, `docs/features/multimodal_inputs.md`, `docs/models/pooling_models/`, `docs/serving/online_serving/speech_to_text.md`, and `vllm/multimodal/encoder_budget.py`; current Hugging Face Diffusers pipeline documentation inspected 2026-09-08

## Context

The first decision correctly refused to freeze the schemas around one text-LLM workload, but its replacement vocabulary was still wrong. `text-autoregressive`, `diffusion`, and `encoder` are not mutually exclusive values of one axis. A vision-language model may contain a vision encoder, projector and autoregressive decoder; a pooling model may use an encoder or a causal decoder; discrete diffusion can produce text while latent diffusion can produce image, video or audio. Selecting a memory formula from a broad model family or semantic modality would therefore reproduce the same false abstraction in a larger schema.

The reviewed vLLM revision proves that this is an implementation concern, not speculative future scope. Its runner types include generation and pooling; its supported tasks include text generation, transcription, realtime transcription, embedding, classification and token-wise pooling; and its multimodal connector accepts image, video and audio. Model support is expressed as combinations such as `T + I` and mutually exclusive alternatives such as `T / I`, not independent booleans. Some checkpoint capabilities are also absent from a particular engine implementation: vLLM documents, for example, that Moondream's native `detect` and `point` skills are not exposed. At the same time, vLLM has no image-generation, video-generation or text-to-speech serving endpoint. Those require a different engine even when the artifact contains a component also called diffusion.

The GLM-5.3 production briefing made the product consequence concrete. Vision capability can change the winning model and deployment even when text benchmark strength is otherwise higher. A task-to-inference-solution product that collapses this distinction cannot answer the decision that motivated ADR-011.

## Decision

1. **Semantic modality is an open vocabulary, not an architecture discriminator.** The built-in interoperable modalities are `text`, `image`, `audio`, and `video`. Namespaced extensions may add modalities without a public-schema fork. MIME type, encoding, transport and storage reference are separate fields. A PDF, web page or GUI recording is a composite application input that may contain text, images, layout, audio or video; it is not silently reduced to one model modality. JSON, tool calls, embeddings, labels, scores, timestamps and bounding boxes are representations or result shapes, not new semantic modalities.

2. **The unit of capability is a `CapabilitySignature`.** It records an operation; required and optional input ports; permitted modality combinations; per-port cardinality, shape and streaming constraints; output ports and result representations; and any application-visible semantics. Examples include `{text,image} -> generated_text`, `{audio_stream} -> text_delta + timestamps`, `{text} -> embedding_vector`, `{text,image} -> score`, and `{text} -> generated_image`. Independent `input_modalities` and `output_modalities` lists cannot assert compatibility because they lose `T + I` versus `T / I` semantics.

3. **Capability is scoped at every boundary.** `ModelSpec` may preserve an `artifact_declared` signature from publisher metadata. An engine adapter separately produces `engine_resolved` signatures for an exact engine image/version and implementation. A configured endpoint records the subset it actually exposes as `endpoint_exposed`. Evidence then promotes only the exact signature and fingerprint through `boot_verified`, `request_verified`, and `task_qualified`. No level inherits a broader capability merely because the checkpoint, another engine, or another endpoint has it.

4. **`ModelSpec` has a common identity core plus a typed component graph.** The common core retains repository, immutable revision, license/gating, files, exact component bytes and dtype, remote-code requirement, lineage and attributed publisher claims. Each component declares its role and an extensible typed execution mechanism, such as `autoregressive_decode`, `discrete_diffusion_decode`, `single_pass_pooling`, `encoder_decoder_generation`, `media_encoder`, `projector`, `latent_denoising`, `vae_decode`, `vocoder`, or another namespaced mechanism. Edges state how components exchange tokens, embeddings, latents, tensors or media. A human-readable architecture summary may be indexed, but it never dispatches calculation or proves capability.

5. **Resource calculation dispatches by execution mechanism and concrete workload shape.** Text autoregressive decode uses instantiated cache mechanisms and token distributions. Multimodal generation additionally uses media preprocessing, encoder weights/compute/cache, media-token expansion and combination limits. Pooling uses sequence/token granularity and output dimensions without inventing a persistent decode KV budget. ASR uses audio duration/chunking, sample rate, encoder-decoder or streaming state and output-token behavior. Latent media generation uses component residency/offload, latent geometry, resolution, frames, denoising steps, batch, guidance and decode stages. Missing mechanism support returns `unknown`; modality names and nominal architecture families never select a fallback formula.

6. **Task, serving and protocol contracts remain separate.** `TaskSuiteSpec` owns the accepted semantic inputs, expected result shapes and success criteria. `ServingWorkloadSpec` owns distributions and concurrency for typed input and output ports: tokens, image dimensions/count, video dimensions/frames/fps/duration, audio duration/sample rate/channels, document pages/components, vector dimensions/count, streaming behavior and operation-specific parameters. The endpoint records its wire protocol separately, such as OpenAI chat/responses, embeddings, rerank, transcription, a media-generation API, or KServe/Triton tensors. A protocol mapping cannot create a capability.

7. **Units and economics follow the signature.** `VerificationReport` and `TaskAttemptRecord` retain raw provider billing units and typed physical/application units, including tokens, requests, images, frames, generated-media seconds, audio seconds, vector items and accepted outcomes. Throughput and latency are reported with operation, direction, scope and denominator. A bare `tokens/s`, `images/s`, or price cannot compare unlike signatures.

8. **The initial vLLM adapter is capability-aware, not text-hardcoded.** Against its pinned image it extracts runner type, supported task set, model implementation, multimodal combination/limits, exposed endpoints and applicable engine configuration. Its contract covers four distinct mechanism surfaces already present upstream: token/text generation; multimodal-input text generation; pooling/embedding/classification/scoring; and batch or realtime speech-to-text. A surface for which calculation, rendering or verification is not conforming remains visible with that stage `unknown` or `unsupported`; engine discovery is not mislabeled as complete Apron support.

9. **Generated media is a first-class product target through a separate engine adapter.** Image, video and audio generation cannot be represented as a vLLM flag variant. A conforming media adapter, for example over a pinned Diffusers-based service or Triton deployment, must implement its own resolver, mechanism-aware calculation, renderer, verifier, diagnostics, workload metrics and endpoint protocol while preserving the common task, solution, evidence and authorization contracts.

10. **Engine neutrality and mechanism/modality neutrality are different proofs.** A second LLM-serving adapter such as SGLang proves that the neutral core is not vLLM-specific. A generated-media adapter proves that the core is not token-decoding-specific. Neither substitutes for the other; the delivery plan carries both obligations and may order them by evidence value and authorized cost.

11. **Compound solutions preserve modality transitions explicitly.** `InferenceSolution` may connect ASR, embedding/reranking, language, vision, image/video generation and TTS endpoints. Every edge declares its input/output representation and transform; every endpoint retains its identity and evidence. End-to-end task success does not conceal an unsupported or changed intermediate capability.

12. **Media is an execution-security boundary.** Remote media retrieval is deny-by-default unless the accepted data policy and authorization allow it. Conforming adapters constrain domains and redirects, validate MIME/content rather than filename alone, enforce byte/duration/dimension/frame/decompression limits, isolate decoders and processors, redact sensitive metadata, and prevent media bytes or derived embeddings from entering logs or evidence without permission.

## Required fixtures and conformance

Before the first public schema freezes, golden fixtures cover at least:

- text to generated text;
- text plus image to generated text, including a forbidden-combination counterexample;
- audio file and audio stream to transcription output;
- text to embedding and query/document to score;
- text to generated image with a component graph containing a text encoder, denoiser and decoder;
- a compound audio-to-text-to-audio solution whose final audio endpoint may remain externally opaque;
- a checkpoint capability that the chosen engine does not expose;
- a discovered engine capability whose Apron calculator is still `unknown`.

Fixtures prove schema, migration and state semantics; they do not fabricate executable evidence. Every executable adapter passes the same signature, resource, protocol, security and evidence conformance suite for the surfaces it claims.

## Consequences

- Phase 1a can still use one bounded text-generation engineering fixture without making the product or public schema text-only.
- Existing records migrate by translating the old `text-autoregressive`, `encoder` and `diffusion` family payloads into component mechanisms plus explicit capability signatures; no information is discarded.
- A multimodal checkpoint is no longer called supported merely because its language backbone boots, and a text-only request does not erase loaded media components or their resource cost.
- The product can qualify managed, self-hosted and compound multimodal solutions honestly while retaining `provider_opaque`, `unknown` and `unsupported` boundaries.
- The permanent self-hosted calculator/deployment/diagnosis loop remains the base engine. Modality breadth composes additional proven mechanisms into that loop rather than replacing it with a generic benchmark router.
