# Arivan architecture

Arivan is a CPU-first coding-agent runtime. Its defining constraint is a hard,
observable memory budget: the engine must account for
resident weights, file-backed working pages, expert caches, state and temporary
workspaces instead of treating `--ram` as an expert-cache hint.

## Product contract

- Preserve the model's selected experts and mathematical execution in the
  quality-preserving mode.
- Preserve vision, tool calling and native MTP when the checkpoint provides
  those capabilities.
- Run multiple logical agents through one shared model process on constrained
  machines; model inference is serialized in the 8 GB profile.
- Match or improve the inherited engine at its normal memory tier. Lower-memory
  tiers minimize and report their unavoidable SSD-I/O slowdown.
- Certify individual checkpoint, context and hardware combinations. “Any model
  in 8 GB at unchanged speed” is explicitly not a product claim.

## Runtime layers

1. `arivan` provides the terminal interface, agent loop and stable environment
   names.
2. The Python control plane detects the model, plans resources, converts model
   containers and exposes the OpenAI-compatible API.
3. Native C adapters implement model graphs and model-specific operators.
4. The shared C runtime owns the hard memory budget, tensor paging, KV state,
   expert placement and asynchronous I/O.
5. Quantized CPU kernels execute directly from the stored representation.

Python must not hold model weights during inference. Performance-critical
allocation, paging, cache and kernel decisions belong to the native runtime.

## Memory profiles

The initial profiles are declared in both `arivan/memory.py` and
`c/arivan_memory.c`. The duplicated definitions are temporary bootstrap code;
the native profile table will become the source of truth once it is exposed to
the control plane through a stable plan protocol.

The native budget records accounted peaks separately for load, vision,
prefill, and decode. Set `ARIVAN_MEMORY_TELEMETRY=1` to print transitions and
the current, global-peak, phase-peak, limit, available byte counts, observed
RSS peak, and the baseline, resident-weight, dense-window, vision-window,
vision-state, and workspace categories. Exact byte counters accompany the
human-readable MiB values. Profile enforcement starts before the first model
allocation. Load-time quantization accounts its temporary f32 source together
with the final resident representation, then transactionally reclassifies or
releases the temporary bytes.

The 8 GB profile reserves 2.75 GiB for the operating system and gives the
engine a hard 5.25 GiB ceiling. It starts with one KV slot, 2,048 text tokens,
128 visual tokens, a 384 MiB pin budget, a 512 MiB ordinary expert cache,
prefill chunks of eight and MTP depth one. When a persistent expert slot for
every sparse layer does not fit, the adapter falls back to one or two bounded
whole-model staging slots. Resident text and vision tensors now use the native
allocation ledger. Under a native profile, transformer matrices are described
at startup and materialized synchronously into a single-layer `dense-window`
allocation, which is released before advancing to the next layer. Token
embeddings are read by prompt row, and the output projection runs in bounded
vocabulary blocks while preserving tied-weight semantics. Vision execution
loads the patch projection, one transformer block, and one merger matrix at a
time; each stage is released before the next, and activation buffers are
admitted against the profile's image-token limit. The profile remains
experimental until all text-forward workspaces use the native budget and a
real checkpoint passes the hardware certification gates.

## Delivery order

1. Independent package, branding, provenance and regression baseline.
2. Native budget accounting and phase transitions.
3. Directly executable, offline-quantized dense weights.
4. Bounded dense-layer and expert streaming with cache capacity zero.
5. Global byte-budgeted pinned/LRU expert cache.
6. Vision-phase loading and release.
7. Int8 MTP with adaptive net-speed gating.
8. Planner admission control and agent-context scheduling.
9. GLM-5.3, OLMoE, Inkling and generic dense/MoE adapters.

## GLM-5.3 Flash 8 GB acceptance gates

- Peak engine commit does not exceed 5.25 GiB during load, vision, prefill or
  decode on the reference configuration.
- The operating system does not enter sustained pagefile/swap thrashing.
- Placement-only changes produce the reference tokens exactly.
- Quantized-container changes satisfy predefined perplexity, coding, tool-use
  and vision thresholds.
- All router-selected experts execute; expert dropping is not permitted in the
  default mode.
- MTP remains enabled only when it produces a measured end-to-end speedup.
- Metrics include peak private/file-backed memory, faults, bytes per token,
  time to first token, cold/warm decode and MTP acceptance.
