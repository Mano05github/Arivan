# Arivan

```text
$ arivan

அறிவன் — Think. Code. Solve.

Ready to explore your codebase.

arivan > Explain this repository
```

Arivan is an experimental, CPU-first coding-agent runtime designed to run
large open-weight models under explicit memory budgets. It combines native C
inference engines with bounded memory profiles, a stable command-line
interface, and a path toward low-memory model execution.

> [!IMPORTANT]
> The 8 GiB profile is currently an admission-control contract, not a claim
> that GLM-5.3 Flash already runs within 8 GiB. Dense and vision paging are
> the next engine milestones.

## Current status

Version: `0.1.0.dev0`

- Independent `arivan` command and Python package
- Explicit 8, 16, and 32 GiB memory profiles
- Native C hard-budget accounting by allocation category and runtime phase
- Pre-allocation admission checks for resident text and vision tensors
- Load-time quantization peak accounting, including temporary f32 buffers
- GLM-5.3 Flash load-peak measurement and expert-cache admission control
- Zero-cache GLM-5.3 expert execution through bounded reusable staging slots
- Opt-in phase telemetry with `ARIVAN_MEMORY_TELEMETRY=1`
- Cross-platform peak-memory measurement
- Temporary compatibility with legacy `COLI_*` environment variables
- Vision, pinned-cache, and MTP budgets represented in the profile contract
- Python CLI regression tests and native C memory-budget tests

Arivan currently refuses unsafe low-memory configurations instead of allowing
the process to overcommit and fail later. The expert-cache floor has been
removed; dense and vision weights remain the principal 8 GiB blockers.

The native runtime currently includes adapters for GLM-5.3-Flash,
GLM-5.2/5.3, Inkling, Kimi K3, OLMoE, Qwen3.6, Qwen3.8-Flash-Next,
DeepSeek V4 Flash, and DeepSeek V4.1 Flash.

## Quick start

Arivan requires Python 3.10 or newer.

```powershell
git clone https://github.com/Mano05github/Arivan.git
cd Arivan
python -m pip install -e .
arivan --version
arivan profile 8gb
```

From a source checkout on Windows, the launcher can also be used directly:

```powershell
.\arivan.cmd --version
.\arivan.cmd profile 8gb
```

To use a native model engine, set a checkpoint directory and run Arivan:

```powershell
$env:ARIVAN_MODEL = "D:\Models\your-model"
arivan chat
```

Native engines must be compiled for the target platform. Model checkpoints are
not included in this repository and retain their own licenses.

## Memory profiles

| Profile | Engine budget | Context | Vision tokens | Pinned cache | Expert cache | Status |
|---|---:|---:|---:|---:|---:|---|
| `8gb` | 5.25 GiB | 2,048 | 128 | 384 MiB | 512 MiB | Experimental |
| `16gb` | 12 GiB | 8,192 | 256 | 1 GiB | 2 GiB | Experimental |
| `32gb` | 26 GiB | 32,768 | 512 | 4 GiB | 8 GiB | Baseline |

Inspect a profile as text or JSON:

```powershell
arivan profile 8gb
arivan profile 8gb --json
```

Apply a profile to an inherited command:

```powershell
arivan --memory-profile 8gb chat --model D:\Models\GLM-5.3-Flash
```

## Architecture

Arivan keeps performance-critical inference work in native C. Python owns the
CLI, planning, conversion, API, and future agent orchestration.

```text
Arivan CLI and agent control plane
              |
Model registry, planner, and API
              |
Native model adapters (C)
              |
Memory budget, tensor pager, KV state, expert cache
              |
Quantized CPU/GPU kernels and storage-backed weights
```

See [docs/ARIVAN_ARCHITECTURE.md](docs/ARIVAN_ARCHITECTURE.md) for the design,
delivery sequence, and GLM-5.3 Flash acceptance gates.

## Roadmap

The next development milestone is correct GLM-5.3 Flash text generation below
the 8 GiB profile's 5.25 GiB engine ceiling:

1. Stream dense text weights through bounded layer windows.
2. Load and release vision weights by runtime phase.
3. Replace equal per-layer pinning with a global byte-budgeted cache.
4. Restore int8 MTP depth one and enable it only when it improves net speed.
5. Validate tokens, memory peaks, SSD traffic, vision, tools, and MTP.

Later releases will add repository tools and multiple logical agents sharing
one serialized model process. Arivan does not promise identical performance at
8 GiB and 32 GiB; lower RAM necessarily increases storage traffic.

## Development

Run the Python regression suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest c.tests.test_arivan_cli c.tests.test_cli_output
```

With a supported native toolchain:

```text
make -C c arivan-check
```

The repository also includes broader engine and integration tests.

## Project layout

```text
arivan/                    Arivan CLI and memory-profile control plane
c/arivan_memory.*          Native memory-budget API
c/glm53.c                  GLM-5.3 Flash adapter and admission hook
c/                         Inherited native engines and shared runtime
docs/ARIVAN_ARCHITECTURE.md
web/                       Inherited web interface
desktop/                   Inherited desktop shell
```

## License

Licensed under the Apache License 2.0. Copyright, attribution, and third-party
notices are retained in [NOTICE](NOTICE), [LICENSE](LICENSE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) as required.
