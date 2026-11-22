# AMP

**Autonomous GPU Optimization Engine**

Give AMP a GPU workload. It reads the code, writes a few alternative implementations, checks whether each one is allowed to win, times them on a modeled device, and keeps the fastest result that still passes.

The first product is not a new GPU language and not a replacement for PyTorch. It optimizes the workload you already have.

Repository: [github.com/ManishaChindam0309/AMP](https://github.com/ManishaChindam0309/AMP)

---

## What you can do with it

1. Pick a target GPU in the header. The default is an RTX 2050 Laptop.
2. Paste PyTorch, CUDA, or Triton, or load one of the five bundled workloads.
3. Write a plain-language instruction such as “keep FP32” or “bf16 is acceptable on my H100”.
4. Run the optimizer. AMP returns the submitted code plus up to three candidates, a numerical gate, a memory check, and a single winner.
5. Open the same study in the profiler, the agent log, the kernel tile lab, or the benchmark table.
6. Use the LLM lab and the multi-GPU planner when the question is capacity and placement rather than a single kernel.

Studies stay in the browser. The latest run and the settings are stored in `localStorage`.

---

## What this version does not do

AMP does **not** attach a real GPU. It does not call NVML, Nsight, `torch.cuda`, or a remote runner.

Timing comes from a roofline model in [`src/lib/amp/model.ts`](src/lib/amp/model.ts):

- FLOPs and bytes are estimated from the operations AMP extracted.
- Each device contributes catalog peaks for FP32, TF32, BF16, FP16, and FP8, plus memory bandwidth, SM count, and a link rate.
- A step is the max of compute time, memory time, launch overhead, and host-sync time.
- Reported numbers are a median and a p95 over a fixed protocol: 10 warmup iterations, 50 timed iterations.
- Achieved TFLOP/s, the bound (`compute`, `memory`, `launch`, or `sync`), and VRAM are part of the same estimate.

Treat the speedup as a reasoned ranking of candidates on that model, not as a measured wall-clock result from your machine. Generated CUDA and Triton are source you can read and take elsewhere. This app does not compile or launch them.

---

## Modules

| Module | What it answers |
| --- | --- |
| Overview | Device peaks, precision support, the latest winner, bundled workloads, and recent studies. |
| Optimizer | Paste code, set the instruction and constraints, generate candidates, and pick a winner. |
| Profiler | Where time goes on the selected candidate: compute, memory, launch, and sync, plus the extracted ops. |
| Agent | A longer search. Each kept or rejected step is logged with its median time and speedup. |
| Kernel lab | Tile shape for this GPU (`BM`, `BN`, `BK`, warps, pipeline stages) and the efficiency of that tile. |
| LLM inference | Weight bytes, KV cache, whether a precision fits, TTFT, time per token, and tokens/sec. |
| Multi-GPU | Data parallel, tensor parallel, pipeline parallel, and a hybrid, and which one fits. |
| Benchmarks | Every candidate from the last study, eligible or rejected, with the reason. |

On a narrow screen the same modules sit in a horizontal bar. The GPU selector stays in the header and applies across the lab.

---

## How a study runs

```
code + instruction + GPU + precision + objective + memory budget
        │
        ▼
   analyze()          language, archetype, shapes, dtype, training, existing opts
        │
        ▼
 interpretInstruction()   "H100", "exact", "throughput", "under 8 GB" override the form
        │
        ▼
   buildModes()       submitted code + up to three alternatives
        │
        ▼
   scoreMode()        roofline time, VRAM, numerical gate, eligibility
        │
        ▼
   pickWinner()       fastest eligible candidate for latency or throughput
        │
        ▼
 explanation + optional agent log
```

### 1. Understand the code

[`analyze`](src/lib/amp/analyze.ts) is a structural read of the source, not a Python or CUDA parser. It looks for:

- Language: PyTorch, CUDA (`__global__`, `threadIdx`), Triton (`@triton.jit`), or mixed.
- Archetype: transformer, GEMM, norm, CUDA kernel, or generic.
- Tensor clues: `nn.Linear(in, out)`, `torch.randn(m, n)`, a `[batch, seq, hidden]` triple, `heads = N`, `n = N`, and `1 << N`.
- Dtype clues: FP32 by default, otherwise FP16, BF16, TF32, or FP8 when the source says so.
- Training vs inference: `backward`, `optimizer`, `loss`, `.step()`.
- Techniques already present: `torch.compile`, autocast, SDPA or FlashAttention, TF32, and activation checkpointing.
- Host sync: `.item()` and `.cpu()`, especially inside a loop. Loop trip count comes from `range(N)`.

The analysis object is what every later stage consumes: architecture summary, parameter count, op list, and the issues AMP will try to remove.

### 2. Read the instruction

[`interpretInstruction`](src/lib/amp/run.ts) only changes controls when the sentence actually says so.

| Phrase | Effect |
| --- | --- |
| A known GPU alias (`2050`, `4090`, `a100`, `h100`, …) | Switches the target device. |
| “throughput”, “tokens per”, “as fast as possible” | Objective becomes throughput. |
| “latency”, “time to first”, “response time” | Objective becomes latency. |
| “exact”, “fp32 only”, “keep fp32”, “do not quantize” | Precision policy becomes exact FP32. |
| “maximum speed”, “quantize”, “fp8”, “int4” | Precision policy becomes maximum speed. |
| “bf16” or “acceptable” | Auto policy; BF16 is allowed. |
| “N GB” together with memory, VRAM, under, or budget | Sets the memory budget. |

Anything the sentence does not mention stays as you set it in the form.

### 3. Write candidates

[`buildModes`](src/lib/amp/model.ts) always keeps the submitted code as the baseline. It then adds alternatives that the archetype and the precision policy allow. The optimizer shows at most four rows (baseline plus three). The agent can walk a longer list, bounded by the step budget in that lab (default 6).

[`renderCandidate`](src/lib/amp/codegen.ts) turns each mode into Python or CUDA you can read:

- Transformer blocks become compiled SDPA, BF16 autocast, or a fused path, depending on the mode.
- GEMMs become a Triton kernel with the tile chosen for the GPU, or a `torch.compile` wrapper.
- Norms become a fused Triton RMSNorm.
- CUDA kernels become a grid-stride launch or a `float4` vectorized kernel.
- Generic PyTorch gets a compile-and-dtype wrapper rather than a fake kernel.

### 4. Score and reject

Each candidate is scored on the selected GPU.

- **Time** splits into compute, memory, launch, and sync. The bound is whichever term dominates.
- **Speedup** is baseline median time divided by this candidate’s median time.
- **Memory** estimates weights, activations, and workspace. A candidate that does not fit the budget is ineligible.
- **Numerical gate** is one of `pass`, `approximate`, or `fail`. Exact FP32 stays a pass. BF16 and TF32 are approximate, with a printed relative tolerance. Gradients are `matched`, `scaled`, or `not-applicable`.
- **Eligibility.** A failed numerical check or an over-budget candidate cannot win. The reject reason is kept and shown in Benchmarks.

The winner is the eligible candidate with the best latency or the best throughput. If nothing else qualifies, the submitted code remains the winner.

The explanation under the winner states the device, the bound, the speedup, and why the other rows lost.

---

## Controls

| Control | Values | Meaning |
| --- | --- | --- |
| Target GPU | See the catalog below | Peaks, VRAM, and link rate used by every model. |
| Instruction | Free text | Optional overrides for device, objective, precision, and budget. |
| Precision | Auto (BF16 allowed), Exact FP32, Maximum speed | Which dtypes a candidate is allowed to use. |
| Objective | Latency or throughput | What “best” means among eligible rows. |
| Memory budget | 1 GB up to the device VRAM, in 0.5 GB steps | Candidates that do not fit are rejected. |
| Agent budget | Steps in the agent lab | How many alternatives the longer search may try. |

Changing the GPU resets the memory budget to that card’s VRAM when you had previously left the budget at the old card’s full size.

---

## Candidate strategies

Not every strategy is generated for every workload. AMP only emits the ones that match the archetype and the precision policy.

| Id | Name you will see | When it appears |
| --- | --- | --- |
| `baseline` | Submitted code | Always. |
| `drop-sync` | Remove host sync | The source calls `.item()` or `.cpu()`. |
| `compile-tf32` | TF32 compile | The GPU has TF32 and the policy is not exact. |
| `compile-fp32` | torch.compile, FP32 | Exact policy, or the GPU has no BF16. On CUDA sources this row is a grid-stride launch. |
| `compile-bf16` | torch.compile + bf16 | Auto or max-speed, and the GPU has BF16 tensor throughput. |
| `sdpa-bf16` | SDPA + bf16 compile | The workload is attention and does not already use SDPA. |
| `triton-fast` | Triton bf16 kernel, or fused Triton RMSNorm | Fast policy on a GEMM, norm, or generic op. |
| `triton-fp32` | Triton FP32 kernel, or fused Triton RMSNorm | Exact policy. On CUDA sources this is a persistent `float4` kernel. |
| `vectorized` | float4 vectorized | CUDA kernels only. |

---

## Device catalog

Peaks are published-style catalog numbers used by the model. They are not read from hardware. FP8 is zero on parts that have no FP8 tensor cores, and a candidate that needs FP8 will not pretend otherwise.

| Id | Card | Arch | VRAM | Bandwidth | FP32 | BF16 | FP8 | Link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rtx-2050` | RTX 2050 Laptop | Ampere 8.6 | 4 GB GDDR6 | 112 GB/s | 5.1 | 20 | — | PCIe 3.0 x8 |
| `rtx-4060` | RTX 4060 Laptop | Ada 8.9 | 8 GB GDDR6 | 256 GB/s | 14.6 | 58 | 116 | PCIe 4.0 x8 |
| `rtx-4090` | RTX 4090 | Ada 8.9 | 24 GB GDDR6X | 1008 GB/s | 82.6 | 330 | 660 | PCIe 4.0 x16 |
| `rtx-5090` | RTX 5090 | Blackwell 12.0 | 32 GB GDDR7 | 1792 GB/s | 105 | 419 | 838 | PCIe 5.0 x16 |
| `a100-80` | A100 80GB | Ampere 8.0 | 80 GB HBM2e | 2039 GB/s | 19.5 | 312 | — | NVLink 300 GB/s |
| `h100-sxm` | H100 SXM | Hopper 9.0 | 80 GB HBM3 | 3350 GB/s | 67 | 989 | 1979 | NVLink 4, 450 GB/s |

Tensor-core figures are dense, without sparsity. Units above are TFLOP/s except bandwidth and link, which are GB/s.

---

## Bundled workloads

Load any of these from Overview or from the chips in the optimizer. Each one ships with an instruction so the first run has a point.

| Sample | Shape of the work | Instruction that loads with it |
| --- | --- | --- |
| Transformer block | Pre-norm block, manual attention, LayerNorm, MLP, and a host sync. `x` is `[8, 2048, 1024]`, 16 heads. Training step included. | Make this training step faster on an RTX 2050. BF16 is acceptable. |
| Tiny GEMM loop | `32×32` FP32 matmul, result pulled to the host 1000 times. | Launch bound. Keep FP32 accuracy. |
| Square GEMM | One `2048²` FP32 matmul. | Make this matrix multiplication faster on an RTX 2050. |
| RMSNorm | Three memory passes over `[65536, 4096]` FP32. | Fuse the norm. Stay inside FP32 tolerance. |
| CUDA vector add | Scalar grid over `1 << 24` floats. | Vectorize the kernel. Exact FP32. |

You can paste your own source over a sample. Doing that clears the sample id so the study is yours.

### What to paste

AMP is most reliable when the source contains the numbers it needs:

- PyTorch training loops and inference functions.
- `nn.Linear`, matmul, LayerNorm, RMSNorm, softmax attention, or `scaled_dot_product_attention`.
- A comment or a tensor constructor with the real shape, for example `# a: [2048, 2048]`.
- CUDA kernels with `n = 1 << 24` or an explicit length.
- Triton kernels marked with `@triton.jit`.

If those clues are missing, AMP still classifies the file, but shapes fall back to defaults (for a transformer, batch 8, sequence 2048, hidden 1024, 16 heads). The summary on the analysis panel tells you what it assumed.

---

## LLM inference lab

Separate from kernel optimization. Pick a model, a GPU, a batch, a prompt length, and a generation length. AMP estimates five precisions and refuses the ones that do not fit.

Models:

- Llama 3.2 1B
- Qwen2.5 7B
- Llama 3 8B (default)
- Gemma 2 9B
- Mistral 7B
- Llama 3 70B

For each of BF16, FP16, FP8, INT8, and INT4 the report shows:

- A relative quality score (100, 100, 97, 94, 86).
- Weight bytes and KV-cache bytes. KV uses the model’s layer count, KV heads, and head dim.
- Total footprint against the memory budget and the card. A row that does not fit is marked and cannot win.
- Whether that precision is native on the card. FP8 is not native when the catalog FP8 peak is zero.
- Time to first token, time per output token, and tokens per second.
- Optional continuous batching and speculative decoding, which change the service-side estimate rather than the weight size.

The winner is the fastest precision that fits and matches the objective and the precision policy.

---

## Multi-GPU planner

Same model catalog. You set the GPU type, how many of them, training or inference, batch, sequence length, and microbatches.

AMP compares four placements:

| Strategy | Idea |
| --- | --- |
| Data parallel / replicas | Copy the model. Split the batch. Training pays a gradient all-reduce on the link. |
| Tensor parallel | Shard each linear layer. Activation all-reduces sit on the link. |
| Pipeline parallel | Shard by layer. A bubble remains, and it shrinks as you add microbatches. |
| Hybrid | Tensor parallel inside a node-sized group, data parallel across groups. |

A placement that does not fit in per-GPU memory is ineligible. The summary names the winner and why the others lost. Link bandwidth comes from the selected GPU’s catalog entry, so an RTX 2050 plan is PCIe and an H100 plan is NVLink.

---

## Kernel lab

For GEMM-shaped work, AMP picks a tile from a fixed list and scores it against the selected GPU:

- Tile sizes such as `64×64×32`, `128×128×64`, and `128×256×64`.
- Warp count and pipeline stages.
- A utilization estimate from how the tile fills the SMs and how much of the roofline it reaches.

The lab is a view of that choice. It does not search a live autotuner on hardware.

---

## Stack

| Piece | Choice |
| --- | --- |
| UI | React 19, Tailwind CSS 4 |
| App shell | TanStack Start and TanStack Router |
| State | Zustand, persisted in `localStorage` |
| Charts | Recharts, used by the profiler and benchmark views |
| Language | TypeScript 5 |
| Build | Vite 8 |

The engine itself is plain TypeScript under `src/lib/amp/`. It has no Python runtime and no CUDA toolkit dependency.

Auth, Postgres, and the PWA install chrome are part of the app shell. The lab does not require an account. Workloads are not uploaded.

---

## Run it locally

Requirements: Node.js 22.

```bash
npm install
npm run dev
```

The dev server listens on port `8080`. Open the URL it prints.

Other scripts:

| Script | Purpose |
| --- | --- |
| `npm run build` | Production build. |
| `npm run typecheck` | `tsc --noEmit`. |
| `npm run lint` | ESLint. |
| `npm run format` | Prettier. |
| `npm test` | Node test runner for the app-shell checks. |
| `npm run preview` | Serve the production build. |

There is no dataset to download and no API key to set for the lab. The models are formulas, not weights.

---

## Repository layout

```
src/components/amp/     screens: overview, optimizer, profiler, agent,
                        kernels, llm, multi-gpu, benchmarks
src/lib/amp/analyze.ts  structural read of the pasted source
src/lib/amp/codegen.ts  Python, Triton, and CUDA for each candidate
src/lib/amp/model.ts    roofline, tiles, eligibility, winner
src/lib/amp/run.ts      one study: analyze, score, explain, agent log
src/lib/amp/gpus.ts     device catalog
src/lib/amp/samples.ts  the five bundled workloads
src/lib/amp/llm.ts      inference precision report
src/lib/amp/parallel.ts data, tensor, pipeline, and hybrid plans
src/lib/amp/store.ts    UI state and history
src/lib/amp/types.ts    shared types
src/routes/             TanStack file routes. The lab is the index page.
```

---

## Product boundary

AMP is a reasoning and ranking engine for GPU work you already wrote.

In scope:

- Classify a PyTorch, CUDA, or Triton snippet.
- Propose a small set of concrete rewrites.
- Reject a rewrite that breaks the numerical policy or the memory budget.
- Rank the rest on a named GPU and show the work.

Out of scope for this repository:

- Compiling or benchmarking on a live device.
- Training or serving a model.
- Replacing PyTorch, Triton, or CUDA.

A useful next milestone is to take the winner AMP already selects and time it on a real RTX 2050, then keep the model only as a prior.
