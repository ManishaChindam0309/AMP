# AMP

Autonomous GPU Optimization Engine.

Paste a PyTorch, CUDA, or Triton workload. AMP analyzes the shape of the work, generates candidate implementations, checks numerical eligibility and memory, benchmarks them on a modeled GPU, and selects the fastest correct result.

Modeled devices include an RTX 2050 and an H100. This app does not attach a real GPU.

```bash
npm install
npm run dev
```

Open the local URL the dev server prints.
