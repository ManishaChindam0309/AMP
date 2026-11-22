import { renderCandidate } from "@/lib/amp/codegen";
import type {
  Analysis,
  Bench,
  Bound,
  Candidate,
  Check,
  DType,
  ExecMode,
  GpuSpec,
  PrecisionPolicy,
  StrategyId,
  Tile,
  WorkOp,
} from "@/lib/amp/types";

export const TILES: Tile[] = [
  { bm: 64, bn: 64, bk: 32, warps: 4, stages: 2 },
  { bm: 64, bn: 128, bk: 32, warps: 4, stages: 3 },
  { bm: 128, bn: 64, bk: 32, warps: 4, stages: 3 },
  { bm: 128, bn: 128, bk: 64, warps: 8, stages: 3 },
  { bm: 32, bn: 32, bk: 32, warps: 2, stages: 2 },
];

const SIMPLICITY: Record<StrategyId, number> = {
  baseline: 0,
  "drop-sync": 1,
  "compile-tf32": 2,
  "compile-fp32": 3,
  "compile-bf16": 4,
  "sdpa-bf16": 5,
  vectorized: 6,
  "triton-fp32": 7,
  "triton-fast": 8,
};

export function bestTile(gpu: GpuSpec): Tile {
  return [...TILES].sort((a, b) => tileEff(gpu, b) - tileEff(gpu, a))[0];
}

export function tileEff(gpu: GpuSpec, tile: Tile): number {
  const smem = (tile.bm * tile.bk + tile.bk * tile.bn) * tile.stages * 2;
  if (smem > 96 * 1024) return 0.18;
  const datacenter = gpu.fp32 >= 40 || gpu.bandwidthGBs >= 1500;
  const large = tile.bm >= 128 && tile.bn >= 128;
  if (datacenter && large) return 0.57;
  if (datacenter) return 0.46;
  if (!large) return tile.bm <= 32 ? 0.52 : 0.7;
  return 0.48;
}

function bytesOf(dtype: DType): number {
  if (dtype === "fp8") return 1;
  if (dtype === "bf16" || dtype === "fp16") return 2;
  return 4;
}

function peakOf(gpu: GpuSpec, dtype: DType): number {
  if (dtype === "fp8") return gpu.fp8 || gpu.bf16 || gpu.fp32;
  if (dtype === "bf16") return gpu.bf16 || gpu.fp32;
  if (dtype === "fp16") return gpu.fp16 || gpu.fp32;
  if (dtype === "tf32") return gpu.tf32 || gpu.fp32;
  return gpu.fp32;
}

function supports(gpu: GpuSpec, dtype: DType): boolean {
  if (dtype === "fp8") return gpu.fp8 > 0;
  if (dtype === "bf16") return gpu.bf16 > 0;
  if (dtype === "tf32") return gpu.tf32 > 0;
  return true;
}

function launchUs(mode: ExecMode, flops: number): number {
  const tiny = flops < 5_000_000;
  if (mode.tritonGemm || mode.tritonNorm) return tiny ? 9 : 4;
  if (mode.compiled) return tiny ? 22 : 2.4;
  if (mode.vectorWidth === 4) return tiny ? 5 : 3;
  return 6.5;
}

type Part = {
  us: number;
  computeUs: number;
  memoryUs: number;
  launchUs: number;
  syncUs: number;
  vramBytes: number;
  flops: number;
};

function emptyPart(): Part {
  return { us: 0, computeUs: 0, memoryUs: 0, launchUs: 0, syncUs: 0, vramBytes: 0, flops: 0 };
}

function add(a: Part, b: Part): Part {
  return {
    us: a.us + b.us,
    computeUs: a.computeUs + b.computeUs,
    memoryUs: a.memoryUs + b.memoryUs,
    launchUs: a.launchUs + b.launchUs,
    syncUs: a.syncUs + b.syncUs,
    vramBytes: a.vramBytes + b.vramBytes,
    flops: a.flops + b.flops,
  };
}

function evalOp(op: WorkOp, gpu: GpuSpec, mode: ExecMode, training: boolean): Part {
  const part = emptyPart();
  const bpe = bytesOf(mode.dtype === "tf32" ? "fp32" : mode.dtype);
  const train = training ? 3 : 1;
  const recompute = mode.checkpoint && training ? 1.3 : 1;

  if (op.kind === "sync") {
    if (mode.dropSync) return part;
    const each = gpu.sms < 40 ? 180 : 70;
    part.syncUs = op.calls * each;
    part.us = part.syncUs;
    return part;
  }

  if (op.kind === "gemm") {
    const flops = 2 * op.m * op.n * op.k * op.count * train * recompute;
    const traffic =
      (op.m * op.k + op.k * op.n + op.m * op.n) * op.count * bpe * (training ? 2.2 : 1);
    const eff = mode.tritonGemm && mode.tile ? tileEff(gpu, mode.tile) : gemmEff(mode);
    const computeUs = (flops / (peakOf(gpu, mode.dtype) * 1e12 * eff)) * 1e6;
    const memoryUs = (traffic / (gpu.bandwidthGBs * 1e9)) * 1e6;
    const launches = op.count * (training && !mode.compiled ? 2 : 1);
    const launch = launches * launchUs(mode, flops / Math.max(1, op.count));
    part.flops = flops;
    part.computeUs = computeUs;
    part.memoryUs = memoryUs;
    part.launchUs = launch;
    part.us = Math.max(computeUs, memoryUs) + launch;
    part.vramBytes = op.m * op.n * op.count * bpe * (training ? 2 : 1);
    return part;
  }

  if (op.kind === "attention") {
    const { batch: B, heads: H, seq: S, headDim: Dh, count } = op;
    const flops = 4 * B * H * S * S * Dh * count * train * recompute;
    const flash = mode.flash;
    const scoreBytes = flash ? 0 : B * H * S * S * count * bpe * (training ? 2 : 1);
    const actBytes = 4 * B * S * H * Dh * count * bpe;
    const traffic = scoreBytes + actBytes * (training ? 2 : 1);
    const eff = flash
      ? mode.tritonGemm && mode.tile
        ? tileEff(gpu, mode.tile) * 0.92
        : 0.52
      : mode.compiled
        ? 0.28
        : 0.18;
    const computeUs = (flops / (peakOf(gpu, mode.dtype) * 1e12 * eff)) * 1e6;
    const memoryUs = (traffic / (gpu.bandwidthGBs * 1e9)) * 1e6;
    const kernels = flash ? count : count * 6;
    const launch = kernels * launchUs(mode, flops) * (training && !mode.compiled && !flash ? 2 : 1);
    part.flops = flops;
    part.computeUs = computeUs;
    part.memoryUs = memoryUs;
    part.launchUs = launch;
    part.us = Math.max(computeUs, memoryUs) + launch;
    part.vramBytes = scoreBytes + actBytes;
    return part;
  }

  if (op.kind === "norm") {
    const fused = mode.tritonNorm || mode.compiled;
    const passes = fused ? 1 : 3;
    const eff = mode.tritonNorm ? 0.8 : mode.compiled ? 0.62 : 0.45;
    const traffic = op.elements * op.count * bpe * 2 * passes * (training ? 2 : 1);
    const memoryUs = (traffic / (gpu.bandwidthGBs * 1e9 * eff)) * 1e6;
    const launch = op.count * (fused ? 1 : passes) * launchUs(mode, op.elements);
    part.memoryUs = memoryUs;
    part.launchUs = launch;
    part.us = memoryUs + launch;
    part.vramBytes = op.elements * op.count * bpe;
    part.flops = op.elements * op.count * 8;
    return part;
  }

  if (op.kind === "pointwise") {
    if (mode.compiled || mode.tritonGemm || op.fused) {
      part.vramBytes = op.elements * bpe * 0.25;
      return part;
    }
    const traffic = op.elements * op.count * bpe * 2;
    const memoryUs = (traffic / (gpu.bandwidthGBs * 1e9 * 0.5)) * 1e6;
    const launch = op.count * launchUs(mode, op.elements);
    part.memoryUs = memoryUs;
    part.launchUs = launch;
    part.us = memoryUs + launch;
    part.vramBytes = op.elements * bpe;
    return part;
  }

  const eff = mode.vectorWidth === 4 ? (mode.compiled ? 0.86 : 0.84) : mode.compiled ? 0.68 : 0.55;
  const traffic = op.elements * op.count * 4 * 3;
  const memoryUs = (traffic / (gpu.bandwidthGBs * 1e9 * eff)) * 1e6;
  const launch = launchUs(mode, op.elements);
  part.memoryUs = memoryUs;
  part.launchUs = launch;
  part.us = memoryUs + launch;
  part.vramBytes = op.elements * 4 * 3;
  return part;
}

function gemmEff(mode: ExecMode): number {
  if (mode.compiled && (mode.dtype === "bf16" || mode.dtype === "fp16")) return 0.62;
  if (mode.compiled && mode.dtype === "tf32") return 0.55;
  if (mode.compiled) return 0.58;
  if (mode.dtype === "bf16" || mode.dtype === "fp16") return 0.4;
  if (mode.dtype === "tf32") return 0.48;
  return 0.42;
}

function weightBytes(params: number, mode: ExecMode, training: boolean): number {
  if (params <= 0) return 0;
  const bpe = bytesOf(mode.dtype === "tf32" ? "fp32" : mode.dtype);
  if (!training) return params * bpe;
  if (mode.dtype === "bf16" || mode.dtype === "fp16" || mode.dtype === "fp8") {
    return params * bpe + params * 4 + params * 8;
  }
  return params * 4 * 3;
}

export function baselineMode(analysis: Analysis): ExecMode {
  let dtype: DType = analysis.dtype === "fp8" ? "fp8" : analysis.dtype;
  if (dtype === "fp32" && analysis.hasTf32) dtype = "tf32";
  return {
    id: "baseline",
    name: "Submitted code",
    compiled: analysis.hasCompile,
    dtype,
    flash: analysis.hasSdpa,
    tritonGemm: false,
    tritonNorm: analysis.language === "triton" && analysis.archetype === "norm",
    dropSync: false,
    checkpoint: analysis.hasCheckpoint,
    vectorWidth: 1,
    tile: null,
  };
}

export function buildModes(analysis: Analysis, gpu: GpuSpec, policy: PrecisionPolicy): ExecMode[] {
  const base = baselineMode(analysis);
  const tile = bestTile(gpu);
  const modes: ExecMode[] = [base];
  const push = (mode: ExecMode) => {
    if (!modes.some((item) => item.id === mode.id)) modes.push(mode);
  };

  if (analysis.archetype === "cuda") {
    push({
      ...base,
      id: "compile-fp32",
      name: "Grid-stride launch",
      compiled: true,
      dtype: "fp32",
      dropSync: true,
      tile: null,
    });
    push({
      ...base,
      id: "vectorized",
      name: "float4 vectorized",
      compiled: false,
      dtype: "fp32",
      vectorWidth: 4,
      dropSync: true,
    });
    push({
      ...base,
      id: "triton-fp32",
      name: "Persistent float4",
      compiled: true,
      dtype: "fp32",
      vectorWidth: 4,
      dropSync: true,
    });
    return modes.slice(0, 4);
  }

  if (analysis.hasSync) {
    push({
      ...base,
      id: "drop-sync",
      name: "Remove host sync",
      dropSync: true,
    });
  }

  const wantFast = policy !== "exact" && gpu.bf16 > 0;
  const hasAttention = analysis.ops.some((op) => op.kind === "attention");

  if (wantFast && hasAttention && !base.flash) {
    push({
      id: "sdpa-bf16",
      name: "SDPA + bf16 compile",
      compiled: true,
      dtype: "bf16",
      flash: true,
      tritonGemm: false,
      tritonNorm: false,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile: null,
    });
  } else if (wantFast) {
    push({
      id: "compile-bf16",
      name: "torch.compile + bf16",
      compiled: true,
      dtype: "bf16",
      flash: base.flash,
      tritonGemm: false,
      tritonNorm: false,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile: null,
    });
  } else {
    push({
      id: "compile-fp32",
      name: "torch.compile, FP32",
      compiled: true,
      dtype: "fp32",
      flash: base.flash,
      tritonGemm: false,
      tritonNorm: false,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile: null,
    });
  }

  if (policy === "exact" || gpu.bf16 <= 0) {
    push({
      id: "triton-fp32",
      name: analysis.archetype === "norm" ? "Fused Triton RMSNorm" : "Triton FP32 kernel",
      compiled: false,
      dtype: "fp32",
      flash: hasAttention,
      tritonGemm: analysis.archetype === "gemm" || analysis.archetype === "generic",
      tritonNorm: true,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile,
    });
  } else {
    push({
      id: "triton-fast",
      name: analysis.archetype === "norm" ? "Fused Triton RMSNorm" : "Triton bf16 kernel",
      compiled: analysis.archetype === "transformer",
      dtype: "bf16",
      flash: hasAttention,
      tritonGemm: analysis.archetype === "gemm" || analysis.archetype === "generic",
      tritonNorm: true,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile,
    });
  }

  if (modes.length < 4 && gpu.tf32 > 0 && policy !== "exact") {
    push({
      id: "compile-tf32",
      name: "TF32 compile",
      compiled: true,
      dtype: "tf32",
      flash: base.flash,
      tritonGemm: false,
      tritonNorm: false,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile: null,
    });
  }

  return modes.slice(0, 4);
}

export function agentModes(analysis: Analysis, gpu: GpuSpec, policy: PrecisionPolicy): ExecMode[] {
  const base = baselineMode(analysis);
  const tile = bestTile(gpu);
  const hasAttention = analysis.ops.some((op) => op.kind === "attention");
  const list: ExecMode[] = [base];
  if (analysis.archetype === "cuda") {
    return buildModes(analysis, gpu, policy);
  }
  if (analysis.hasSync) {
    list.push({ ...base, id: "drop-sync", name: "Remove host sync", dropSync: true });
  }
  if (gpu.tf32 > 0) {
    list.push({
      id: "compile-tf32",
      name: "TF32 compile",
      compiled: true,
      dtype: "tf32",
      flash: base.flash,
      tritonGemm: false,
      tritonNorm: false,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile: null,
    });
  }
  list.push({
    id: "compile-fp32",
    name: "torch.compile, FP32",
    compiled: true,
    dtype: "fp32",
    flash: base.flash,
    tritonGemm: false,
    tritonNorm: false,
    dropSync: true,
    checkpoint: false,
    vectorWidth: 1,
    tile: null,
  });
  if (policy !== "exact" && gpu.bf16 > 0) {
    list.push({
      id: "compile-bf16",
      name: "bf16 autocast + compile",
      compiled: true,
      dtype: "bf16",
      flash: base.flash,
      tritonGemm: false,
      tritonNorm: false,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile: null,
    });
    if (hasAttention) {
      list.push({
        id: "sdpa-bf16",
        name: "Flash SDPA + bf16",
        compiled: true,
        dtype: "bf16",
        flash: true,
        tritonGemm: false,
        tritonNorm: false,
        dropSync: true,
        checkpoint: false,
        vectorWidth: 1,
        tile: null,
      });
    }
    list.push({
      id: "triton-fast",
      name: "Triton bf16 autotune",
      compiled: true,
      dtype: "bf16",
      flash: hasAttention,
      tritonGemm: true,
      tritonNorm: true,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile,
    });
  } else {
    list.push({
      id: "triton-fp32",
      name: "Triton FP32 autotune",
      compiled: false,
      dtype: "fp32",
      flash: hasAttention,
      tritonGemm: true,
      tritonNorm: true,
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile,
    });
  }
  return list;
}

function techniqueList(mode: ExecMode): string[] {
  const items: string[] = [];
  if (mode.id === "baseline") items.push("submitted implementation");
  if (mode.dropSync && mode.id !== "baseline") items.push("sync removed");
  if (mode.compiled) items.push("torch.compile");
  if (mode.dtype === "bf16") items.push("bf16 autocast");
  if (mode.dtype === "fp16") items.push("fp16 autocast");
  if (mode.dtype === "tf32") items.push("TF32 matmul");
  if (mode.dtype === "fp32" && mode.id !== "baseline") items.push("FP32");
  if (mode.flash) items.push("FlashAttention / SDPA");
  if (mode.tritonGemm) items.push("Triton matmul");
  if (mode.tritonNorm) items.push("fused Triton norm");
  if (mode.vectorWidth === 4) items.push("float4 loads");
  if (mode.checkpoint) items.push("activation checkpoint");
  return items.length ? items : ["no rewrite"];
}

function summarize(mode: ExecMode, analysis: Analysis): string {
  if (mode.id === "baseline") return "Unmodified source, scored as the reference.";
  if (mode.id === "drop-sync") {
    return analysis.loopIters > 1
      ? "Keeps the math. Moves the host read out of the inner loop so the GPU can stay busy."
      : "Returns the tensor instead of a host scalar, so the step no longer syncs early.";
  }
  if (mode.id === "vectorized") return "Issues 128-bit loads and a grid-stride loop to raise achieved bandwidth.";
  if (mode.flash && mode.tritonGemm) {
    return "SDPA for attention, a tiled Triton GEMM for the projections, fused norm, bf16 where the device allows it.";
  }
  if (mode.flash) return "Replaces the score matrix with scaled-dot-product attention and compiles the block.";
  if (mode.tritonNorm && analysis.archetype === "norm") {
    return "One kernel computes variance, rsqrt, and the weight multiply.";
  }
  if (mode.tritonGemm) return `Tiled kernel, block ${mode.tile?.bm}×${mode.tile?.bn}×${mode.tile?.bk}.`;
  if (mode.dtype === "bf16") return "Compiles the eager graph and runs GEMMs on tensor cores in bf16.";
  if (mode.dtype === "tf32") return "Leaves storage in FP32. Matmuls use TF32 tensor cores.";
  return "Compiles the graph to cut launch overhead without changing precision.";
}

export function scoreMode(
  analysis: Analysis,
  gpu: GpuSpec,
  mode: ExecMode,
  policy: PrecisionPolicy,
  memoryBudgetGb: number,
  baselineUs: number | null,
): Candidate {
  let part = emptyPart();
  for (const op of analysis.ops) part = add(part, evalOp(op, gpu, mode, analysis.training));
  const weights = weightBytes(analysis.params, mode, analysis.training);
  const overhead = 256 * 1024 * 1024;
  let vramBytes = part.vramBytes + weights + overhead;
  if (mode.checkpoint && analysis.training) vramBytes = weights + overhead + part.vramBytes * 0.45;

  const fitsDevice = vramBytes <= gpu.vramGb * 1024 * 1024 * 1024 * 0.92;
  const bound = dominantBound(part);
  const medianUs = Math.max(0.5, part.us);
  const tail =
    part.syncUs > medianUs * 0.25 ? 1.42 : part.launchUs > medianUs * 0.25 ? 1.3 : 1.15;
  const seconds = medianUs / 1e6;
  const bench: Bench = {
    medianUs,
    p95Us: medianUs * tail,
    itersPerSec: 1 / seconds,
    vramMb: vramBytes / (1024 * 1024),
    achievedTflops: part.flops > 0 ? part.flops / seconds / 1e12 : 0,
    bound,
    speedup: baselineUs && baselineUs > 0 ? baselineUs / medianUs : 1,
    computeUs: part.computeUs,
    memoryUs: part.memoryUs,
    launchUs: part.launchUs,
    syncUs: part.syncUs,
  };

  const check = buildCheck(mode, analysis, fitsDevice);
  const overBudget = bench.vramMb / 1024 > memoryBudgetGb;
  let eligible = check.numerical !== "fail" && fitsDevice && !overBudget;
  let rejectReason: string | undefined;
  if (!supports(gpu, mode.dtype)) {
    eligible = false;
    rejectReason = `${mode.dtype.toUpperCase()} is not native on ${gpu.name}.`;
    check.numerical = "fail";
  } else if (check.numerical === "fail") {
    rejectReason = "Failed the numerical gate for this precision policy.";
  } else if (!fitsDevice) {
    rejectReason = `Working set ${bench.vramMb.toFixed(0)} MB does not fit in ${gpu.vramGb} GB.`;
  } else if (overBudget) {
    rejectReason = `Uses ${(bench.vramMb / 1024).toFixed(2)} GB, over the ${memoryBudgetGb} GB budget.`;
  } else if (policy === "exact" && check.numerical !== "pass") {
    eligible = false;
    rejectReason = "Approximate precision rejected. Policy is exact FP32.";
  }

  return {
    id: mode.id,
    name: mode.name,
    summary: summarize(mode, analysis),
    techniques: techniqueList(mode),
    code: renderCandidate(analysis, mode, gpu),
    language: analysis.archetype === "cuda" ? "cuda" : "python",
    bench,
    check,
    eligible,
    rejectReason,
    simplicity: SIMPLICITY[mode.id],
  };
}

function buildCheck(mode: ExecMode, analysis: Analysis, fits: boolean): Check {
  const exact = mode.dtype === "fp32" || (mode.dtype === "tf32" && false);
  const numerical = mode.dtype === "fp8" ? "fail" : mode.dtype === "fp32" ? "pass" : "approximate";
  const rtol =
    numerical === "pass" ? "1e-4" : mode.dtype === "tf32" ? "1e-3" : mode.dtype === "bf16" ? "2e-2" : "1e-2";
  const gradients = !analysis.training
    ? "not-applicable"
    : numerical === "pass"
      ? "matched"
      : "scaled";
  const notes: string[] = [];
  if (mode.dtype === "bf16") notes.push("bf16 autocast. fp32 master weights assumed for training.");
  if (mode.dtype === "tf32") notes.push("TF32 changes GEMM rounding only. Reductions stay FP32.");
  if (mode.dtype === "fp32") notes.push("Same precision class as a careful FP32 reference.");
  if (mode.flash) notes.push("Online softmax matches SDPA within the stated tolerance.");
  if (!fits) notes.push("Allocator would have to spill. Treated as a failed run.");
  if (exact) {
    /* marker for readers */
  }
  return {
    numerical: mode.dtype === "fp8" ? "fail" : numerical,
    rtol,
    gradients,
    memorySafe: fits && mode.vectorWidth !== 1 ? true : fits,
    notes,
  };
}

function dominantBound(part: Part): Bound {
  const hiddenCompute = Math.max(0, part.computeUs - part.memoryUs);
  const hiddenMemory = Math.max(0, part.memoryUs - part.computeUs);
  const overlap = Math.min(part.computeUs, part.memoryUs);
  const compute = hiddenCompute + (part.computeUs >= part.memoryUs ? overlap : 0);
  const memory = hiddenMemory + (part.memoryUs > part.computeUs ? overlap : 0);
  const pairs: [Bound, number][] = [
    ["compute", compute],
    ["memory", memory],
    ["launch", part.launchUs],
    ["sync", part.syncUs],
  ];
  pairs.sort((a, b) => b[1] - a[1]);
  return pairs[0][0];
}

export function pickWinner(candidates: Candidate[], objective: "latency" | "throughput"): Candidate {
  const eligible = candidates.filter((c) => c.eligible);
  const pool = eligible.length ? eligible : candidates;
  const best = Math.min(...pool.map((c) => c.bench.medianUs));
  const near = pool.filter((c) => c.bench.medianUs <= best * 1.03);
  near.sort((a, b) => {
    if (objective === "throughput" && Math.abs(a.bench.vramMb - b.bench.vramMb) > 32) {
      const timeTie = Math.abs(a.bench.medianUs - b.bench.medianUs) <= best * 0.03;
      if (timeTie) return a.bench.vramMb - b.bench.vramMb;
    }
    if (a.simplicity !== b.simplicity) return a.simplicity - b.simplicity;
    return a.bench.medianUs - b.bench.medianUs;
  });
  return near[0];
}

export function scoreAll(
  analysis: Analysis,
  gpu: GpuSpec,
  modes: ExecMode[],
  policy: PrecisionPolicy,
  memoryBudgetGb: number,
  objective: "latency" | "throughput",
): Candidate[] {
  const baseline = scoreMode(analysis, gpu, modes[0], policy, memoryBudgetGb, null);
  const rest = modes.slice(1).map((mode) =>
    scoreMode(analysis, gpu, mode, policy, memoryBudgetGb, baseline.bench.medianUs),
  );
  baseline.bench.speedup = 1;
  return [baseline, ...rest].map((candidate) => ({
    ...candidate,
    bench: { ...candidate.bench, speedup: baseline.bench.medianUs / candidate.bench.medianUs },
  }));
}
