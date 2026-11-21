export type Objective = "latency" | "throughput";
export type PrecisionPolicy = "auto" | "exact" | "max-speed";
export type DType = "fp32" | "fp16" | "bf16" | "tf32" | "fp8";
export type Archetype = "transformer" | "gemm" | "norm" | "cuda" | "generic";
export type Language = "pytorch" | "cuda" | "triton" | "mixed";
export type Bound = "compute" | "memory" | "launch" | "sync";

export type ModuleId =
  | "overview"
  | "optimizer"
  | "profiler"
  | "agent"
  | "kernels"
  | "llm"
  | "multigpu"
  | "benchmarks";

export type StrategyId =
  | "baseline"
  | "drop-sync"
  | "compile-tf32"
  | "compile-fp32"
  | "compile-bf16"
  | "sdpa-bf16"
  | "triton-fast"
  | "triton-fp32"
  | "vectorized";

export type WorkOp =
  | { kind: "gemm"; label: string; m: number; n: number; k: number; count: number }
  | {
      kind: "attention";
      label: string;
      batch: number;
      heads: number;
      seq: number;
      headDim: number;
      count: number;
    }
  | { kind: "norm"; label: string; elements: number; count: number }
  | { kind: "pointwise"; label: string; elements: number; count: number; fused: boolean }
  | { kind: "sync"; label: string; calls: number }
  | { kind: "element"; label: string; elements: number; count: number };

export type Analysis = {
  archetype: Archetype;
  title: string;
  summary: string;
  language: Language;
  architecture: string;
  batch: number | null;
  seq: number | null;
  hidden: number | null;
  heads: number | null;
  dtype: DType;
  training: boolean;
  params: number;
  ops: WorkOp[];
  existing: string[];
  issues: string[];
  hasSync: boolean;
  hasCompile: boolean;
  hasAutocast: boolean;
  hasSdpa: boolean;
  hasTf32: boolean;
  hasCheckpoint: boolean;
  loopIters: number;
};

export type Tile = {
  bm: number;
  bn: number;
  bk: number;
  warps: number;
  stages: number;
};

export type ExecMode = {
  id: StrategyId;
  name: string;
  compiled: boolean;
  dtype: DType;
  flash: boolean;
  tritonGemm: boolean;
  tritonNorm: boolean;
  dropSync: boolean;
  checkpoint: boolean;
  vectorWidth: 1 | 4;
  tile: Tile | null;
};

export type Bench = {
  medianUs: number;
  p95Us: number;
  itersPerSec: number;
  vramMb: number;
  achievedTflops: number;
  bound: Bound;
  speedup: number;
  computeUs: number;
  memoryUs: number;
  launchUs: number;
  syncUs: number;
};

export type Check = {
  numerical: "pass" | "approximate" | "fail";
  rtol: string;
  gradients: "matched" | "scaled" | "not-applicable";
  memorySafe: boolean;
  notes: string[];
};

export type Candidate = {
  id: StrategyId;
  name: string;
  summary: string;
  techniques: string[];
  code: string;
  language: "python" | "cuda";
  bench: Bench;
  check: Check;
  eligible: boolean;
  rejectReason?: string;
  simplicity: number;
};

export type Bottleneck = {
  name: string;
  share: number;
  kind: Bound | "data";
  detail: string;
};

export type AgentStep = {
  title: string;
  detail: string;
  decision: string;
  kept: boolean;
  medianUs: number;
  speedup: number;
};

export type RunResult = {
  id: string;
  createdAt: number;
  example: boolean;
  gpuId: string;
  sampleId: string | null;
  instruction: string;
  code: string;
  objective: Objective;
  precisionPolicy: PrecisionPolicy;
  memoryBudgetGb: number;
  analysis: Analysis;
  candidates: Candidate[];
  winnerId: StrategyId;
  bottlenecks: Bottleneck[];
  explanation: string[];
  agentLog?: AgentStep[];
};

export type GpuSpec = {
  id: string;
  name: string;
  vendor: string;
  arch: string;
  chip: string;
  cc: string;
  sms: number;
  cudaCores: number;
  tensorCores: number;
  vramGb: number;
  bandwidthGBs: number;
  fp32: number;
  tf32: number;
  bf16: number;
  fp16: number;
  fp8: number;
  l2Mb: number;
  memType: string;
  tdpW: number;
  interconnect: string;
  linkGBs: number;
  aliases: string[];
};

export type LlmModel = {
  id: string;
  name: string;
  paramsB: number;
  layers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  headDim: number;
};

export type LlmPrec = "bf16" | "fp16" | "fp8" | "int8" | "int4";

export type LlmRow = {
  prec: LlmPrec;
  label: string;
  quality: number;
  weightGb: number;
  kvGb: number;
  totalGb: number;
  fits: boolean;
  native: boolean;
  ttftMs: number;
  tpotMs: number;
  toksPerSec: number;
  concurrency: number;
  note: string;
};

export type LlmReport = {
  model: LlmModel;
  gpu: GpuSpec;
  rows: LlmRow[];
  winner: LlmPrec | null;
  summary: string;
};

export type ParallelStrategy = "dp" | "tp" | "pp" | "hybrid";

export type ParallelRow = {
  id: ParallelStrategy;
  name: string;
  fits: boolean;
  stepMs: number;
  commMs: number;
  bubble: number;
  throughput: number;
  detail: string;
};

export type ParallelReport = {
  rows: ParallelRow[];
  winner: ParallelStrategy | null;
  summary: string;
};
