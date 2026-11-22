import { analyze } from "@/lib/amp/analyze";
import { matchGpu } from "@/lib/amp/gpus";
import { formatGb, formatSpeedup, formatTime } from "@/lib/amp/format";
import { agentModes, buildModes, pickWinner, scoreAll, scoreMode, TILES, tileEff } from "@/lib/amp/model";
import type {
  AgentStep,
  Analysis,
  Bottleneck,
  Candidate,
  GpuSpec,
  Objective,
  PrecisionPolicy,
  RunResult,
} from "@/lib/amp/types";

export const PROTOCOL = {
  warmup: 10,
  iterations: 50,
  statistic: "median",
  tail: "p95",
};

export type RunInput = {
  code: string;
  gpu: GpuSpec;
  objective: Objective;
  precisionPolicy: PrecisionPolicy;
  memoryBudgetGb: number;
  instruction: string;
  sampleId: string | null;
  id?: string;
  createdAt?: number;
  example?: boolean;
  agent?: boolean;
  budget?: number;
};

export function interpretInstruction(
  instruction: string,
  current: {
    gpuId: string;
    objective: Objective;
    precisionPolicy: PrecisionPolicy;
    memoryBudgetGb: number;
  },
): {
  gpuId: string;
  objective: Objective;
  precisionPolicy: PrecisionPolicy;
  memoryBudgetGb: number;
  notes: string[];
} {
  const notes: string[] = [];
  let { gpuId, objective, precisionPolicy, memoryBudgetGb } = current;
  const gpu = matchGpu(instruction);
  if (gpu) {
    gpuId = gpu;
    notes.push(`Target device read as ${gpu}.`);
  }
  if (/throughput|tokens per|tok\/s|as fast as possible|maximum speed/.test(instruction.toLowerCase())) {
    objective = "throughput";
    notes.push("Objective set to throughput.");
  } else if (/latency|time to first|ttft|response time/.test(instruction.toLowerCase())) {
    objective = "latency";
    notes.push("Objective set to latency.");
  }
  if (/exact|fp32 only|full precision|keep fp32|within fp32|do not quant|don't quant/.test(instruction.toLowerCase())) {
    precisionPolicy = "exact";
    notes.push("Precision policy set to exact FP32.");
  } else if (/max(?:imum)? speed|quantize|int4|fp8|as fast/.test(instruction.toLowerCase())) {
    precisionPolicy = "max-speed";
    notes.push("Precision policy set to maximum speed.");
  } else if (/bf16|acceptable/.test(instruction.toLowerCase())) {
    precisionPolicy = "auto";
    notes.push("bf16 is allowed.");
  }
  const mem = instruction.match(/(\d+(?:\.\d+)?)\s*gb/i);
  if (mem && /mem|vram|under|below|budget|keep/.test(instruction.toLowerCase())) {
    memoryBudgetGb = Number(mem[1]);
    notes.push(`Memory budget set to ${memoryBudgetGb} GB.`);
  }
  return { gpuId, objective, precisionPolicy, memoryBudgetGb, notes };
}

function bottlenecks(analysis: Analysis, baseline: Candidate): Bottleneck[] {
  const bench = baseline.bench;
  const compute = bench.computeUs >= bench.memoryUs ? bench.computeUs : 0;
  const memory = bench.memoryUs > bench.computeUs ? bench.memoryUs : 0;
  const total = Math.max(1, compute + memory + bench.launchUs + bench.syncUs);
  const rows: Bottleneck[] = [
    {
      name: "Compute",
      share: compute / total,
      kind: "compute",
      detail:
        compute > 0
          ? "Math intensity is above the ridge point. Precision is the lever."
          : "Compute is hidden behind memory or launch.",
    },
    {
      name: "Memory",
      share: memory / total,
      kind: "memory",
      detail:
        analysis.ops.some((op) => op.kind === "attention") && !analysis.hasSdpa
          ? "The score matrix dominates traffic. Fusion removes it."
          : "The working set is streaming. Fewer passes help.",
    },
    {
      name: "Launch",
      share: bench.launchUs / total,
      kind: "launch",
      detail:
        bench.launchUs / total > 0.25
          ? "The GPU is waiting on the CPU to launch small kernels."
          : "Launch cost is present, but it is not the limiter.",
    },
    {
      name: "Host sync",
      share: bench.syncUs / total,
      kind: "sync",
      detail: analysis.hasSync
        ? "A device-to-host read flushes the queue."
        : "No host sync on the timed path.",
    },
  ];
  return rows.filter((row) => row.share > 0.02);
}

function explain(winner: Candidate, candidates: Candidate[], gpu: GpuSpec, analysis: Analysis): string[] {
  const baseline = candidates[0];
  const lines: string[] = [];
  if (winner.id === "baseline") {
    lines.push(`${gpu.name} does not get faster from the rewrites AMP tried. The submitted code stays.`);
  } else {
    lines.push(
      `${winner.name} is ${formatSpeedup(winner.bench.speedup)} the baseline (${formatTime(winner.bench.medianUs)} vs ${formatTime(baseline.bench.medianUs)}).`,
    );
  }
  if (!baseline.eligible) {
    lines.push(
      `The baseline needs ${formatGb(baseline.bench.vramMb)} and does not fit in ${gpu.vramGb} GB, so it cannot win.`,
    );
  }
  lines.push(
    winner.check.numerical === "pass"
      ? `Numerical gate passed at rtol ${winner.check.rtol}.`
      : `Numerical gate is approximate at rtol ${winner.check.rtol}. ${winner.check.notes[0] ?? ""}`.trim(),
  );
  if (winner.bench.bound === "launch") lines.push("The winner is still launch-limited. Further kernel fusion has little left to give.");
  if (winner.bench.bound === "memory") lines.push("The winner is bandwidth-bound on this device.");
  if (winner.bench.bound === "compute") lines.push(`Achieved about ${winner.bench.achievedTflops.toFixed(2)} TFLOP/s of catalog peak.`);
  if (analysis.issues[0]) lines.push(analysis.issues[0]);
  const rejected = candidates.filter((c) => !c.eligible && c.id !== winner.id);
  if (rejected.length) {
    lines.push(`${rejected.length} candidate${rejected.length === 1 ? "" : "s"} failed a gate and ${rejected.length === 1 ? "was" : "were"} excluded.`);
  }
  return lines;
}

export function runOptimization(input: RunInput): RunResult {
  const analysis = analyze(input.code);
  const modes = input.agent
    ? agentModes(analysis, input.gpu, input.precisionPolicy).slice(0, Math.max(3, Math.min(8, input.budget ?? 6)))
    : buildModes(analysis, input.gpu, input.precisionPolicy);
  const candidates = scoreAll(
    analysis,
    input.gpu,
    modes,
    input.precisionPolicy,
    input.memoryBudgetGb,
    input.objective,
  );
  const winner = pickWinner(candidates, input.objective);
  const agentLog = input.agent ? buildLog(candidates, input.budget ?? candidates.length, winner.id) : undefined;
  return {
    id: input.id ?? `run-${input.createdAt ?? 0}`,
    createdAt: input.createdAt ?? 0,
    example: Boolean(input.example),
    gpuId: input.gpu.id,
    sampleId: input.sampleId,
    instruction: input.instruction,
    code: input.code,
    objective: input.objective,
    precisionPolicy: input.precisionPolicy,
    memoryBudgetGb: input.memoryBudgetGb,
    analysis,
    candidates,
    winnerId: winner.id,
    bottlenecks: bottlenecks(analysis, candidates[0]),
    explanation: explain(winner, candidates, input.gpu, analysis),
    agentLog,
  };
}

function buildLog(candidates: Candidate[], budget: number, winnerId: Candidate["id"]): AgentStep[] {
  const winner = candidates.find((candidate) => candidate.id === winnerId) ?? candidates[0];
  const log: AgentStep[] = [];
  for (const candidate of candidates) {
    if (log.length >= budget) break;
    const isWinner = candidate.id === winner.id;
    const faster = candidate.bench.medianUs < winner.bench.medianUs * 0.97;
    let decision: string;
    if (!candidate.eligible) {
      decision = `Reject. ${candidate.rejectReason ?? "Failed a gate."}`;
    } else if (candidate.id === candidates[0]?.id && isWinner) {
      decision = "The submitted code holds. Nothing eligible was faster by enough to replace it.";
    } else if (candidate.id === candidates[0]?.id) {
      decision = `Reference established at ${formatTime(candidate.bench.medianUs)}.`;
    } else if (isWinner) {
      decision = `Keep. ${formatSpeedup(candidate.bench.speedup)} versus the submitted code.`;
    } else if (faster) {
      decision = "Slightly faster, but inside 3%. The simpler candidate stays.";
    } else {
      decision = "No meaningful gain over the current selection.";
    }
    log.push({
      title: candidate.name,
      detail: candidate.summary,
      decision,
      kept: isWinner,
      medianUs: candidate.bench.medianUs,
      speedup: candidate.bench.speedup,
    });
  }
  return log;
}

export function autotuneRows(analysis: Analysis, gpu: GpuSpec, policy: PrecisionPolicy) {
  const dtype = policy === "exact" || gpu.bf16 <= 0 ? "fp32" : "bf16";
  const seed = scoreMode(
    analysis,
    gpu,
    {
      id: "triton-fp32",
      name: "tile",
      compiled: false,
      dtype,
      flash: analysis.ops.some((op) => op.kind === "attention"),
      tritonGemm: analysis.archetype === "gemm" || analysis.archetype === "generic",
      tritonNorm: analysis.archetype !== "gemm",
      dropSync: true,
      checkpoint: false,
      vectorWidth: 1,
      tile: TILES[0],
    },
    policy,
    gpu.vramGb,
    null,
  );
  return TILES.map((tile) => {
    const scored = scoreMode(
      analysis,
      gpu,
      {
        id: "triton-fast",
        name: `${tile.bm}×${tile.bn}×${tile.bk}`,
        compiled: false,
        dtype,
        flash: analysis.ops.some((op) => op.kind === "attention"),
        tritonGemm: analysis.archetype === "gemm" || analysis.archetype === "transformer",
        tritonNorm: analysis.archetype === "norm" || analysis.archetype === "transformer",
        dropSync: true,
        checkpoint: false,
        vectorWidth: analysis.archetype === "cuda" ? 4 : 1,
        tile,
      },
      policy,
      gpu.vramGb,
      seed.bench.medianUs,
    );
    const smem = (tile.bm * tile.bk + tile.bk * tile.bn) * tile.stages * 2;
    return {
      tile,
      eff: tileEff(gpu, tile),
      smem,
      medianUs: scored.bench.medianUs,
      bound: scored.bench.bound,
    };
  }).sort((a, b) => a.medianUs - b.medianUs);
}
