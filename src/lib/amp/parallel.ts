import type { GpuSpec, LlmModel, ParallelReport, ParallelRow, ParallelStrategy } from "@/lib/amp/types";

export function reportParallel(input: {
  gpu: GpuSpec;
  model: LlmModel;
  count: number;
  training: boolean;
  batch: number;
  seq: number;
  micros: number;
}): ParallelReport {
  const n = Math.max(1, input.count);
  const params = input.model.paramsB * 1e9;
  const weightInfer = params * 2;
  const weightTrain = params * 14;
  const weights = input.training ? weightTrain : weightInfer;
  const kv = 2 * input.model.layers * input.model.kvHeads * input.model.headDim * input.batch * input.seq * 2;
  const link = input.gpu.linkGBs * 1e9;
  const vram = input.gpu.vramGb * 1e9 * 0.9;

  const computeMs = (batch: number) => {
    const linear = 2 * params * batch * input.seq * (input.training ? 3 : 1);
    const attn = 4 * input.model.layers * batch * input.seq * input.seq * input.model.hidden * (input.training ? 2 : 1);
    const peak = (input.gpu.bf16 || input.gpu.fp32) * 1e12 * 0.35;
    return ((linear + attn) / peak) * 1e3;
  };

  const full = computeMs(input.batch);

  const dp: ParallelRow = (() => {
    const fits = weights + kv / n < vram;
    const local = Math.max(1, input.batch / n);
    const compute = computeMs(local);
    const grad = input.training ? (2 * params * 2 * ((n - 1) / n)) / link * 1e3 : 0;
    const step = compute + grad;
    const throughput = fits ? (input.training ? (local * n) / (step / 1e3) : (n * local) / (step / 1e3)) : 0;
    return {
      id: "dp",
      name: input.training ? "Data parallel" : "Replicas",
      fits,
      stepMs: step,
      commMs: grad,
      bubble: 0,
      throughput,
      detail: input.training
        ? "Full weights on every device. One all-reduce of the gradients per step."
        : "Independent copies. No communication on the decode path. The model must fit once.",
    };
  })();

  const tp: ParallelRow = (() => {
    const headsOk = input.model.heads % n === 0;
    const kvShard = input.model.kvHeads % n === 0 ? kv / n : kv;
    const fits = headsOk && weights / n + kvShard < vram;
    const compute = (full / n) * 1.08;
    const collectives = input.model.layers * (input.training ? 4 : 2);
    const commBytes = collectives * input.batch * input.seq * input.model.hidden * 2 * ((n - 1) / n);
    const comm = n === 1 ? 0 : (commBytes / link) * 1e3;
    const step = compute + comm;
    return {
      id: "tp",
      name: "Tensor parallel",
      fits,
      stepMs: step,
      commMs: comm,
      bubble: 0,
      throughput: fits ? input.batch / (step / 1e3) : 0,
      detail: headsOk
        ? "Weights and attention heads are sharded. Collectives run every layer."
        : "Head count does not divide the device count, so this shard is invalid.",
    };
  })();

  const pp: ParallelRow = (() => {
    const fits = weights / n + kv < vram;
    const bubble = n === 1 ? 0 : (n - 1) / Math.max(1, input.micros);
    const compute = full / n;
    const comm = n === 1 ? 0 : ((input.batch * input.seq * input.model.hidden * 2) / link) * 1e3 * 2;
    const step = compute * (1 + bubble) + comm;
    return {
      id: "pp",
      name: "Pipeline parallel",
      fits,
      stepMs: step,
      commMs: comm,
      bubble,
      throughput: fits ? input.batch / (step / 1e3) : 0,
      detail: `Layers are staged. Bubble is ${(bubble * 100).toFixed(0)}% at ${input.micros} micro-batches.`,
    };
  })();

  const hybrid: ParallelRow = (() => {
    if (n < 4) {
      return {
        id: "hybrid",
        name: "TP 2 + pipeline",
        fits: false,
        stepMs: 0,
        commMs: 0,
        bubble: 0,
        throughput: 0,
        detail: "Needs at least 4 devices. Two-way tensor parallel, the rest as pipeline stages.",
      };
    }
    const tpWidth = 2;
    const stages = n / tpWidth;
    const headsOk = input.model.heads % tpWidth === 0;
    const fits = headsOk && weights / n + kv / tpWidth < vram;
    const bubble = (stages - 1) / Math.max(1, input.micros);
    const compute = full / n;
    const comm =
      ((input.model.layers * input.batch * input.seq * input.model.hidden * 2 * 0.5) / link) * 1e3 +
      ((input.batch * input.seq * input.model.hidden * 2) / link) * 1e3;
    const step = compute * (1 + bubble) + comm;
    return {
      id: "hybrid",
      name: "TP 2 + pipeline",
      fits,
      stepMs: step,
      commMs: comm,
      bubble,
      throughput: fits ? input.batch / (step / 1e3) : 0,
      detail: `${tpWidth}-way tensor parallel across ${stages} pipeline stages.`,
    };
  })();

  const rows = [dp, tp, pp, hybrid];
  const fitting = rows.filter((row) => row.fits);
  const winner = fitting.sort((a, b) => a.stepMs - b.stepMs)[0] ?? null;
  const summary = winner
    ? `${winner.name} has the lowest step time on ${n}× ${input.gpu.name} over ${input.gpu.interconnect}.`
    : `${input.model.name} does not fit on ${n}× ${input.gpu.name} in any of these placements.`;

  return { rows, winner: (winner?.id as ParallelStrategy | undefined) ?? null, summary };
}
