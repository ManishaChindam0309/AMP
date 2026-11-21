import type { GpuSpec, LlmModel, LlmPrec, LlmReport, LlmRow, Objective, PrecisionPolicy } from "@/lib/amp/types";

export const LLM_MODELS: LlmModel[] = [
  { id: "llama32-1b", name: "Llama 3.2 1B", paramsB: 1.24, layers: 16, hidden: 2048, heads: 32, kvHeads: 8, headDim: 64 },
  { id: "qwen25-7b", name: "Qwen2.5 7B", paramsB: 7.62, layers: 28, hidden: 3584, heads: 28, kvHeads: 4, headDim: 128 },
  { id: "llama3-8b", name: "Llama 3 8B", paramsB: 8.03, layers: 32, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128 },
  { id: "gemma2-9b", name: "Gemma 2 9B", paramsB: 9.24, layers: 42, hidden: 3584, heads: 16, kvHeads: 8, headDim: 256 },
  { id: "mistral-7b", name: "Mistral 7B", paramsB: 7.25, layers: 32, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128 },
  { id: "llama3-70b", name: "Llama 3 70B", paramsB: 70.6, layers: 80, hidden: 8192, heads: 64, kvHeads: 8, headDim: 128 },
];

const PRECS: { id: LlmPrec; label: string; bytes: number; quality: number }[] = [
  { id: "bf16", label: "BF16", bytes: 2, quality: 100 },
  { id: "fp16", label: "FP16", bytes: 2, quality: 100 },
  { id: "fp8", label: "FP8", bytes: 1, quality: 97 },
  { id: "int8", label: "INT8", bytes: 1, quality: 94 },
  { id: "int4", label: "INT4", bytes: 0.5, quality: 86 },
];

export function getLlmModel(id: string): LlmModel {
  return LLM_MODELS.find((model) => model.id === id) ?? LLM_MODELS[2];
}

export function reportLlm(input: {
  model: LlmModel;
  gpu: GpuSpec;
  batch: number;
  prompt: number;
  generated: number;
  budgetGb: number;
  continuous: boolean;
  speculative: boolean;
  objective: Objective;
  policy: PrecisionPolicy;
}): LlmReport {
  const params = input.model.paramsB * 1e9;
  const memEff = input.gpu.bandwidthGBs >= 1500 ? 0.7 : input.gpu.bandwidthGBs >= 400 ? 0.55 : 0.45;
  const budget = Math.min(input.budgetGb, input.gpu.vramGb);
  const rows: LlmRow[] = PRECS.map((prec) => {
    const native = prec.id !== "fp8" || input.gpu.fp8 > 0;
    const weightBytes = params * prec.bytes;
    const kvPerToken = 2 * input.model.layers * input.model.kvHeads * input.model.headDim * prec.bytes;
    const seq = input.prompt + input.generated;
    const kvBytes = kvPerToken * input.batch * seq;
    const overhead = 0.55e9;
    const total = weightBytes + kvBytes + overhead;
    const fits = total <= budget * 1e9 * 0.92;
    const peak =
      prec.id === "fp8" && input.gpu.fp8 > 0
        ? input.gpu.fp8
        : prec.id === "fp16"
          ? input.gpu.fp16 || input.gpu.fp32
          : input.gpu.bf16 || input.gpu.fp32;
    const dequant = prec.id === "int4" ? 0.55 : prec.id === "int8" ? 0.7 : native ? 1 : 0.35;
    const linear = 2 * params * input.batch * input.prompt;
    const attn = 4 * input.model.layers * input.batch * input.prompt * input.prompt * input.model.hidden;
    const prefillS = Math.max(
      (linear + attn) / (peak * 1e12 * 0.32 * dequant),
      (weightBytes + kvPerToken * input.batch * input.prompt) / (input.gpu.bandwidthGBs * 1e9 * memEff),
    );
    const free = budget * 1e9 * 0.92 - weightBytes - overhead;
    const kvPerSeq = kvPerToken * seq;
    const slots = Math.max(0, Math.floor(free / Math.max(kvPerSeq, 1)));
    const usedBatch = input.continuous ? Math.max(input.batch, Math.min(slots, Math.max(input.batch, 1) * 16)) : input.batch;
    const kvDecode = kvPerToken * usedBatch * (input.prompt + input.generated / 2);
    const tpotS = Math.max(
      (weightBytes * (prec.id === "int4" ? 1.08 : 1) + kvDecode) / (input.gpu.bandwidthGBs * 1e9 * memEff),
      (2 * params) / (peak * 1e12 * 0.45 * dequant),
    );
    let toks = fits && usedBatch > 0 ? usedBatch / tpotS : 0;
    let note = fits ? (native ? "Native math." : "Emulated. Bandwidth win, compute loss.") : "Does not fit.";
    if (fits && input.speculative && weightBytes * 1.2 + kvBytes < budget * 1e9 * 0.92) {
      toks *= 1.4;
      note = "Draft model assumed at 65% acceptance.";
    } else if (input.speculative && fits) {
      note = "No room for a draft model beside the weights.";
    }
    if (input.continuous && fits) note = `${note} ${slots} resident sequences.`.trim();
    return {
      prec: prec.id,
      label: prec.label,
      quality: prec.quality,
      weightGb: weightBytes / 1e9,
      kvGb: kvBytes / 1e9,
      totalGb: total / 1e9,
      fits,
      native,
      ttftMs: fits ? prefillS * 1e3 : 0,
      tpotMs: fits ? (tpotS / Math.max(usedBatch, 1)) * 1e3 * usedBatch : 0,
      toksPerSec: toks,
      concurrency: fits ? slots : 0,
      note,
    };
  });

  const floor = input.policy === "exact" ? 100 : input.policy === "max-speed" ? 0 : 90;
  const pool = rows.filter((row) => row.fits && row.quality >= floor);
  const winner = pool.sort((a, b) =>
    input.objective === "latency" ? a.ttftMs - b.ttftMs || b.toksPerSec - a.toksPerSec : b.toksPerSec - a.toksPerSec,
  )[0];

  const summary = winner
    ? `${winner.label} is the selection under a ${floor} quality floor. ${winner.toksPerSec.toFixed(1)} tok/s, first token ${winner.ttftMs.toFixed(0)} ms, KV cache ${winner.kvGb.toFixed(2)} GB.`
    : `${input.model.name} does not fit on ${input.gpu.name} inside ${budget} GB at the allowed quality.`;

  return { model: input.model, gpu: input.gpu, rows, winner: winner?.prec ?? null, summary };
}
