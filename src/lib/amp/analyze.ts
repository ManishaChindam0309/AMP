import type { Analysis, Archetype, DType, Language, WorkOp } from "@/lib/amp/types";

function firstNumber(code: string, re: RegExp): number | null {
  const match = code.match(re);
  return match ? Number(match[1]) : null;
}

export function analyze(code: string): Analysis {
  const src = code.replace(/\r\n/g, "\n");
  const lower = src.toLowerCase();
  const isCuda = /__global__|#include\s*<cuda|threadIdx|blockIdx/.test(src);
  const isTriton = /@triton\.jit|import triton/.test(src);
  const language: Language =
    isCuda && /torch/.test(src) ? "mixed" : isCuda ? "cuda" : isTriton ? "triton" : "pytorch";

  const hasCompile = /torch\.compile/.test(src);
  const hasAutocast = /autocast|bfloat16|float16|\bbf16\b|\bfp16\b/.test(lower);
  const hasSdpa = /scaled_dot_product_attention|flash_attn|FlashAttention/.test(src);
  const hasTf32 = /allow_tf32|matmul_precision|\btf32\b/.test(lower);
  const hasCheckpoint = /checkpoint\(/.test(src);
  const itemCalls = (src.match(/\.item\(\)/g) ?? []).length;
  const cpuCalls = (src.match(/\.cpu\(\)/g) ?? []).length;
  const syncInLoop = /for[\s\S]{0,600}\.(item|cpu)\(/.test(src);
  const hasSync = itemCalls + cpuCalls > 0;
  const rangeMatch = src.match(/range\(\s*(\d+)\s*\)/);
  const loopIters = rangeMatch ? Number(rangeMatch[1]) : 1;

  const dtype: DType = /float8|fp8|e4m3|e5m2/.test(lower)
    ? "fp8"
    : /bfloat16|\bbf16\b/.test(lower)
      ? "bf16"
      : /float16|\bfp16\b/.test(lower)
        ? "fp16"
        : "fp32";

  const training = /backward\(|\.step\(\)|optimizer|opt\.step|loss/.test(src);
  const triple = src.match(/\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/);
  const heads = firstNumber(src, /heads\s*[:=]\s*(\d+)/i);

  const linears: { inn: number; out: number }[] = [];
  for (const match of src.matchAll(/nn\.Linear\(\s*(\d+)\s*,\s*(\d+)\s*\)/g)) {
    linears.push({ inn: Number(match[1]), out: Number(match[2]) });
  }
  const params = linears.reduce((sum, lin) => sum + lin.inn * lin.out + lin.out, 0);

  const matrices: number[][] = [];
  for (const match of src.matchAll(
    /torch\.(?:randn|zeros|empty|ones)\(\s*(\d+)\s*,\s*(\d+)/g,
  )) {
    matrices.push([Number(match[1]), Number(match[2])]);
  }

  const shift = src.match(/1\s*<<\s*(\d+)/);
  const namedN = firstNumber(src, /\bn\s*=\s*(\d+)/);
  const softmax = /softmax/.test(src);
  const matmul = /matmul|(?<![\w.])@(?![\w])|nn\.Linear|F\.linear/.test(src);
  const hasAttention =
    hasSdpa || (softmax && matmul) || /MultiheadAttention/.test(src) || linears.some((l) => l.out === l.inn * 3);
  const hasNorm = /LayerNorm|RMSNorm|rms_norm|layer_norm/.test(src);

  let archetype: Archetype = "generic";
  if (isCuda && !/import torch|torch\./.test(src)) archetype = "cuda";
  else if (hasAttention && (hasNorm || linears.length >= 2)) archetype = "transformer";
  else if (hasNorm && !matmul) archetype = "norm";
  else if (matmul || matrices.length > 0) archetype = "gemm";

  const ops: WorkOp[] = [];
  let batch: number | null = null;
  let seq: number | null = null;
  let hidden: number | null = null;
  let architecture = "Unclassified workload";
  let title = "Workload";

  if (archetype === "transformer") {
    batch = triple ? Number(triple[1]) : 8;
    seq = triple ? Number(triple[2]) : 2048;
    hidden = triple ? Number(triple[3]) : (linears[0]?.inn ?? 1024);
    const headCount = heads ?? 16;
    const headDim = Math.max(1, Math.round(hidden / headCount));
    const tokens = batch * seq;
    title = "Transformer block";
    architecture = training ? "Pre-norm transformer block, training step" : "Pre-norm transformer block";
    ops.push({
      kind: "attention",
      label: hasSdpa ? "SDPA attention" : "Manual attention",
      batch,
      heads: headCount,
      seq,
      headDim,
      count: 1,
    });
    const gems = linears.length
      ? linears.map((lin) => ({
          label: `Linear ${lin.inn}→${lin.out}`,
          m: tokens,
          n: lin.out,
          k: lin.inn,
        }))
      : [
          { label: "QKV", m: tokens, n: hidden * 3, k: hidden },
          { label: "Proj", m: tokens, n: hidden, k: hidden },
          { label: "MLP up", m: tokens, n: hidden * 4, k: hidden },
          { label: "MLP down", m: tokens, n: hidden, k: hidden * 4 },
        ];
    for (const gem of gems) {
      ops.push({ kind: "gemm", ...gem, count: 1 });
    }
    ops.push({ kind: "norm", label: "Norms", elements: tokens * hidden, count: 2 });
    ops.push({
      kind: "pointwise",
      label: "GELU",
      elements: tokens * hidden * 4,
      count: 1,
      fused: hasCompile,
    });
  } else if (archetype === "norm") {
    const rows = matrices[0]?.[0] ?? 65536;
    const cols = matrices[0]?.[1] ?? 4096;
    hidden = cols;
    batch = rows;
    title = "Normalization";
    architecture = "Memory-bound normalization";
    ops.push({ kind: "norm", label: "RMSNorm", elements: rows * cols, count: 1 });
  } else if (archetype === "gemm") {
    const left = matrices[0] ?? [1024, 1024];
    const right = matrices[1] ?? [left[1], left[0]];
    const m = left[0];
    const k = left[1];
    const n = right[0] === k ? right[1] : right[0];
    const tiny = m * n * k < 5_000_000;
    const count = tiny && loopIters > 1 ? loopIters : 1;
    title = count > 1 ? "Tiny GEMM loop" : "Matrix multiply";
    architecture = count > 1 ? "Launch-bound matmul loop" : "Dense matrix multiply";
    batch = m;
    hidden = k;
    ops.push({ kind: "gemm", label: `GEMM ${m}×${k}×${n}`, m, n, k, count });
  } else if (archetype === "cuda") {
    const n = shift ? 2 ** Number(shift[1]) : (namedN ?? 1_048_576);
    title = "CUDA kernel";
    architecture = "Elementwise CUDA kernel";
    ops.push({ kind: "element", label: "Vector add", elements: n, count: 1 });
  } else {
    title = "Generic workload";
    architecture = "Unrecognized pattern, modeled as a 1024² GEMM";
    ops.push({ kind: "gemm", label: "Assumed GEMM", m: 1024, n: 1024, k: 1024, count: 1 });
  }

  if (hasSync) {
    const calls = Math.max(1, itemCalls + cpuCalls) * (syncInLoop ? Math.max(loopIters, 1) : 1);
    ops.push({ kind: "sync", label: "Host synchronization", calls });
  }

  const existing: string[] = [];
  if (hasCompile) existing.push("torch.compile");
  if (hasAutocast) existing.push(dtype === "fp16" ? "fp16 autocast" : "bf16 autocast");
  if (hasSdpa) existing.push("SDPA / FlashAttention");
  if (hasTf32) existing.push("TF32 matmul");
  if (hasCheckpoint) existing.push("gradient checkpointing");
  if (!existing.length) existing.push("none");

  const issues: string[] = [];
  if (archetype === "transformer" && !hasSdpa) {
    issues.push("Attention writes the full score matrix.");
  }
  if (hasSync) issues.push("Host sync via .item() or .cpu() is on the hot path.");
  if (!hasCompile && archetype !== "cuda") issues.push("Eager PyTorch dispatch. torch.compile is unused.");
  if (dtype === "fp32" && !hasTf32 && archetype !== "cuda" && archetype !== "norm") {
    issues.push("FP32 math leaves tensor cores idle.");
  }
  if (archetype === "generic") {
    issues.push("No concrete shapes found. A 1024² GEMM was assumed.");
  }
  if (archetype === "cuda") {
    issues.push("Scalar loads. The kernel is not vectorized.");
  }

  const summary = `${architecture}. ${training ? "Training" : "Inference"} · ${dtype}. ${
    existing[0] === "none" ? "No optimizations applied yet." : `Already using ${existing.join(", ")}.`
  }`;

  return {
    archetype,
    title,
    summary,
    language,
    architecture,
    batch,
    seq,
    hidden,
    heads,
    dtype,
    training,
    params,
    ops,
    existing,
    issues,
    hasSync,
    hasCompile,
    hasAutocast,
    hasSdpa,
    hasTf32,
    hasCheckpoint,
    loopIters,
  };
}
