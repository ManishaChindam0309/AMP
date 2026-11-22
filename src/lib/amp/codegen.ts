import type { Analysis, ExecMode, GpuSpec, Tile } from "@/lib/amp/types";

function tileOf(mode: ExecMode): Tile {
  return mode.tile ?? { bm: 64, bn: 64, bk: 32, warps: 4, stages: 2 };
}

export function renderCandidate(analysis: Analysis, mode: ExecMode, gpu: GpuSpec): string {
  if (analysis.archetype === "cuda") return renderCuda(analysis, mode);
  if (analysis.archetype === "norm") return renderNorm(analysis, mode);
  if (analysis.archetype === "gemm") return renderGemm(analysis, mode, gpu);
  if (analysis.archetype === "transformer") return renderBlock(analysis, mode, gpu);
  return renderGeneric(analysis, mode, gpu);
}

function renderCuda(analysis: Analysis, mode: ExecMode): string {
  const elements = analysis.ops.find((op) => op.kind === "element");
  const n = elements && elements.kind === "element" ? elements.elements : 1 << 24;
  if (mode.vectorWidth === 4) {
    return `#include <cuda_runtime.h>

// AMP vectorized rewrite. n = ${n}
__global__ void add_vec4(const float4* __restrict__ a,
                         const float4* __restrict__ b,
                         float4* __restrict__ c,
                         int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n4; i += stride) {
        float4 x = a[i];
        float4 y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

void launch(const float* a, const float* b, float* c, int n) {
    int n4 = n >> 2;
    int threads = 256;
    int blocks = 2048;
    add_vec4<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(a),
        reinterpret_cast<const float4*>(b),
        reinterpret_cast<float4*>(c),
        n4);
}
`;
  }
  if (mode.compiled) {
    return `#include <cuda_runtime.h>

// AMP launch rewrite: grid-stride, 256 threads, enough blocks to fill ${analysis.archetype}.
__global__ void add_stride(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) c[i] = a[i] + b[i];
}

void launch(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = 1024;
    add_stride<<<blocks, threads>>>(a, b, c, n);
}
`;
  }
  return `#include <cuda_runtime.h>

// Baseline scalar kernel. n = ${n}
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void launch(const float* a, const float* b, float* c, int n) {
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, c, n);
}
`;
}

function renderNorm(analysis: Analysis, mode: ExecMode): string {
  const norm = analysis.ops.find((op) => op.kind === "norm");
  const elements = norm && norm.kind === "norm" ? norm.elements : 65536 * 4096;
  const cols = analysis.hidden ?? 4096;
  const rows = Math.max(1, Math.round(elements / cols));
  if (mode.tritonNorm) {
    const tile = tileOf(mode);
    return `import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, rows, cols, eps,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = cols_off < cols
    x = tl.load(x_ptr + row * cols + cols_off, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / cols
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols_off, mask=mask, other=0.0)
    y = (x * rstd).to(tl.float32) * w
    tl.store(y_ptr + row * cols + cols_off, y, mask=mask)

def rms_norm(x, weight, eps=1e-6):
    # x: [${rows}, ${cols}]
    y = torch.empty_like(x)
    rmsnorm_kernel[(x.shape[0],)](
        x, weight, y, x.shape[0], x.shape[1], eps, BLOCK=${Math.max(cols, tile.bm)}
    )
    return y
`;
  }
  if (mode.compiled) {
    return `import torch

@torch.compile(fullgraph=True)
def rms_norm(x, weight, eps=1e-6):
    # Fused by Inductor. x: [${rows}, ${cols}] ${mode.dtype}
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    return y * weight
`;
  }
  return `import torch

def rms_norm(x, weight, eps=1e-6):
    # Baseline, three kernels. x: [${rows}, ${cols}]
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    return y * weight
`;
}

function renderGemm(analysis: Analysis, mode: ExecMode, gpu: GpuSpec): string {
  const gemm = analysis.ops.find((op) => op.kind === "gemm");
  const m = gemm && gemm.kind === "gemm" ? gemm.m : 1024;
  const n = gemm && gemm.kind === "gemm" ? gemm.n : 1024;
  const k = gemm && gemm.kind === "gemm" ? gemm.k : 1024;
  const loops = gemm && gemm.kind === "gemm" ? gemm.count : 1;
  const tile = tileOf(mode);

  if (mode.id === "drop-sync") {
    return `import torch

def bench():
    # Same ${m}×${k} @ ${k}×${n}, host sync removed from the loop.
    a = torch.randn(${m}, ${k}, device="cuda")
    b = torch.randn(${k}, ${n}, device="cuda")
    acc = torch.zeros((), device="cuda")
    for _ in range(${loops}):
        c = a @ b
        acc = acc + c[0, 0]
    return acc.item()  # one sync, after the timed loop
`;
  }

  if (mode.tritonGemm) {
    return `import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k * BLOCK_K + offs_k)[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & ((k * BLOCK_K + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k * BLOCK_K + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=((k * BLOCK_K + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float32 if ${mode.dtype === "fp32" ? "True" : "False"} else tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def gemm(a, b):
    # Tuned for ${gpu.name}: BLOCK ${tile.bm}×${tile.bn}×${tile.bk}, ${tile.warps} warps, ${tile.stages} stages
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, ${tile.bm}), triton.cdiv(N, ${tile.bn}))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=${tile.bm}, BLOCK_N=${tile.bn}, BLOCK_K=${tile.bk},
        num_warps=${tile.warps}, num_stages=${tile.stages},
    )
    return c
`;
  }

  const precision =
    mode.dtype === "tf32"
      ? `torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
`
      : mode.dtype === "bf16" || mode.dtype === "fp16"
        ? ""
        : `torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
`;
  const body =
    mode.dtype === "bf16" || mode.dtype === "fp16"
      ? `    with torch.autocast("cuda", dtype=torch.${mode.dtype === "bf16" ? "bfloat16" : "float16"}):
        return a @ b
`
      : `    return a @ b
`;
  const wrapped = mode.compiled
    ? `@torch.compile(mode="max-autotune")
def gemm(a, b):
${body}`
    : `def gemm(a, b):
${body}`;
  return `import torch

${precision}${wrapped}
# a: [${m}, ${k}]  b: [${k}, ${n}]  on ${gpu.name}
`;
}

function renderBlock(analysis: Analysis, mode: ExecMode, gpu: GpuSpec): string {
  const batch = analysis.batch ?? 8;
  const seq = analysis.seq ?? 2048;
  const hidden = analysis.hidden ?? 1024;
  const heads = analysis.heads ?? 16;
  const dtypeLine =
    mode.dtype === "bf16" || mode.dtype === "fp16"
      ? `torch.${mode.dtype === "bf16" ? "bfloat16" : "float16"}`
      : null;
  const flash = mode.flash;
  const syncBack = mode.dropSync ? "return loss.detach()" : "return loss.item()";
  const tf32 =
    mode.dtype === "tf32"
      ? `\ntorch.backends.cuda.matmul.allow_tf32 = True\ntorch.set_float32_matmul_precision("high")\n`
      : "";
  const attention = flash
    ? `        qkv = self.qkv(h).view(B, S, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, D)`
    : `        qkv = self.qkv(h).view(B, S, 3, self.heads, D // self.heads)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * ((D // self.heads) ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)`;

  const tritonNorm = mode.tritonNorm
    ? `
@triton.jit
def fused_ln_kernel(x_ptr, w_ptr, b_ptr, y_ptr, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / cols
    var = tl.sum((x - mean) * (x - mean), axis=0) / cols
    y = (x - mean) * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * cols + offs, y * w + b, mask=mask)
`
    : "";

  const compileLine = mode.compiled ? `\nmodel = torch.compile(Block().cuda(), mode="max-autotune")\n` : `\nmodel = Block().cuda()\n`;
  const autocast = dtypeLine
    ? `    with torch.autocast("cuda", dtype=${dtypeLine}):\n        y = model(x)\n        loss = y.float().pow(2).mean()\n`
    : `    y = model(x)\n    loss = y.pow(2).mean()\n`;

  return `import torch
import torch.nn as nn
import torch.nn.functional as F
${mode.tritonNorm ? "import triton\nimport triton.language as tl\n" : ""}${tf32}${tritonNorm}
class Block(nn.Module):
    """AMP candidate for ${gpu.name}. x: [${batch}, ${seq}, ${hidden}], heads=${heads}."""

    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(${hidden})
        self.qkv = nn.Linear(${hidden}, ${hidden * 3})
        self.proj = nn.Linear(${hidden}, ${hidden})
        self.n2 = nn.LayerNorm(${hidden})
        self.fc1 = nn.Linear(${hidden}, ${hidden * 4})
        self.fc2 = nn.Linear(${hidden * 4}, ${hidden})
        self.heads = ${heads}

    def forward(self, x):
        B, S, D = x.shape
        h = self.n1(x)
${attention}
        x = x + self.proj(out)
        h = self.n2(x)
        x = x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))
        return x

def train_step(model, x, opt):
    opt.zero_grad(set_to_none=True)
${autocast}    loss.backward()
    opt.step()
    ${syncBack}
${compileLine}`;
}

function renderGeneric(analysis: Analysis, mode: ExecMode, gpu: GpuSpec): string {
  return `# AMP could not map this source onto a known pattern.
# Modeled as ${analysis.architecture} on ${gpu.name}.
# Candidate: ${mode.name}
# Re-insert shapes, for example: x: [8, 2048, 1024]

${mode.compiled ? "import torch\nmodel = torch.compile(model)\n" : ""}${
    mode.dtype === "bf16" ? "import torch\n# with torch.autocast('cuda', dtype=torch.bfloat16):\n" : ""
  }`;
}
