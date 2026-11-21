export type Sample = {
  id: string;
  name: string;
  blurb: string;
  language: "pytorch" | "cuda";
  instruction: string;
  code: string;
};

export const SAMPLES: Sample[] = [
  {
    id: "transformer",
    name: "Transformer block",
    blurb: "Manual attention, LayerNorm, MLP, and a host sync on every step.",
    language: "pytorch",
    instruction: "Make this training step faster on my RTX 2050. bf16 is acceptable.",
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """One pre-norm transformer block. x: [8, 2048, 1024], heads=16."""

    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(1024)
        self.qkv = nn.Linear(1024, 3072)
        self.proj = nn.Linear(1024, 1024)
        self.n2 = nn.LayerNorm(1024)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.heads = 16

    def forward(self, x):
        B, S, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).view(B, S, 3, self.heads, D // self.heads)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = (D // self.heads) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        x = x + self.proj(out)
        h = self.n2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

def train_step(model, x, opt):
    opt.zero_grad()
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    opt.step()
    return loss.item()
`,
  },
  {
    id: "tiny-gemm",
    name: "Tiny GEMM loop",
    blurb: "32×32 matmul pulled back to the host a thousand times.",
    language: "pytorch",
    instruction: "This is launch bound. Keep FP32 accuracy.",
    code: `import torch

def bench():
    # a, b: [32, 32] fp32
    a = torch.randn(32, 32, device="cuda")
    b = torch.randn(32, 32, device="cuda")
    acc = 0.0
    for _ in range(1000):
        c = a @ b
        acc += c[0, 0].item()
    return acc
`,
  },
  {
    id: "square-gemm",
    name: "Square GEMM",
    blurb: "A single 2048² fp32 matmul. Big enough for tensor cores to matter.",
    language: "pytorch",
    instruction: "Make this matrix multiplication faster on my RTX 2050.",
    code: `import torch

def gemm(a, b):
    # a: [2048, 2048], b: [2048, 2048], fp32
    return a @ b

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
c = gemm(a, b)
torch.cuda.synchronize()
`,
  },
  {
    id: "rmsnorm",
    name: "RMSNorm",
    blurb: "Three memory passes over a wide activation. Fusion is the whole game.",
    language: "pytorch",
    instruction: "Fuse this norm. Keep the output within fp32 tolerance.",
    code: `import torch

def rms_norm(x, weight, eps=1e-6):
    # x: [65536, 4096] fp32, weight: [4096]
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    return y * weight

x = torch.randn(65536, 4096, device="cuda")
weight = torch.ones(4096, device="cuda")
y = rms_norm(x, weight)
`,
  },
  {
    id: "cuda-add",
    name: "CUDA vector add",
    blurb: "A scalar grid over 16M floats. Bandwidth left on the table.",
    language: "cuda",
    instruction: "Vectorize this kernel. Exact fp32.",
    code: `#include <cuda_runtime.h>

// n = 1 << 24
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void launch(const float* a, const float* b, float* c, int n) {
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, c, n);
}
`,
  },
];

export function getSample(id: string | null): Sample {
  return SAMPLES.find((s) => s.id === id) ?? SAMPLES[0];
}
