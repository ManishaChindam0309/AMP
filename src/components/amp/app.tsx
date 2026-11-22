import { useEffect, useState } from "react";
import {
  Activity,
  Boxes,
  ChartColumn,
  Cpu,
  LayoutDashboard,
  Network,
  Waypoints,
  Workflow,
  type LucideIcon,
} from "lucide-react";
import { GPUS, getGpu } from "@/lib/amp/gpus";
import { hydrateAmp, useAmp } from "@/lib/amp/store";
import type { ModuleId } from "@/lib/amp/types";
import { cn } from "@/lib/utils";
import { AgentLab } from "@/components/amp/agent";
import { BenchCenter } from "@/components/amp/benches";
import { KernelLab } from "@/components/amp/kernels";
import { LlmLab } from "@/components/amp/llm-lab";
import { MultiLab } from "@/components/amp/multi-gpu";
import { Optimizer } from "@/components/amp/optimizer";
import { Overview } from "@/components/amp/overview";
import { Profiler } from "@/components/amp/profiler";

const NAV: { id: ModuleId; label: string; icon: LucideIcon }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "optimizer", label: "Optimizer", icon: Cpu },
  { id: "profiler", label: "Profiler", icon: Activity },
  { id: "agent", label: "Agent", icon: Workflow },
  { id: "kernels", label: "Kernel lab", icon: Boxes },
  { id: "llm", label: "LLM inference", icon: Network },
  { id: "multigpu", label: "Multi-GPU", icon: Waypoints },
  { id: "benchmarks", label: "Benchmarks", icon: ChartColumn },
];

function Mark() {
  return (
    <svg viewBox="0 0 32 32" className="size-8 shrink-0" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--color-primary)" />
      <path d="M8 23 L16 8 L24 23" fill="none" stroke="var(--color-primary-fg)" strokeWidth="2.4" />
      <path d="M11.2 17.6 H20.8" stroke="var(--color-primary-fg)" strokeWidth="2.4" />
    </svg>
  );
}

export function AmpApp() {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    hydrateAmp();
    setReady(true);
  }, []);
  if (!ready) {
    return (
      <div className="flex h-dvh flex-col bg-bg text-fg">
        <header className="flex h-14 items-center gap-3 border-b border-border px-4">
          <Mark />
          <div>
            <p className="text-sm font-semibold">AMP</p>
            <p className="text-xs text-muted">Autonomous GPU optimization</p>
          </div>
        </header>
        <p className="p-6 text-sm text-muted">Loading the lab…</p>
      </div>
    );
  }
  return <Shell />;
}

function Shell() {
  const module = useAmp((s) => s.module);
  const setModule = useAmp((s) => s.setModule);
  const gpuId = useAmp((s) => s.gpuId);
  const setGpu = useAmp((s) => s.setGpu);
  const gpu = getGpu(gpuId);

  return (
    <div className="flex h-dvh flex-col bg-bg text-fg">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border px-3 sm:px-4">
        <Mark />
        <div className="min-w-0">
          <p className="text-sm font-semibold leading-tight">AMP</p>
          <p className="truncate text-xs text-muted">Autonomous GPU optimization</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <label className="sr-only" htmlFor="gpu-select">
            Target GPU
          </label>
          <select
            id="gpu-select"
            className="h-10 max-w-44 rounded-md border border-border bg-surface px-2 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-primary sm:max-w-none"
            value={gpuId}
            onChange={(event) => setGpu(event.target.value)}
          >
            {GPUS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} · {item.vramGb} GB
              </option>
            ))}
          </select>
        </div>
      </header>
      <div className="flex min-h-0 flex-1">
        <nav className="hidden w-52 shrink-0 flex-col gap-1 border-r border-border p-3 md:flex" aria-label="Modules">
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = module === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setModule(item.id)}
                className={cn(
                  "flex h-11 items-center gap-2 rounded-md px-3 text-left text-sm",
                  active ? "bg-surface-2 text-fg" : "text-muted hover:bg-surface hover:text-fg",
                )}
              >
                <Icon className={cn("size-4", active && "text-primary")} />
                {item.label}
              </button>
            );
          })}
          <p className="mt-auto px-3 pt-4 font-mono text-xs leading-5 text-muted">
            {gpu.arch} · cc {gpu.cc}
            <br />
            {gpu.bandwidthGBs} GB/s · {gpu.fp32} TFLOP/s
          </p>
        </nav>
        <div className="flex min-w-0 flex-1 flex-col">
          <nav className="flex gap-1 overflow-x-auto border-b border-border px-2 py-2 md:hidden" aria-label="Modules">
            {NAV.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setModule(item.id)}
                className={cn(
                  "h-11 shrink-0 rounded-md px-3 text-sm",
                  module === item.id ? "bg-primary text-primary-fg" : "bg-surface text-muted",
                )}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <main className="min-h-0 flex-1 overflow-auto p-4 sm:p-6">
            {module === "overview" && <Overview />}
            {module === "optimizer" && <Optimizer />}
            {module === "profiler" && <Profiler />}
            {module === "agent" && <AgentLab />}
            {module === "kernels" && <KernelLab />}
            {module === "llm" && <LlmLab />}
            {module === "multigpu" && <MultiLab />}
            {module === "benchmarks" && <BenchCenter />}
          </main>
        </div>
      </div>
    </div>
  );
}
