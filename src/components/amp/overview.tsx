import { getGpu } from "@/lib/amp/gpus";
import { formatGb, formatTime } from "@/lib/amp/format";
import { SAMPLES } from "@/lib/amp/samples";
import { useAmp } from "@/lib/amp/store";
import { Metric, PageHead, Panel, Speed, Tag } from "@/components/amp/bits";

export function Overview() {
  const gpu = getGpu(useAmp((s) => s.gpuId));
  const run = useAmp((s) => s.lastRun);
  const history = useAmp((s) => s.history);
  const loadSample = useAmp((s) => s.loadSample);
  const openRun = useAmp((s) => s.openRun);
  const setModule = useAmp((s) => s.setModule);
  const winner = run.candidates.find((c) => c.id === run.winnerId) ?? run.candidates[0];
  const precisions = [
    ["TF32", gpu.tf32 > 0],
    ["BF16", gpu.bf16 > 0],
    ["FP8", gpu.fp8 > 0],
  ] as const;

  return (
    <div>
      <PageHead
        kicker="DASHBOARD"
        title="What this device can carry"
        lede="AMP reads a workload, writes a few implementations, and keeps the fastest one that still passes the numerical gate."
      />
      <div className="grid gap-4 lg:grid-cols-5">
        <Panel className="p-4 lg:col-span-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs text-muted">Selected device</p>
              <h2 className="mt-1 text-xl font-semibold">
                {gpu.vendor} {gpu.name}
              </h2>
              <p className="mt-1 text-sm text-muted">
                {gpu.chip} · {gpu.arch} · compute capability {gpu.cc}
              </p>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {precisions.map(([name, on]) => (
                <Tag key={name} tone={on ? "good" : "neutral"}>
                  {name} {on ? "yes" : "no"}
                </Tag>
              ))}
            </div>
          </div>
          <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Metric label="Memory" value={`${gpu.vramGb} GB`} hint={gpu.memType} />
            <Metric label="Bandwidth" value={`${gpu.bandwidthGBs} GB/s`} hint="catalog" />
            <Metric label="FP32" value={`${gpu.fp32} TF`} hint="CUDA cores" />
            <Metric label="BF16 tensor" value={gpu.bf16 ? `${gpu.bf16} TF` : "—"} hint="dense, no sparsity" />
            <Metric label="Tensor cores" value={gpu.tensorCores.toLocaleString()} hint={`${gpu.sms} SMs`} />
            <Metric label="Link" value={`${gpu.linkGBs} GB/s`} hint={gpu.interconnect} />
          </div>
        </Panel>
        <Panel className="flex flex-col p-4 lg:col-span-2">
          <p className="text-xs text-muted">{run.example ? "Worked example" : "Latest selection"}</p>
          <h2 className="mt-1 text-lg font-semibold">{winner.name}</h2>
          <p className="mt-2 font-mono text-4xl tabular-nums text-primary">
            <Speed value={winner.bench.speedup} />
          </p>
          <p className="mt-2 text-sm leading-6 text-muted">
            {run.analysis.title} on {getGpu(run.gpuId).name}. Median {formatTime(winner.bench.medianUs)}, peak
            memory {formatGb(winner.bench.vramMb)}.
          </p>
          <button
            type="button"
            className="mt-4 h-11 rounded-md bg-primary px-4 text-sm font-medium text-primary-fg"
            onClick={() => setModule(run.agentLog ? "agent" : "optimizer")}
          >
            Open the study
          </button>
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel className="p-4">
          <h2 className="text-sm font-semibold">Load a workload</h2>
          <ul className="mt-3 divide-y divide-border">
            {SAMPLES.map((sample) => (
              <li key={sample.id} className="flex items-center gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{sample.name}</p>
                  <p className="truncate text-xs text-muted">{sample.blurb}</p>
                </div>
                <button
                  type="button"
                  className="h-10 shrink-0 rounded-md border border-border px-3 text-sm hover:border-primary/50"
                  onClick={() => loadSample(sample.id)}
                >
                  Load
                </button>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel className="p-4">
          <h2 className="text-sm font-semibold">Recent studies</h2>
          <ul className="mt-3 divide-y divide-border">
            {history.slice(0, 5).map((item) => {
              const pick = item.candidates.find((c) => c.id === item.winnerId);
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    className="flex w-full items-center gap-3 py-3 text-left"
                    onClick={() => openRun(item.id)}
                  >
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm">{item.analysis.title}</p>
                      <p className="truncate text-xs text-muted">
                        {getGpu(item.gpuId).name} · {pick?.name}
                      </p>
                    </div>
                    <Speed value={pick?.bench.speedup ?? 1} />
                  </button>
                </li>
              );
            })}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
