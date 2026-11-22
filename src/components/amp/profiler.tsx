import { getGpu } from "@/lib/amp/gpus";
import { formatTime } from "@/lib/amp/format";
import { useAmp } from "@/lib/amp/store";
import { PageHead, Panel, shareOf } from "@/components/amp/bits";

const TONE: Record<string, string> = {
  compute: "bg-primary",
  memory: "bg-accent",
  launch: "bg-muted",
  sync: "bg-danger",
  data: "bg-muted",
};

export function Profiler() {
  const run = useAmp((s) => s.lastRun);
  const setModule = useAmp((s) => s.setModule);
  const gpu = getGpu(run.gpuId);
  const baseline = run.candidates[0];

  return (
    <div>
      <PageHead
        kicker="PROFILER"
        title={run.analysis.title}
        lede={run.analysis.summary}
      >
        <button
          type="button"
          className="h-11 rounded-md border border-border px-4 text-sm hover:border-primary/50"
          onClick={() => setModule("optimizer")}
        >
          Back to source
        </button>
      </PageHead>
      <Panel className="p-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs text-muted">Baseline on {gpu.name}</p>
            <p className="mt-1 font-mono text-2xl tabular-nums">{formatTime(baseline.bench.medianUs)}</p>
          </div>
          <p className="max-w-sm text-sm text-muted">
            Shares are the visible part of the roofline. The smaller of compute and memory is hidden behind the other.
          </p>
        </div>
        <ul className="mt-5 space-y-4">
          {run.bottlenecks.map((item) => (
            <li key={item.name}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <p className="text-sm font-medium">{item.name}</p>
                <p className="font-mono text-sm tabular-nums">{shareOf(item.share)}</p>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                <div className={`h-full rounded-full ${TONE[item.kind] ?? "bg-primary"}`} style={{ width: `${Math.max(2, item.share * 100)}%` }} />
              </div>
              <p className="mt-1.5 text-sm leading-6 text-muted">{item.detail}</p>
            </li>
          ))}
        </ul>
      </Panel>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {run.analysis.issues.map((issue) => (
          <Panel key={issue} className="p-4">
            <p className="text-xs text-primary">Finding</p>
            <p className="mt-1 text-sm leading-6">{issue}</p>
          </Panel>
        ))}
        {run.analysis.ops.map((op) => (
          <Panel key={op.label + op.kind} className="p-4">
            <p className="font-mono text-xs text-muted">{op.kind}</p>
            <p className="mt-1 text-sm font-medium">{op.label}</p>
            <p className="mt-1 font-mono text-xs text-muted">{shapeOf(op)}</p>
          </Panel>
        ))}
      </div>
    </div>
  );
}

function shapeOf(op: {
  kind: string;
  label: string;
  m?: number;
  n?: number;
  k?: number;
  count?: number;
  batch?: number;
  heads?: number;
  seq?: number;
  headDim?: number;
  elements?: number;
  calls?: number;
}): string {
  if (op.kind === "gemm") return `${op.m}×${op.k} × ${op.k}×${op.n} · ×${op.count}`;
  if (op.kind === "attention") return `B${op.batch} H${op.heads} S${op.seq} D${op.headDim}`;
  if (op.kind === "norm" || op.kind === "pointwise" || op.kind === "element") return `${op.elements?.toLocaleString()} elements · ×${op.count}`;
  if (op.kind === "sync") return `${op.calls} host reads`;
  return op.label;
}
