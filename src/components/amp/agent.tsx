import { useState } from "react";
import { getGpu } from "@/lib/amp/gpus";
import { formatTime } from "@/lib/amp/format";
import { useAmp } from "@/lib/amp/store";
import { cn } from "@/lib/utils";
import { PageHead, Panel, Speed, Tag } from "@/components/amp/bits";

export function AgentLab() {
  const budget = useAmp((s) => s.agentBudget);
  const setBudget = useAmp((s) => s.setAgentBudget);
  const run = useAmp((s) => s.run);
  const result = useAmp((s) => s.lastRun);
  const instruction = useAmp((s) => s.instruction);
  const [busy, setBusy] = useState(false);
  const [shown, setShown] = useState(result.agentLog?.length ?? 0);
  const gpu = getGpu(result.gpuId);
  const log = result.agentLog ?? [];

  function start() {
    if (busy) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    run(true);
    const next = useAmp.getState().lastRun.agentLog ?? [];
    if (reduce) {
      setShown(next.length);
      return;
    }
    setBusy(true);
    setShown(1);
    let step = 1;
    const timer = window.setInterval(() => {
      step += 1;
      setShown(step);
      if (step >= next.length) {
        window.clearInterval(timer);
        setBusy(false);
      }
    }, 420);
  }

  const visible = log.slice(0, shown);

  return (
    <div>
      <PageHead
        kicker="AGENT"
        title="Search until the gains stop"
        lede="The agent tries one change at a time, keeps a rewrite only when it is faster and still eligible, and stops after two attempts that do not move the result."
      />
      <Panel className="p-4">
        <p className="text-sm text-muted">Instruction</p>
        <p className="mt-1 text-sm leading-6">{instruction}</p>
        <label className="mt-4 block max-w-md">
          <span className="flex items-center justify-between text-xs text-muted">
            Step budget
            <span className="font-mono tabular-nums text-fg">{budget}</span>
          </span>
          <input
            type="range"
            min={3}
            max={8}
            step={1}
            value={budget}
            onChange={(event) => setBudget(Number(event.target.value))}
            className="mt-2 w-full accent-primary"
          />
        </label>
        <button
          type="button"
          onClick={start}
          disabled={busy}
          className="mt-4 h-11 rounded-md bg-primary px-4 text-sm font-semibold text-primary-fg disabled:opacity-60"
        >
          {busy ? "Searching…" : "Run the search"}
        </button>
      </Panel>
      <ol className="mt-4 space-y-3">
        {visible.length === 0 ? (
          <Panel className="p-4">
            <p className="text-sm text-muted">No agent trace on the open study. Run the search on the current source.</p>
          </Panel>
        ) : (
          visible.map((step, index) => (
            <li key={`${step.title}-${index}`}>
              <Panel className={cn("p-4", step.kept && "border-primary")}>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs text-muted">0{index + 1}</span>
                  <h2 className="text-sm font-semibold">{step.title}</h2>
                  {step.kept ? <Tag tone="warn">Kept</Tag> : <Tag>Passed</Tag>}
                  <span className="ml-auto font-mono text-sm tabular-nums">
                    {formatTime(step.medianUs)} · <Speed value={step.speedup} />
                  </span>
                </div>
                <p className="mt-2 text-sm leading-6 text-muted">{step.detail}</p>
                <p className="mt-1 text-sm">{step.decision}</p>
              </Panel>
            </li>
          ))
        )}
      </ol>
      {log.length > 0 ? (
        <p className="mt-4 text-xs text-muted">
          Trace for {result.analysis.title} on {gpu.name}. Budget {budget} steps.
        </p>
      ) : null}
    </div>
  );
}
