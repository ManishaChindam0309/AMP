import { useEffect, useState } from "react";
import { getGpu } from "@/lib/amp/gpus";
import { formatGb, formatTime } from "@/lib/amp/format";
import { SAMPLES } from "@/lib/amp/samples";
import { useAmp } from "@/lib/amp/store";
import type { Candidate, StrategyId } from "@/lib/amp/types";
import { cn } from "@/lib/utils";
import { CodeView, controlClass, Field, PageHead, Panel, Speed, Tag } from "@/components/amp/bits";
import { HorizontalBars } from "@/components/amp/charts";

const STAGES = [
  "Reading the workload",
  "Mapping shapes and dtypes",
  "Classifying the limiter",
  "Writing three candidates",
  "Checking numerical gates",
  "Scoring the device model",
];

export function Optimizer() {
  const code = useAmp((s) => s.code);
  const setCode = useAmp((s) => s.setCode);
  const instruction = useAmp((s) => s.instruction);
  const setInstruction = useAmp((s) => s.setInstruction);
  const sampleId = useAmp((s) => s.sampleId);
  const loadSample = useAmp((s) => s.loadSample);
  const budget = useAmp((s) => s.memoryBudgetGb);
  const setBudget = useAmp((s) => s.setBudget);
  const policy = useAmp((s) => s.precisionPolicy);
  const setPolicy = useAmp((s) => s.setPolicy);
  const objective = useAmp((s) => s.objective);
  const setObjective = useAmp((s) => s.setObjective);
  const gpuId = useAmp((s) => s.gpuId);
  const run = useAmp((s) => s.run);
  const result = useAmp((s) => s.lastRun);
  const notes = useAmp((s) => s.notes);
  const gpu = getGpu(gpuId);
  const [phase, setPhase] = useState<number | null>(null);
  const [openId, setOpenId] = useState<string>(result.winnerId);

  useEffect(() => {
    setOpenId(result.winnerId);
  }, [result.id, result.winnerId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        if (phase !== null) return;
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (reduce) {
          run(false);
          return;
        }
        setPhase(0);
        let step = 0;
        const timer = window.setInterval(() => {
          step += 1;
          if (step >= STAGES.length) {
            window.clearInterval(timer);
            setPhase(null);
            run(false);
          } else {
            setPhase(step);
          }
        }, 260);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, run]);

  function start() {
    if (phase !== null) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      run(false);
      return;
    }
    setPhase(0);
    let step = 0;
    const timer = window.setInterval(() => {
      step += 1;
      if (step >= STAGES.length) {
        window.clearInterval(timer);
        setPhase(null);
        run(false);
      } else {
        setPhase(step);
      }
    }, 260);
  }

  const winner = result.candidates.find((c) => c.id === result.winnerId) ?? result.candidates[0];
  const active = result.candidates.find((c) => c.id === openId) ?? winner;
  const sameStudy = result.code === code && result.gpuId === gpuId;

  return (
    <div>
      <PageHead
        kicker="OPTIMIZER"
        title="Three candidates, one winner"
        lede="Paste PyTorch or CUDA. AMP writes alternatives, rejects anything that fails correctness or memory, and keeps the fastest that remains."
      />
      <div className="grid gap-4 xl:grid-cols-12">
        <div className="flex flex-col gap-4 xl:col-span-5">
          <Panel className="p-4">
            <Field label="Instruction">
              <textarea
                className={cn(controlClass, "h-20 resize-none py-2 leading-5")}
                value={instruction}
                onChange={(event) => setInstruction(event.target.value)}
                aria-label="Optimization instruction"
              />
            </Field>
            <div className="mt-3 flex flex-wrap gap-2">
              {SAMPLES.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  onClick={() => loadSample(sample.id)}
                  className={cn(
                    "h-11 rounded-md border px-3 text-xs",
                    sampleId === sample.id
                      ? "border-primary bg-primary/15 text-fg"
                      : "border-border text-muted hover:text-fg",
                  )}
                >
                  {sample.name}
                </button>
              ))}
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <Field label="Precision">
                <select
                  className={controlClass}
                  value={policy}
                  onChange={(event) => setPolicy(event.target.value as typeof policy)}
                >
                  <option value="auto">Auto, bf16 allowed</option>
                  <option value="exact">Exact FP32</option>
                  <option value="max-speed">Maximum speed</option>
                </select>
              </Field>
              <Field label="Objective">
                <select
                  className={controlClass}
                  value={objective}
                  onChange={(event) => setObjective(event.target.value as typeof objective)}
                >
                  <option value="latency">Latency</option>
                  <option value="throughput">Throughput</option>
                </select>
              </Field>
            </div>
            <label className="mt-4 block">
              <span className="flex items-center justify-between text-xs text-muted">
                Memory budget
                <span className="font-mono tabular-nums text-fg">{budget.toFixed(1)} GB</span>
              </span>
              <input
                type="range"
                min={1}
                max={gpu.vramGb}
                step={0.5}
                value={Math.min(budget, gpu.vramGb)}
                onChange={(event) => setBudget(Number(event.target.value))}
                className="mt-2 w-full accent-primary"
              />
            </label>
          </Panel>
          <Panel className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-border px-3 py-2">
              <span className="font-mono text-xs text-muted">
                {code.includes("__global__") ? "cuda" : "python"} · {code.split("\n").length} lines
              </span>
              <span className="font-mono text-xs text-muted">{gpu.name}</span>
            </div>
            <textarea
              value={code}
              onChange={(event) => setCode(event.target.value)}
              spellCheck={false}
              aria-label="Workload source"
              className="h-80 w-full resize-y bg-bg p-3 font-mono text-xs leading-5 text-fg outline-none"
            />
          </Panel>
          <button
            type="button"
            onClick={start}
            disabled={phase !== null}
            className="h-12 rounded-md bg-primary text-sm font-semibold text-primary-fg disabled:opacity-60"
          >
            {phase !== null ? "Scoring…" : "Run optimization"}
          </button>
        </div>

        <div className="flex flex-col gap-4 xl:col-span-7">
          {phase !== null ? (
            <Panel className="p-4">
              <ol className="space-y-2">
                {STAGES.map((stage, index) => (
                  <li key={stage} className="flex items-center gap-3 text-sm">
                    <span
                      className={cn(
                        "grid size-6 place-items-center rounded-full font-mono text-xs",
                        index < phase
                          ? "bg-accent text-bg"
                          : index === phase
                            ? "bg-primary text-primary-fg"
                            : "bg-surface-2 text-muted",
                      )}
                    >
                      {index + 1}
                    </span>
                    <span className={index > phase ? "text-muted" : "text-fg"}>{stage}</span>
                  </li>
                ))}
              </ol>
            </Panel>
          ) : (
            <>
              {!sameStudy ? (
                <p className="text-sm text-muted">Source changed since the last score. Run again to refresh the study.</p>
              ) : null}
              <Panel className="p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-xs text-muted">{result.example ? "Worked example" : "Selection"}</p>
                    <h2 className="mt-1 text-xl font-semibold">{winner.name}</h2>
                    <p className="mt-1 max-w-xl text-sm leading-6 text-muted">{winner.summary}</p>
                  </div>
                  <p className="font-mono text-3xl tabular-nums">
                    <Speed value={winner.bench.speedup} />
                  </p>
                </div>
                <ul className="mt-3 space-y-1.5 text-sm leading-6 text-fg">
                  {result.explanation.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
                {notes.length ? (
                  <p className="mt-3 text-xs text-muted">Read from the instruction: {notes.join(" ")}</p>
                ) : null}
              </Panel>

              <div className="grid gap-3 sm:grid-cols-3">
                <Panel className="p-3">
                  <p className="text-xs text-muted">Architecture</p>
                  <p className="mt-1 text-sm">{result.analysis.architecture}</p>
                </Panel>
                <Panel className="p-3">
                  <p className="text-xs text-muted">Shape</p>
                  <p className="mt-1 font-mono text-sm">
                    {[result.analysis.batch, result.analysis.seq, result.analysis.hidden].filter(Boolean).join(" × ") ||
                      "—"}
                  </p>
                </Panel>
                <Panel className="p-3">
                  <p className="text-xs text-muted">Already present</p>
                  <p className="mt-1 text-sm">{result.analysis.existing.join(", ")}</p>
                </Panel>
              </div>

              <div className="flex flex-col gap-3">
                {result.candidates.map((candidate) => (
                  <CandidateRow
                    key={candidate.id}
                    candidate={candidate}
                    winner={candidate.id === result.winnerId}
                    open={openId === candidate.id}
                    onOpen={() => setOpenId(candidate.id)}
                  />
                ))}
              </div>

              <Panel className="p-4">
                <h2 className="text-sm font-semibold">Median time</h2>
                <HorizontalBars
                  unit="ms"
                  rows={result.candidates.map((candidate) => ({
                    name: shortLabel(candidate.id),
                    value: Number((candidate.bench.medianUs / 1000).toFixed(2)),
                    win: candidate.id === result.winnerId,
                  }))}
                />
                <p className="text-xs leading-5 text-muted">
                  Times are AMP’s hardware model: roofline from catalog peak and bandwidth, plus launch and sync.
                  A device run would use 10 warmup iterations and 50 measured ones, then report the median and p95.
                </p>
              </Panel>

              <CodeView code={active.code} language={active.language === "cuda" ? "cuda" : "python"} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function shortLabel(id: StrategyId): string {
  switch (id) {
    case "baseline":
      return "Baseline";
    case "drop-sync":
      return "No sync";
    case "sdpa-bf16":
      return "SDPA";
    case "compile-bf16":
      return "BF16";
    case "compile-fp32":
      return "Compile";
    case "compile-tf32":
      return "TF32";
    case "triton-fast":
    case "triton-fp32":
      return "Triton";
    case "vectorized":
      return "float4";
    default:
      return id;
  }
}

function CandidateRow({
  candidate,
  winner,
  open,
  onOpen,
}: {
  candidate: Candidate;
  winner: boolean;
  open: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className={cn(
        "rounded-lg border p-3 text-left",
        winner ? "border-primary bg-primary/5" : "border-border bg-surface",
        open && "ring-2 ring-primary/40",
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-sm font-semibold">{candidate.name}</p>
        {winner ? <Tag tone="warn">Selected</Tag> : null}
        <Tag
          tone={
            candidate.check.numerical === "pass"
              ? "good"
              : candidate.check.numerical === "approximate"
                ? "warn"
                : "bad"
          }
        >
          {candidate.check.numerical} · rtol {candidate.check.rtol}
        </Tag>
        {!candidate.eligible ? <Tag tone="bad">Excluded</Tag> : null}
        <span className="ml-auto">
          <Speed value={candidate.bench.speedup} />
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-xs tabular-nums text-muted sm:grid-cols-4">
        <span>med {formatTime(candidate.bench.medianUs)}</span>
        <span>p95 {formatTime(candidate.bench.p95Us)}</span>
        <span>{formatGb(candidate.bench.vramMb)}</span>
        <span>{candidate.bench.bound}</span>
      </div>
      {candidate.rejectReason ? <p className="mt-2 text-xs text-danger">{candidate.rejectReason}</p> : null}
    </button>
  );
}
