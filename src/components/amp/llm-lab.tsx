import { useMemo } from "react";
import { getGpu } from "@/lib/amp/gpus";
import { formatToks } from "@/lib/amp/format";
import { getLlmModel, LLM_MODELS, reportLlm } from "@/lib/amp/llm";
import { useAmp } from "@/lib/amp/store";
import { cn } from "@/lib/utils";
import { controlClass, Field, PageHead, Panel, Tag } from "@/components/amp/bits";

export function LlmLab() {
  const gpu = getGpu(useAmp((s) => s.gpuId));
  const modelId = useAmp((s) => s.llmModelId);
  const prompt = useAmp((s) => s.promptTokens);
  const generated = useAmp((s) => s.generatedTokens);
  const batch = useAmp((s) => s.llmBatch);
  const continuous = useAmp((s) => s.continuous);
  const speculative = useAmp((s) => s.speculative);
  const budget = useAmp((s) => s.memoryBudgetGb);
  const objective = useAmp((s) => s.objective);
  const policy = useAmp((s) => s.precisionPolicy);
  const setLlm = useAmp((s) => s.setLlm);
  const model = getLlmModel(modelId);
  const report = useMemo(
    () =>
      reportLlm({
        model,
        gpu,
        batch,
        prompt,
        generated,
        budgetGb: budget,
        continuous,
        speculative,
        objective,
        policy,
      }),
    [model, gpu, batch, prompt, generated, budget, continuous, speculative, objective, policy],
  );

  return (
    <div>
      <PageHead
        kicker="LLM INFERENCE"
        title="Tokens, cache, and the memory wall"
        lede={report.summary}
      />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Field label="Model">
          <select className={controlClass} value={modelId} onChange={(e) => setLlm({ llmModelId: e.target.value })}>
            {LLM_MODELS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Prompt tokens">
          <input
            className={controlClass}
            type="number"
            min={16}
            max={32768}
            value={prompt}
            onChange={(e) => setLlm({ promptTokens: clamp(Number(e.target.value), 16, 32768) })}
          />
        </Field>
        <Field label="Generated tokens">
          <input
            className={controlClass}
            type="number"
            min={1}
            max={8192}
            value={generated}
            onChange={(e) => setLlm({ generatedTokens: clamp(Number(e.target.value), 1, 8192) })}
          />
        </Field>
        <Field label="Batch">
          <input
            className={controlClass}
            type="number"
            min={1}
            max={128}
            value={batch}
            onChange={(e) => setLlm({ llmBatch: clamp(Number(e.target.value), 1, 128) })}
          />
        </Field>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <Toggle on={continuous} label="Continuous batching" onClick={() => setLlm({ continuous: !continuous })} />
        <Toggle on={speculative} label="Speculative decoding" onClick={() => setLlm({ speculative: !speculative })} />
      </div>
      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {report.rows.map((row) => {
          const selected = report.winner === row.prec;
          return (
            <Panel key={row.prec} className={cn("p-4", selected && "border-primary")}>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold">{row.label}</h2>
                {selected ? <Tag tone="warn">Selected</Tag> : null}
                <Tag tone={row.fits ? "good" : "bad"}>{row.fits ? "Fits" : "OOM"}</Tag>
                <span className="ml-auto font-mono text-sm tabular-nums text-muted">q {row.quality}</span>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="TTFT" value={row.fits ? `${row.ttftMs.toFixed(0)} ms` : "—"} />
                <Stat label="TPOT" value={row.fits ? `${row.tpotMs.toFixed(2)} ms` : "—"} />
                <Stat label="Throughput" value={row.fits ? formatToks(row.toksPerSec) : "—"} />
                <Stat label="KV cache" value={`${row.kvGb.toFixed(2)} GB`} />
              </div>
              <p className="mt-3 text-sm leading-6 text-muted">
                Weights {row.weightGb.toFixed(2)} GB · resident {row.totalGb.toFixed(2)} GB · {row.concurrency} sequences. {row.note}
              </p>
            </Panel>
          );
        })}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 font-mono text-sm tabular-nums">{value}</p>
    </div>
  );
}

function Toggle({ on, label, onClick }: { on: boolean; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "h-10 rounded-md border px-3 text-sm",
        on ? "border-primary bg-primary/15 text-fg" : "border-border text-muted",
      )}
      aria-pressed={on}
    >
      {label}
    </button>
  );
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}
