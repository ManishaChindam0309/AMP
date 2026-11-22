import { useMemo } from "react";
import { getGpu } from "@/lib/amp/gpus";
import { getLlmModel, LLM_MODELS } from "@/lib/amp/llm";
import { reportParallel } from "@/lib/amp/parallel";
import { useAmp } from "@/lib/amp/store";
import { cn } from "@/lib/utils";
import { controlClass, Field, PageHead, Panel, Tag } from "@/components/amp/bits";

export function MultiLab() {
  const gpu = getGpu(useAmp((s) => s.gpuId));
  const modelId = useAmp((s) => s.llmModelId);
  const count = useAmp((s) => s.parallelCount);
  const training = useAmp((s) => s.parallelTraining);
  const batch = useAmp((s) => s.parallelBatch);
  const seq = useAmp((s) => s.parallelSeq);
  const micros = useAmp((s) => s.micros);
  const setParallel = useAmp((s) => s.setParallel);
  const setLlm = useAmp((s) => s.setLlm);
  const model = getLlmModel(modelId);
  const report = useMemo(
    () => reportParallel({ gpu, model, count, training, batch, seq, micros }),
    [gpu, model, count, training, batch, seq, micros],
  );

  return (
    <div>
      <PageHead kicker="MULTI-GPU" title="Where to cut the model" lede={report.summary} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        <Field label="Model">
          <select className={controlClass} value={modelId} onChange={(e) => setLlm({ llmModelId: e.target.value })}>
            {LLM_MODELS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Devices">
          <select
            className={controlClass}
            value={count}
            onChange={(e) => setParallel({ parallelCount: Number(e.target.value) })}
          >
            {[1, 2, 4, 8].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Job">
          <select
            className={controlClass}
            value={training ? "train" : "serve"}
            onChange={(e) => setParallel({ parallelTraining: e.target.value === "train" })}
          >
            <option value="serve">Inference serving</option>
            <option value="train">Training step</option>
          </select>
        </Field>
        <Field label="Global batch">
          <input
            className={controlClass}
            type="number"
            min={1}
            max={256}
            value={batch}
            onChange={(e) => setParallel({ parallelBatch: clamp(Number(e.target.value), 1, 256) })}
          />
        </Field>
        <Field label="Sequence">
          <input
            className={controlClass}
            type="number"
            min={128}
            max={32768}
            value={seq}
            onChange={(e) => setParallel({ parallelSeq: clamp(Number(e.target.value), 128, 32768) })}
          />
        </Field>
        <Field label="Micro-batches">
          <input
            className={controlClass}
            type="number"
            min={1}
            max={64}
            value={micros}
            onChange={(e) => setParallel({ micros: clamp(Number(e.target.value), 1, 64) })}
          />
        </Field>
      </div>
      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {report.rows.map((row) => (
          <Panel key={row.id} className={cn("p-4", report.winner === row.id && "border-primary")}>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold">{row.name}</h2>
              {report.winner === row.id ? <Tag tone="warn">Selected</Tag> : null}
              <Tag tone={row.fits ? "good" : "bad"}>{row.fits ? "Fits" : "Does not fit"}</Tag>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3">
              <Stat label="Step" value={row.fits ? `${row.stepMs.toFixed(1)} ms` : "—"} />
              <Stat label="Comm" value={row.fits ? `${row.commMs.toFixed(1)} ms` : "—"} />
              <Stat label="Samples/s" value={row.fits ? row.throughput.toFixed(1) : "—"} />
            </div>
            <p className="mt-3 text-sm leading-6 text-muted">{row.detail}</p>
          </Panel>
        ))}
      </div>
      <p className="mt-4 text-xs leading-5 text-muted">
        Communication uses {gpu.interconnect} at {gpu.linkGBs} GB/s. Collectives are a bus model, not an NCCL trace.
      </p>
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

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}
