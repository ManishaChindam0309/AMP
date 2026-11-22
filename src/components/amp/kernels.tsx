import { useMemo, useState } from "react";
import { analyze } from "@/lib/amp/analyze";
import { renderCandidate } from "@/lib/amp/codegen";
import { getGpu } from "@/lib/amp/gpus";
import { formatTime } from "@/lib/amp/format";
import { bestTile } from "@/lib/amp/model";
import { autotuneRows } from "@/lib/amp/run";
import { useAmp } from "@/lib/amp/store";
import { cn } from "@/lib/utils";
import { CodeView, PageHead, Panel, Tag } from "@/components/amp/bits";

export function KernelLab() {
  const code = useAmp((s) => s.code);
  const gpu = getGpu(useAmp((s) => s.gpuId));
  const policy = useAmp((s) => s.precisionPolicy);
  const last = useAmp((s) => s.lastRun);
  const analysis = last.code === code ? last.analysis : analyze(code);
  const rows = useMemo(() => autotuneRows(analysis, gpu, policy), [analysis, gpu, policy]);
  const [picked, setPicked] = useState(0);
  const safeIndex = Math.min(picked, Math.max(0, rows.length - 1));
  const row = rows[safeIndex];
  const tile = row?.tile ?? bestTile(gpu);
  const source = renderCandidate(
    analysis,
    {
      id: policy === "exact" ? "triton-fp32" : "triton-fast",
      name: "autotuned",
      compiled: false,
      dtype: policy === "exact" || gpu.bf16 <= 0 ? "fp32" : "bf16",
      flash: analysis.ops.some((op) => op.kind === "attention"),
      tritonGemm: analysis.archetype !== "norm" && analysis.archetype !== "cuda",
      tritonNorm: analysis.archetype === "norm" || analysis.archetype === "transformer",
      dropSync: true,
      checkpoint: false,
      vectorWidth: analysis.archetype === "cuda" ? 4 : 1,
      tile,
    },
    gpu,
  );

  return (
    <div>
      <PageHead
        kicker="KERNEL LAB"
        title="Tile the kernel for this device"
        lede="Block size, stages, and shared memory are scored on the same roofline. The top row is the configuration AMP would cache for this shape."
      />
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="bg-surface text-xs text-muted">
            <tr>
              <th className="px-3 py-3 font-medium">Tile</th>
              <th className="px-3 py-3 font-medium">Warps</th>
              <th className="px-3 py-3 font-medium">Stages</th>
              <th className="px-3 py-3 font-medium">Shared</th>
              <th className="px-3 py-3 font-medium">Efficiency</th>
              <th className="px-3 py-3 font-medium">Median</th>
              <th className="px-3 py-3 font-medium">Bound</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((item, index) => (
              <tr
                key={`${item.tile.bm}-${item.tile.bn}-${item.tile.bk}-${item.tile.stages}`}
                className={cn("cursor-pointer border-t border-border", index === safeIndex ? "bg-primary/10" : "bg-bg")}
                onClick={() => setPicked(index)}
              >
                <td className="px-3 py-3 font-mono">
                  {item.tile.bm}×{item.tile.bn}×{item.tile.bk}
                  {index === 0 ? (
                    <span className="ml-2">
                      <Tag tone="warn">Best</Tag>
                    </span>
                  ) : null}
                </td>
                <td className="px-3 py-3 font-mono tabular-nums">{item.tile.warps}</td>
                <td className="px-3 py-3 font-mono tabular-nums">{item.tile.stages}</td>
                <td className="px-3 py-3 font-mono tabular-nums">{(item.smem / 1024).toFixed(1)} KB</td>
                <td className="px-3 py-3 font-mono tabular-nums">{Math.round(item.eff * 100)}%</td>
                <td className="px-3 py-3 font-mono tabular-nums">{formatTime(item.medianUs)}</td>
                <td className="px-3 py-3">{item.bound}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4">
        <CodeView code={source} language={analysis.archetype === "cuda" ? "cuda" : "python"} />
      </div>
    </div>
  );
}
