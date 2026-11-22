import { getGpu } from "@/lib/amp/gpus";
import { formatTime } from "@/lib/amp/format";
import { PROTOCOL } from "@/lib/amp/run";
import { useAmp } from "@/lib/amp/store";
import { PageHead, Panel, Speed } from "@/components/amp/bits";
import { HorizontalBars } from "@/components/amp/charts";

export function BenchCenter() {
  const history = useAmp((s) => s.history);
  const openRun = useAmp((s) => s.openRun);
  const forgetRun = useAmp((s) => s.forgetRun);
  const rows = history.map((item) => {
    const winner = item.candidates.find((c) => c.id === item.winnerId) ?? item.candidates[0];
    return {
      id: item.id,
      name: item.analysis.title.slice(0, 18),
      value: winner.bench.speedup,
      win: winner.id !== "baseline",
    };
  });

  return (
    <div>
      <PageHead
        kicker="BENCHMARKS"
        title="What was tried"
        lede={`Protocol held constant: ${PROTOCOL.warmup} warmup iterations, ${PROTOCOL.iterations} measured, ${PROTOCOL.statistic} and ${PROTOCOL.tail}. Numbers come from the hardware model so a study can be repeated without a device.`}
      />
      {history.length > 1 ? (
        <Panel className="mb-4 p-4">
          <h2 className="text-sm font-semibold">Speedup against each study’s own baseline</h2>
          <HorizontalBars unit="×" rows={rows} />
        </Panel>
      ) : null}
      <div className="flex flex-col gap-3">
        {history.map((item) => {
          const winner = item.candidates.find((c) => c.id === item.winnerId) ?? item.candidates[0];
          const baseline = item.candidates[0];
          return (
            <Panel key={item.id} className="p-4">
              <div className="flex flex-wrap items-start gap-3">
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-muted">
                    {item.example ? "Example" : new Date(item.createdAt).toLocaleString()} · {getGpu(item.gpuId).name}
                  </p>
                  <h2 className="mt-1 text-base font-semibold">{item.analysis.title}</h2>
                  <p className="mt-1 text-sm text-muted">
                    {winner.name} · {formatTime(baseline.bench.medianUs)} → {formatTime(winner.bench.medianUs)}
                  </p>
                </div>
                <p className="font-mono text-2xl">
                  <Speed value={winner.bench.speedup} />
                </p>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  className="h-10 rounded-md border border-border px-3 text-sm hover:border-primary/50"
                  onClick={() => openRun(item.id)}
                >
                  Open
                </button>
                <button
                  type="button"
                  className="h-10 rounded-md border border-border px-3 text-sm hover:border-primary/50"
                  onClick={() => download(`${slug(item.analysis.title)}.py`, winner.code)}
                >
                  Download winner
                </button>
                {!item.example ? (
                  <button
                    type="button"
                    className="h-10 rounded-md border border-border px-3 text-sm text-muted hover:border-danger/50 hover:text-danger"
                    onClick={() => forgetRun(item.id)}
                  >
                    Remove
                  </button>
                ) : null}
              </div>
            </Panel>
          );
        })}
      </div>
    </div>
  );
}

function download(filename: string, text: string) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "amp-winner";
}
