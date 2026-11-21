export function formatTime(us: number): string {
  if (!Number.isFinite(us)) return "—";
  const abs = Math.abs(us);
  if (abs < 1000) return `${us.toFixed(abs < 10 ? 2 : abs < 100 ? 1 : 0)} µs`;
  if (abs < 1_000_000) {
    const ms = us / 1000;
    return `${ms.toFixed(ms < 10 ? 2 : ms < 100 ? 1 : 0)} ms`;
  }
  return `${(us / 1_000_000).toFixed(2)} s`;
}

export function formatSpeedup(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return `${x.toFixed(x >= 10 ? 1 : 2)}×`;
}

export function formatGb(mb: number): string {
  if (mb < 1024) return `${mb.toFixed(mb < 10 ? 1 : 0)} MB`;
  return `${(mb / 1024).toFixed(mb / 1024 >= 10 ? 1 : 2)} GB`;
}

export function formatNum(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(digits)}T`;
  if (abs >= 1e9) return `${(n / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(digits)}k`;
  return n.toFixed(digits);
}

export function formatToks(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 100) return `${n.toFixed(0)} tok/s`;
  if (n >= 10) return `${n.toFixed(1)} tok/s`;
  return `${n.toFixed(2)} tok/s`;
}
