import { useState, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatSpeedup } from "@/lib/amp/format";

export function Panel({ className, children }: { className?: string; children: ReactNode }) {
  return <section className={cn("rounded-lg border border-border bg-surface", className)}>{children}</section>;
}

export function PageHead({
  kicker,
  title,
  lede,
  children,
}: {
  kicker: string;
  title: string;
  lede?: string;
  children?: ReactNode;
}) {
  return (
    <header className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
      <div className="min-w-0">
        <p className="font-mono text-xs tracking-widest text-primary">{kicker}</p>
        <h1 className="mt-1 text-2xl font-semibold text-fg">{title}</h1>
        {lede ? <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">{lede}</p> : null}
      </div>
      {children}
    </header>
  );
}

export function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="min-w-0">
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 truncate font-mono text-lg tabular-nums text-fg">{value}</p>
      {hint ? <p className="mt-0.5 truncate text-xs text-muted">{hint}</p> : null}
    </div>
  );
}

export function Speed({ value }: { value: number }) {
  const tone = value > 1.03 ? "text-accent" : value < 0.97 ? "text-danger" : "text-muted";
  return <span className={cn("font-mono tabular-nums", tone)}>{formatSpeedup(value)}</span>;
}

export function Tag({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-xs",
        tone === "neutral" && "border-border bg-surface-2 text-muted",
        tone === "good" && "border-accent/40 bg-accent/10 text-accent",
        tone === "warn" && "border-primary/40 bg-primary/10 text-primary",
        tone === "bad" && "border-danger/40 bg-danger/10 text-danger",
      )}
    >
      {children}
    </span>
  );
}

export function CodeView({ code, language }: { code: string; language: string }) {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-bg">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <span className="font-mono text-xs text-muted">{language}</span>
        <CopyButton text={code} />
      </div>
      <pre className="max-h-96 overflow-auto p-3 font-mono text-xs leading-5 text-fg">{code}</pre>
    </div>
  );
}

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 text-xs text-fg hover:border-primary/50"
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setDone(true);
          window.setTimeout(() => setDone(false), 1200);
        });
      }}
    >
      {done ? <Check className="size-3.5 text-accent" /> : <Copy className="size-3.5" />}
      {done ? "Copied" : label}
    </button>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0">
      <span className="mb-1.5 block text-xs text-muted">{label}</span>
      {children}
    </label>
  );
}

export const controlClass =
  "h-11 w-full rounded-md border border-border bg-bg px-3 text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-primary";

export function shareOf(share: number): string {
  return `${Math.round(share * 100)}%`;
}
