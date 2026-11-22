import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function HorizontalBars({
  rows,
  unit,
}: {
  rows: { name: string; value: number; win?: boolean }[];
  unit: string;
}) {
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  if (!ready) return <div className="h-52 w-full" />;
  return (
    <div className="h-52 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
          <XAxis
            type="number"
            tick={{ fill: "var(--color-muted)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            unit={unit}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={108}
            tick={{ fill: "var(--color-fg)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "var(--color-surface-2)" }}
            contentStyle={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              color: "var(--color-fg)",
              fontSize: 12,
            }}
            formatter={(value) => [`${Number(value).toFixed(2)} ${unit}`, ""]}
          />
          <Bar dataKey="value" radius={4} barSize={14}>
            {rows.map((row) => (
              <Cell key={row.name} fill={row.win ? "var(--color-primary)" : "var(--color-border)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
