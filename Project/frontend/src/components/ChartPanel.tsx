import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type ChartPoint = {
  ts: string;
  true?: number;
  pred?: number;
};

type ChartPanelProps = {
  title: string;
  data: ChartPoint[];
};

export default function ChartPanel({ title, data }: ChartPanelProps) {
  return (
    <div className="chart-card">
      <p className="chart-title">{title}</p>
      {data.length ? (
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e6ded4" />
              <XAxis dataKey="ts" tickFormatter={(v) => String(v).slice(5, 16)} />
              <YAxis width={36} tickFormatter={(v) => Number(v).toFixed(2)} />
              <Tooltip />
              <Line type="monotone" dataKey="true" stroke="#0e6b6f" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="pred" stroke="#f08c2e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="helper">No plot data available yet.</p>
      )}
    </div>
  );
}
