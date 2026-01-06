import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { CheckCircle, TrendingUp, BarChart3, Clock, Cpu } from "lucide-react";

interface ForecastResultsProps {
  isLoading: boolean;
  task: any | null;
  device: string;
  selectedModel: string;
  featureCols: string[];
}

export function ForecastResults({ isLoading, task, device, selectedModel, featureCols }: ForecastResultsProps) {
  const metricsPayload = task?.metrics || {};
  const plotData = task?.plot_data || {};
  const valPlot = plotData?.val;
  const testPlot = plotData?.test;

  const predictions = task?.predictions || task?.artifacts?.predictions || task?.artifacts?.data?.predictions || [];

  if (!task && !isLoading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-16 flex items-center justify-center">
        <div className="text-center text-gray-400 max-w-md">
          <div className="p-6 bg-gray-100 rounded-full w-32 h-32 mx-auto mb-6 flex items-center justify-center">
            <TrendingUp className="w-16 h-16 opacity-50" />
          </div>
          <h3 className="text-gray-600 mb-2">Ready to Forecast</h3>
          <p className="text-gray-500">
            Upload your data and configure your model settings to begin generating forecasts.
          </p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-16 flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-24 h-24 mx-auto mb-6">
            <div className="absolute inset-0 border-4 border-indigo-200 rounded-full" />
            <div className="absolute inset-0 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
            <TrendingUp className="absolute inset-0 m-auto w-10 h-10 text-indigo-600" />
          </div>
          <h3 className="text-gray-900 mb-2">Training {selectedModel} Model</h3>
          <p className="text-gray-600">Processing your data on {device}...</p>
          <div className="mt-6 flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-indigo-600 rounded-full animate-pulse" style={{ animationDelay: "0ms" }} />
            <div className="w-2 h-2 bg-indigo-600 rounded-full animate-pulse" style={{ animationDelay: "150ms" }} />
            <div className="w-2 h-2 bg-indigo-600 rounded-full animate-pulse" style={{ animationDelay: "300ms" }} />
          </div>
        </div>
      </div>
    );
  }

  const metrics = [
    {
      label: "Model",
      value: task?.task_model || task?.model_name || selectedModel,
      icon: TrendingUp,
      color: "from-indigo-500 to-purple-500",
    },
    {
      label: "Device",
      value: device,
      icon: Cpu,
      color: "from-green-500 to-emerald-500",
    },
    {
      label: "Status",
      value: task?.status || "pending",
      icon: Clock,
      color: "from-amber-500 to-orange-500",
    },
    {
      label: "Degraded",
      value: task?.degraded ? "Yes" : "No",
      icon: BarChart3,
      color: "from-blue-500 to-cyan-500",
    },
  ];

  const pickPlotData = () => {
    if (testPlot && Array.isArray(testPlot.ts)) {
      return testPlot.ts.map((t: any, i: number) => ({
        ts: t,
        true: testPlot.true?.[i],
        pred: testPlot.pred?.[i],
      }));
    }
    if (valPlot && Array.isArray(valPlot.ts)) {
      return valPlot.ts.map((t: any, i: number) => ({
        ts: t,
        true: valPlot.true?.[i],
        pred: valPlot.pred?.[i],
      }));
    }
    return predictions?.map((v: number, idx: number) => ({ ts: idx, pred: v })) || [];
  };

  const chartData = pickPlotData();
  const valMetrics = metricsPayload?.validation || {};
  const testMetrics = metricsPayload?.test || {};

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 space-y-8">
      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className="bg-gradient-to-br from-gray-50 to-white border border-gray-100 rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className={`p-2 rounded-lg bg-gradient-to-br ${metric.color} text-white shadow`}>
                <metric.icon className="w-4 h-4" />
              </div>
              <div className="text-sm text-gray-500">{metric.label}</div>
            </div>
            <div className="text-2xl font-semibold text-gray-900">{metric.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <div className="text-gray-900 font-semibold mb-2">Validation Metrics</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>RMSE: {valMetrics?.rmse ?? "—"}</div>
            <div>MAPE: {valMetrics?.mape ?? valMetrics?.mape_safe ?? "—"}</div>
          </div>
        </div>
        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <div className="text-gray-900 font-semibold mb-2">Test Metrics</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>RMSE: {testMetrics?.rmse ?? "—"}</div>
            <div>MAPE: {testMetrics?.mape ?? testMetrics?.mape_safe ?? "—"}</div>
          </div>
        </div>
      </div>

      {featureCols?.length ? (
        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm text-sm text-gray-700">
          <div className="font-semibold text-gray-900 mb-2">Selected feature columns</div>
          <p className="text-gray-500 mb-2">Auto-filtered: drop missing rate &gt; 0.4 and low correlation (&lt; 0.05)</p>
          <div className="flex flex-wrap gap-2">
            {featureCols.map((f) => (
              <span key={f} className="px-2 py-1 bg-indigo-50 text-indigo-700 rounded-lg border border-indigo-100">
                {f}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-gray-900 font-semibold flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-indigo-600" />
                Forecast
              </div>
              <div className="text-gray-500 text-sm">
                {selectedModel} on {device}
              </div>
            </div>
            <div className="px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-sm">
              {task?.status || "pending"}
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis dataKey="ts" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="true"
                  stroke="#94a3b8"
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                  name="True"
                />
                <Line
                  type="monotone"
                  dataKey="pred"
                  stroke="#4f46e5"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls
                  name="Pred"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <div className="flex items-center gap-2 mb-3 text-gray-900 font-semibold">
            <CheckCircle className="w-4 h-4 text-emerald-600" />
            Raw predictions
          </div>
          <div className="max-h-64 overflow-auto text-sm text-gray-700">
            {predictions && predictions.length ? (
              <ul className="space-y-1">
                {predictions.slice(0, 50).map((v: number, i: number) => (
                  <li key={i} className="flex justify-between border-b border-gray-100 py-1">
                    <span className="text-gray-500">t+{i + 1}</span>
                    <span className="font-medium text-gray-900">{v.toFixed ? v.toFixed(4) : v}</span>
                  </li>
                ))}
                {predictions.length > 50 && (
                  <li className="text-gray-400 text-xs">...{predictions.length - 50} more</li>
                )}
              </ul>
            ) : (
              <div className="text-gray-400">No predictions yet.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
