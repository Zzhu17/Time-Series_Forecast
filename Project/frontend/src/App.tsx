import {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import Papa from "papaparse";

import { fetchLatestArtifacts, fetchModels, trainFileSync } from "./api";
import type { ModelInfo, PlotSeries, TrainResponse } from "./types";

const DEFAULT_API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const ChartPanel = lazy(() => import("./components/ChartPanel"));

type PreviewRow = Record<string, unknown>;

type RunState = "idle" | "running" | "success" | "error";

type ModelOption = ModelInfo & {
  backendName?: string;
  residualModeling?: Record<string, unknown>;
};

const formatNumber = (value?: number, digits = 4) => {
  if (value === null || value === undefined) {
    return "--";
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "--";
  }
  return num.toFixed(digits);
};

type MetricSummary = { label: string; value: number };

const COMBO_MODELS: ModelOption[] = [
  {
    name: "xgboost+informer",
    description: "Informer forecast + XGBoost residual correction.",
    backendName: "informer",
    residualModeling: {
      enabled: true,
      model_type: "xgboost",
      lags: [1, 2, 3, 6, 12, 24],
      rolling_windows: [6, 12, 24, 48],
      diffs: [1, 24],
    },
  },
  {
    name: "xgboost+lstm",
    description: "LSTM forecast + XGBoost residual correction.",
    backendName: "lstm",
    residualModeling: {
      enabled: true,
      model_type: "xgboost",
      lags: [1, 2, 3, 6, 12, 24],
      rolling_windows: [6, 12, 24, 48],
      diffs: [1, 24],
    },
  },
];

const buildMetricSummary = (
  metrics?: Record<string, Record<string, number>> | null
) => {
  const val = metrics?.validation || metrics?.val;
  const test = metrics?.test;
  const summary: MetricSummary[] = [];
  const push = (label: string, value?: number) => {
    if (value === null || value === undefined || Number.isNaN(value)) return;
    summary.push({ label, value });
  };
  push("Val RMSE", val?.rmse);
  push("Val MAPE", val?.mape);
  push("Val nRMSE", val?.nrmse);
  push("Val sMAPE", val?.smape);
  return summary;
};

const toChartData = (series?: PlotSeries | null) => {
  if (!series || !Array.isArray(series.ts)) {
    return [] as Array<{ ts: string; true: number; pred: number }>;
  }
  return series.ts
    .map((ts, idx) => ({
      ts: String(ts),
      true: Number(series.true?.[idx]),
      pred: Number(series.pred?.[idx]),
    }))
    .filter((row) => Number.isFinite(row.true) && Number.isFinite(row.pred));
};

const pickTimeCol = (columns: string[]) =>
  columns.find((col) => /date|time|timestamp/i.test(col)) || columns[0] || "";

const pickValueCol = (columns: string[], timeCol: string) => {
  const candidate = columns.find((col) => /value|target|y/i.test(col));
  if (candidate && candidate !== timeCol) {
    return candidate;
  }
  return columns.find((col) => col !== timeCol) || "";
};

export default function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [apiBaseInput, setApiBaseInput] = useState(
    () => localStorage.getItem("tsf_api_base") || DEFAULT_API
  );
  const [apiToken, setApiToken] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loadingModels, setLoadingModels] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [rows, setRows] = useState<PreviewRow[]>([]);
  const [rowCount, setRowCount] = useState(0);
  const [timeCol, setTimeCol] = useState("");
  const [valueCol, setValueCol] = useState("");
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [autoFeatures, setAutoFeatures] = useState(true);
  const [horizon, setHorizon] = useState(24);
  const [allowDegrade, setAllowDegrade] = useState(false);
  const [device, setDevice] = useState("cpu");
  const [parsing, setParsing] = useState(false);

  const [runState, setRunState] = useState<RunState>("idle");
  const [runError, setRunError] = useState("");
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [latestResult, setLatestResult] = useState<TrainResponse | null>(null);

  const apiBase = apiBaseInput || DEFAULT_API;
  const modelOptions = useMemo(() => {
    const comboMap = new Map(COMBO_MODELS.map((model) => [model.name, model]));
    const merged = models.map((model) => {
      const combo = comboMap.get(model.name);
      return combo ? { ...combo, ...model } : model;
    });
    const seen = new Set(models.map((model) => model.name));
    const extras = COMBO_MODELS.filter((model) => !seen.has(model.name));
    return [...merged, ...extras];
  }, [models]);
  const selectedOption = useMemo(
    () => modelOptions.find((model) => model.name === selectedModel),
    [modelOptions, selectedModel]
  );
  const selectedRequestName = selectedOption?.name || selectedModel;
  const selectedResidualModeling = selectedOption?.residualModeling ?? null;
  const selectedMissingDeps = selectedOption?.missing_deps || [];
  const selectedUnavailable = selectedOption?.available === false;

  useEffect(() => {
    localStorage.setItem("tsf_api_base", apiBaseInput);
  }, [apiBaseInput]);

  useEffect(() => {
    localStorage.removeItem("tsf_api_token");
  }, []);

  useEffect(() => {
    let mounted = true;
    setLoadingModels(true);
    fetchModels(apiBase, apiToken)
      .then((items) => {
        if (!mounted) return;
        setModels(items);
      })
      .catch((err: Error) => {
        if (!mounted) return;
        setRunError(err.message || "Failed to load models");
        setRunState("error");
      })
      .finally(() => {
        if (!mounted) return;
        setLoadingModels(false);
      });
    return () => {
      mounted = false;
    };
  }, [apiBase, apiToken]);

  useEffect(() => {
    if (!modelOptions.length) return;
    setSelectedModel((prev) => prev || modelOptions[0]?.name || "");
  }, [modelOptions]);

  useEffect(() => {
    let mounted = true;
    fetchLatestArtifacts(apiBase, apiToken)
      .then((payload) => {
        if (!mounted) return;
        setLatestResult(payload);
      })
      .catch(() => {
        if (!mounted) return;
        setLatestResult(null);
      });
    return () => {
      mounted = false;
    };
  }, [apiBase, apiToken]);

  const featureCandidates = useMemo(() => {
    return columns.filter((col) => col !== timeCol && col !== valueCol);
  }, [columns, timeCol, valueCol]);

  useEffect(() => {
    if (!columns.length) return;
    setFeatureCols((prev) => {
      const next = prev.filter((col) => featureCandidates.includes(col));
      return next.length ? next : featureCandidates;
    });
  }, [featureCandidates, columns.length]);

  const timeRange = useMemo(() => {
    if (!rows.length || !timeCol) return "--";
    const values = rows
      .map((row) => row[timeCol])
      .filter((val) => val !== null && val !== undefined && val !== "")
      .map((val) => new Date(String(val)))
      .filter((d) => !Number.isNaN(d.getTime()));
    if (!values.length) return "--";
    const sorted = values.sort((a, b) => a.getTime() - b.getTime());
    const start = sorted[0].toISOString().slice(0, 10);
    const end = sorted[sorted.length - 1].toISOString().slice(0, 10);
    return `${start} to ${end}`;
  }, [rows, timeCol]);

  const previewRows = useMemo(() => rows.slice(0, 8), [rows]);

  const onFileSelected = (nextFile: File) => {
    setParsing(true);
    setFile(nextFile);
    setRunState("idle");
    setRunError("");
    setResult(null);

    Papa.parse(nextFile, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (results) => {
        const dataRows = (results.data || []) as PreviewRow[];
        const cols =
          (results.meta && results.meta.fields
            ? results.meta.fields
            : Object.keys(dataRows[0] || {})) || [];
        const cleaned = dataRows.filter((row) =>
          Object.values(row || {}).some(
            (val) => val !== null && val !== undefined && String(val) !== ""
          )
        );
        const nonEmptyCols = cols.filter((col) =>
          cleaned.some((row) => {
            const val = row?.[col];
            return val !== null && val !== undefined && String(val) !== "";
          })
        );

        setColumns(nonEmptyCols);
        setRows(cleaned);
        setRowCount(cleaned.length);

        const nextTime = pickTimeCol(nonEmptyCols);
        const nextValue = pickValueCol(nonEmptyCols, nextTime);
        setTimeCol(nextTime);
        setValueCol(nextValue);
        setFeatureCols(nonEmptyCols.filter((col) => col !== nextTime && col !== nextValue));
        setParsing(false);
      },
      error: () => {
        setParsing(false);
        setRunError("Failed to parse CSV file");
        setRunState("error");
      },
    });
  };

  const handleFileChange = (evt: ChangeEvent<HTMLInputElement>) => {
    const nextFile = evt.target.files?.[0];
    if (nextFile) {
      onFileSelected(nextFile);
    }
  };

  const handleDrop = (evt: React.DragEvent<HTMLDivElement>) => {
    evt.preventDefault();
    const nextFile = evt.dataTransfer.files?.[0];
    if (nextFile) {
      onFileSelected(nextFile);
    }
  };

  const handleRun = async () => {
    if (!file || !timeCol || !valueCol || !selectedRequestName) {
      setRunError("Please upload a CSV and select model/columns first.");
      setRunState("error");
      return;
    }
    if (selectedUnavailable) {
      const deps = selectedMissingDeps.length
        ? `Missing: ${selectedMissingDeps.join(", ")}`
        : "Missing required dependencies.";
      setRunError(`Selected model is unavailable. ${deps}`);
      setRunState("error");
      return;
    }
    setRunState("running");
    setRunError("");
    setResult(null);

    try {
      const payload = await trainFileSync(
        apiBase,
        {
          file,
          modelName: selectedRequestName,
          timeCol,
          valueCol,
          horizon,
          featureCols: autoFeatures ? null : featureCols,
          allowDegrade,
          device,
          residualModeling: selectedResidualModeling,
        },
        apiToken || undefined
      );
      setResult(payload);
      setLatestResult(payload);
      setRunState("success");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Run failed";
      setRunError(msg);
      setRunState("error");
    }
  };

  const runBadge = useMemo(() => {
    if (runState === "running") return "Running";
    if (runState === "success") return "Complete";
    if (runState === "error") return "Attention";
    return "Idle";
  }, [runState]);

  const activeResult = result ?? latestResult;
  const valMetrics =
    activeResult?.metrics?.validation || activeResult?.metrics?.val || undefined;
  const testMetrics = activeResult?.metrics?.test || undefined;

  const trainSeries = activeResult?.data?.plot_data?.train || null;
  const valSeries = activeResult?.data?.plot_data?.val || null;
  const testSeries = activeResult?.data?.plot_data?.test || null;
  const trainData = useMemo(() => toChartData(trainSeries), [trainSeries]);
  const valData = useMemo(() => toChartData(valSeries), [valSeries]);
  const testData = useMemo(() => toChartData(testSeries), [testSeries]);
  const leaderboard = activeResult?.data?.leaderboard || [];
  const leaderboardPath = activeResult?.data?.leaderboard_path;
  const reportPath = activeResult?.data?.report_path;
  const drift = activeResult?.data?.drift;
  const activeRunId = activeResult?.run_id;
  const baseUrl = apiBase.replace(/\/$/, "");
  const reportUrl =
    activeRunId && reportPath
      ? `${baseUrl}/artifacts/${activeRunId}/report`
      : undefined;
  const leaderboardUrl =
    activeRunId && leaderboardPath
      ? `${baseUrl}/artifacts/${activeRunId}/leaderboard`
      : undefined;

  const artifacts = activeResult?.artifacts || {};
  const artifactEntries = Object.entries(artifacts);
  const formatArtifactValue = (value: unknown) => {
    if (value === null || value === undefined) {
      return "--";
    }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  };

  const degraded = activeResult?.data?.degraded;
  const degradedReason = activeResult?.data?.degraded_reason;
  const latestRunId = latestResult?.run_id;
  const latestModelName =
    latestResult?.model_name ||
    latestResult?.task_model ||
    latestResult?.model_record?.name ||
    "--";
  const latestSummary = useMemo(
    () => buildMetricSummary(latestResult?.metrics),
    [latestResult]
  );
  const modelHelperText = useMemo(() => {
    if (selectedUnavailable) {
      if (selectedMissingDeps.length) {
        return `Unavailable. Missing: ${selectedMissingDeps.join(", ")}`;
      }
      return "Unavailable due to missing dependencies.";
    }
    return selectedOption?.description || "Select a model from the registry.";
  }, [selectedOption, selectedUnavailable, selectedMissingDeps]);
  const displayModelName =
    result && (selectedOption?.name || selectedModel)
      ? selectedOption?.name || selectedModel
      : activeResult?.model_name ||
        activeResult?.task_model ||
        selectedOption?.name ||
        selectedModel ||
        "--";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">FS</span>
          <div>
            <p className="brand-title">ForecastServe</p>
            <p className="brand-subtitle">Customer Forecast Console</p>
          </div>
        </div>
        <div className="topbar-controls">
          <label className="input-group">
            <span>API</span>
            <input
              value={apiBaseInput}
              onChange={(evt) => setApiBaseInput(evt.target.value)}
              placeholder="http://localhost:8000"
            />
          </label>
          <label className="input-group">
            <span>Token</span>
            <input
              type="password"
              value={apiToken}
              onChange={(evt) => setApiToken(evt.target.value)}
              placeholder="Optional"
            />
          </label>
        </div>
      </header>

      <main className="main">
        <section className="hero" data-animate>
          <div>
            <h1>Forecasts your customers can trust.</h1>
            <p>
              Import a time series, choose a model, and ship production-ready
              forecasts with clear metrics and artifacts.
            </p>
            <div className="hero-tags">
              <span>Upload</span>
              <span>Configure</span>
              <span>Train</span>
              <span>Predict</span>
              <span>Report</span>
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="panel">
            <div className="card" data-animate>
              <div className="card-header">
                <h3>1. Import data</h3>
                <span className="pill">CSV only</span>
              </div>
              <div
                className={`dropzone ${parsing ? "busy" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(evt) => evt.preventDefault()}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  hidden
                />
                <p className="dropzone-title">
                  {file ? file.name : "Drop your CSV here"}
                </p>
                <p className="dropzone-subtitle">
                  {parsing
                    ? "Parsing data..."
                    : "Click to browse or drag & drop"}
                </p>
              </div>

              <div className="grid-two">
                <label className="input-group">
                  <span>Time column</span>
                  <select
                    value={timeCol}
                    onChange={(evt) => setTimeCol(evt.target.value)}
                  >
                    <option value="">Select</option>
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="input-group">
                  <span>Value column</span>
                  <select
                    value={valueCol}
                    onChange={(evt) => setValueCol(evt.target.value)}
                  >
                    <option value="">Select</option>
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="toggle-row">
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={autoFeatures}
                    onChange={(evt) => setAutoFeatures(evt.target.checked)}
                  />
                  <span>Auto feature selection</span>
                </label>
              </div>

              {!autoFeatures && (
                <div className="feature-list">
                  {featureCandidates.map((col) => (
                    <label key={col} className="feature-item">
                      <input
                        type="checkbox"
                        checked={featureCols.includes(col)}
                        onChange={() => {
                          setFeatureCols((prev) =>
                            prev.includes(col)
                              ? prev.filter((item) => item !== col)
                              : [...prev, col]
                          );
                        }}
                      />
                      <span>{col}</span>
                    </label>
                  ))}
                </div>
              )}

              {previewRows.length > 0 && (
                <div className="preview">
                  <p className="section-label">Preview</p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          {columns.slice(0, 6).map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewRows.map((row, idx) => (
                          <tr key={idx}>
                            {columns.slice(0, 6).map((col) => (
                              <td key={col}>{String(row[col] ?? "")}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            <div className="card" data-animate>
              <div className="card-header">
                <h3>2. Choose model</h3>
                <span className="pill">{loadingModels ? "Syncing" : "Live"}</span>
              </div>
              <label className="input-group">
                <span>Model</span>
                <select
                  value={selectedModel}
                  onChange={(evt) => setSelectedModel(evt.target.value)}
                >
                  <option value="">Select</option>
                  {modelOptions.map((model) => (
                    <option
                      key={model.name}
                      value={model.name}
                      disabled={model.available === false}
                    >
                      {model.available === false
                        ? `${model.name} (unavailable)`
                        : model.name}
                    </option>
                  ))}
                </select>
              </label>
              <p className="helper">
                {modelHelperText}
              </p>
              <div className="grid-two">
                <label className="input-group">
                  <span>Horizon</span>
                  <input
                    type="number"
                    min={1}
                    value={horizon}
                    onChange={(evt) => setHorizon(Number(evt.target.value))}
                  />
                </label>
                <label className="input-group">
                  <span>Device</span>
                  <select
                    value={device}
                    onChange={(evt) => setDevice(evt.target.value)}
                  >
                    <option value="cpu">cpu</option>
                    <option value="cuda">cuda</option>
                  </select>
                </label>
              </div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={allowDegrade}
                  onChange={(evt) => setAllowDegrade(evt.target.checked)}
                />
                <span>Allow baseline fallback</span>
              </label>
            </div>

            <div className="card actions" data-animate>
              <div>
                <h3>3. Run forecast</h3>
                <p className="helper">
                  Train, evaluate, and generate forecasts in one shot.
                </p>
              </div>
              <div className="action-row">
                <button
                  className="primary"
                  onClick={handleRun}
                  disabled={runState === "running" || !file}
                >
                  {runState === "running" ? "Running..." : "Train & Forecast"}
                </button>
                <button
                  className="ghost"
                  onClick={() => {
                    setResult(null);
                    setRunError("");
                    setRunState("idle");
                  }}
                >
                  Reset
                </button>
              </div>
              <p className="helper">Runs via /train_file_sync.</p>
            </div>

            <div className="card hero-card" data-animate>
              <p className="hero-label">Live status</p>
              <div className="hero-metric">
                <div>
                  <p className="hero-value">{runBadge}</p>
                  <p className="hero-caption">Pipeline status</p>
                </div>
                <div>
                  <p className="hero-value">
                    {activeResult?.model_record?.stage || "candidate"}
                  </p>
                  <p className="hero-caption">Model stage</p>
                </div>
              </div>
              <div className="hero-meta">
                <div>
                  <p>Rows</p>
                  <span>{rowCount || "--"}</span>
                </div>
                <div>
                  <p>Columns</p>
                  <span>{columns.length || "--"}</span>
                </div>
                <div>
                  <p>Range</p>
                  <span>{timeRange}</span>
                </div>
                <div>
                  <p>Run ID</p>
                  <span>{activeRunId || "--"}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="card" data-animate>
              <div className="card-header">
                <h3>Latest Run</h3>
                <span className="pill">Snapshot</span>
              </div>
              <div className="summary-grid">
                <div>
                  <p>Run id</p>
                  <span>{latestRunId || "--"}</span>
                </div>
                <div>
                  <p>Model</p>
                  <span>{latestModelName}</span>
                </div>
              </div>
              {latestSummary.length ? (
                <div className="metric-grid">
                  {latestSummary.map((item) => (
                    <div key={item.label} className="metric">
                      <span>{item.label}</span>
                      <strong>{formatNumber(item.value)}</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="helper">No metrics available.</p>
              )}
            </div>

            <div className="card" data-animate>
              <div className="card-header">
                <h3>Run summary</h3>
                <span className={`pill status-${runState}`}>{runBadge}</span>
              </div>
              {runError && <div className="banner">{runError}</div>}
              {degraded && (
                <div className="banner warning">
                  Degraded prediction: {degradedReason || "baseline fallback"}
                </div>
              )}
              <div className="summary-grid">
                <div>
                  <p>Run id</p>
                  <span>
                    {activeRunId || activeResult?.model_record?.id || "--"}
                  </span>
                </div>
                <div>
                  <p>Model</p>
                  <span>{displayModelName}</span>
                </div>
                <div>
                  <p>Stage</p>
                  <span>{activeResult?.model_record?.stage || "--"}</span>
                </div>
                <div>
                  <p>Version</p>
                  <span>{activeResult?.model_record?.version || "--"}</span>
                </div>
              </div>
            </div>

            <div className="card" data-animate>
              <div className="card-header">
                <h3>Metrics</h3>
                <span className="pill">Validation / Test</span>
              </div>
              <div className="metrics-grid">
                <MetricBlock title="Validation" metrics={valMetrics} />
                <MetricBlock title="Test" metrics={testMetrics} />
              </div>
            </div>

            <div className="card" data-animate>
              <div className="card-header">
                <h3>Forecast plots</h3>
                <span className="pill">Actual vs Pred</span>
              </div>
              <div className="chart-grid">
                <Suspense fallback={<ChartSkeleton title="Train" />}>
                  <ChartPanel title="Train" data={trainData} />
                </Suspense>
                <Suspense fallback={<ChartSkeleton title="Validation" />}>
                  <ChartPanel title="Validation" data={valData} />
                </Suspense>
                <Suspense fallback={<ChartSkeleton title="Test" />}>
                  <ChartPanel title="Test" data={testData} />
                </Suspense>
              </div>
            </div>

            <div className="card" data-animate>
              <div className="card-header">
                <h3>Artifacts</h3>
                <span className="pill">Paths</span>
              </div>
              {artifactEntries.length ? (
                <ul className="artifact-list">
                  {artifactEntries.map(([key, value]) => (
                    <li key={key}>
                      <span>{key}</span>
                      <span>{formatArtifactValue(value)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="helper">Artifacts will appear after a run.</p>
              )}
            </div>

            <div className="card" data-animate>
              <div className="card-header">
                <h3>Leaderboard</h3>
                <span className="pill">Models</span>
              </div>
              {leaderboard.length ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {Object.keys(leaderboard[0]).map((key) => (
                          <th key={key}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {leaderboard.map((row, idx) => (
                        <tr key={idx}>
                          {Object.keys(leaderboard[0]).map((key) => (
                            <td key={key}>{String((row as any)[key] ?? "")}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="helper">Leaderboard will appear after a run.</p>
              )}
              {(leaderboardPath || reportPath) && (
                <div className="helper">
                  {leaderboardPath && (
                    <div>
                      Leaderboard:{" "}
                      {leaderboardUrl ? (
                        <a href={leaderboardUrl} target="_blank" rel="noreferrer">
                          {leaderboardPath}
                        </a>
                      ) : (
                        leaderboardPath
                      )}
                    </div>
                  )}
                  {reportPath && (
                    <div>
                      Report:{" "}
                      {reportUrl ? (
                        <a href={reportUrl} target="_blank" rel="noreferrer">
                          {reportPath}
                        </a>
                      ) : (
                        reportPath
                      )}
                    </div>
                  )}
                </div>
              )}
              {drift && (
                <div className="helper">
                  Drift: {JSON.stringify(drift)}
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function MetricBlock({
  title,
  metrics,
}: {
  title: string;
  metrics?: Record<string, number> | null;
}) {
  const entries = metrics ? Object.entries(metrics) : [];
  return (
    <div className="metric-block">
      <p className="metric-title">{title}</p>
      {entries.length ? (
        <div className="metric-grid">
          {entries.map(([key, value]) => (
            <div key={key} className="metric">
              <span>{key}</span>
              <strong>{formatNumber(value)}</strong>
            </div>
          ))}
        </div>
      ) : (
        <p className="helper">No metrics available.</p>
      )}
    </div>
  );
}

function ChartSkeleton({ title }: { title: string }) {
  return (
    <div className="chart-card">
      <p className="chart-title">{title}</p>
      <p className="helper">Loading chart module...</p>
    </div>
  );
}
