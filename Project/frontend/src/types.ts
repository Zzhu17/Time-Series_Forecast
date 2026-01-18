export type ModelInfo = {
  name: string;
  description: string;
  available?: boolean;
  missing_deps?: string[] | null;
};

export type PlotSeries = {
  ts: string[];
  true: number[];
  pred: number[];
};

export type PlotData = {
  train?: PlotSeries | null;
  val?: PlotSeries | null;
  test?: PlotSeries | null;
};

export type ModelRecord = {
  id: string;
  name: string;
  stage: string;
  version?: string | null;
  params?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  artifacts?: Record<string, unknown> | null;
  created_at?: string | null;
  updated_at?: string | null;
  promoted_at?: string | null;
};

export type TrainResponse = {
  status?: string;
  message?: string;
  metrics?: Record<string, Record<string, number>>;
  model_name?: string;
  data?: {
    plot_data?: PlotData;
    leaderboard?: Array<Record<string, unknown>>;
    leaderboard_path?: string;
    report_path?: string;
    drift?: Record<string, unknown>;
    degraded?: boolean;
    degraded_mode?: string;
    degraded_reason?: string;
    mean_abs_true_val?: number;
    mean_abs_true_test?: number;
  };
  artifacts?: Record<string, string>;
  model_record?: ModelRecord | null;
  task_model?: string;
  feature_cols?: string[] | null;
  run_id?: string;
};
