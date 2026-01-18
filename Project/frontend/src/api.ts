import type { ModelInfo, TrainResponse } from "./types";

export type TrainPayload = {
  file: File;
  modelName: string;
  timeCol: string;
  valueCol: string;
  horizon: number;
  featureCols?: string[] | null;
  allowDegrade?: boolean;
  device?: string;
  residualModeling?: Record<string, unknown> | null;
};

const buildUrl = (base: string, path: string) => {
  const trimmed = base.replace(/\/$/, "");
  return `${trimmed}${path}`;
};

const authHeaders = (token?: string) =>
  token ? { Authorization: `Bearer ${token}` } : {};

const parseJson = async (res: Response) => {
  const text = await res.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return { message: text };
  }
};

export async function fetchModels(
  apiBase: string,
  token?: string
): Promise<ModelInfo[]> {
  const res = await fetch(buildUrl(apiBase, "/models"), {
    headers: { ...authHeaders(token) },
  });
  const data = await parseJson(res);
  if (!res.ok) {
    const detail = data?.detail || data?.message || res.statusText;
    throw new Error(detail);
  }
  return Array.isArray(data) ? (data as ModelInfo[]) : [];
}

export async function trainFileSync(
  apiBase: string,
  payload: TrainPayload,
  token?: string
): Promise<TrainResponse> {
  const form = new FormData();
  form.append("file", payload.file);
  form.append("model_name", payload.modelName);
  form.append("time_col", payload.timeCol);
  form.append("value_col", payload.valueCol);
  form.append("horizon", String(payload.horizon));
  if (payload.featureCols && payload.featureCols.length) {
    form.append("feature_cols", JSON.stringify(payload.featureCols));
  }
  if (payload.allowDegrade) {
    form.append("allow_degrade", "true");
  }
  if (payload.device) {
    form.append("device", payload.device);
  }
  if (payload.residualModeling) {
    form.append("residual_modeling", JSON.stringify(payload.residualModeling));
  }

  const res = await fetch(buildUrl(apiBase, "/train_file_sync"), {
    method: "POST",
    body: form,
    headers: { ...authHeaders(token) },
  });
  const data = await parseJson(res);
  if (!res.ok) {
    const detail = data?.detail || data?.message || res.statusText;
    throw new Error(detail);
  }
  return (data || {}) as TrainResponse;
}

export async function fetchLatestArtifacts(
  apiBase: string,
  token?: string
): Promise<TrainResponse> {
  const res = await fetch(buildUrl(apiBase, "/artifacts/latest"), {
    headers: { ...authHeaders(token) },
  });
  const data = await parseJson(res);
  if (!res.ok) {
    const detail = data?.detail || data?.message || res.statusText;
    throw new Error(detail);
  }
  return (data || {}) as TrainResponse;
}
