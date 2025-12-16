import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict

# --- Helper: Convert any datetime-like scalar to Matplotlib float ---
from datetime import datetime
from typing import Any

def _to_num_scalar(ts: Any) -> float:
    """Convert a single datetime-like value to a Matplotlib date float.
    Accepts Timestamp/str/py-datetime/numpy scalar/Series/Index and returns float.
    NOTE: If a sequence is passed accidentally, we take its first element to keep type-checkers happy.
    """
    # Normalize potential containers to a scalar
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0] if len(ts) else pd.NaT
    elif isinstance(ts, pd.Index):
        ts = ts[0] if len(ts) else pd.NaT
    elif isinstance(ts, (np.ndarray, list, tuple)):
        ts = ts[0] if len(ts) else pd.NaT

    # Coerce to pandas Timestamp using a one-element object Series to please type checkers
    try:
        # Build a one-element object Series and coerce with errors='coerce' to avoid overload warnings
        ts_ser = pd.Series([ts], dtype="object")
        ts_coerced = pd.to_datetime(ts_ser, errors="coerce").iloc[0]

        # Handle numpy datetime64 scalar -> python datetime
        if hasattr(ts_coerced, "item") and not isinstance(ts_coerced, (pd.Timestamp, datetime)):
            try:
                ts_coerced = ts_coerced.item()
            except Exception:
                pass

        if isinstance(ts_coerced, pd.Timestamp):
            dt = ts_coerced.to_pydatetime()
        elif isinstance(ts_coerced, datetime):
            dt = ts_coerced
        else:
            return float("nan")

        return float(mdates.date2num(dt))
    except Exception:
        return float("nan")

def plot_results(
    train_df: pd.DataFrame,
    val_df_aligned: Optional[pd.DataFrame] = None,
    test_df_aligned: Optional[pd.DataFrame] = None,
    time_col: str = 'date',
    value_col: str = 'value',
    title: str = 'Time Series Forecast Results',
    payload: Optional[dict] = None,
    val_long: Optional[Dict] = None,
    test_long: Optional[Dict] = None,
    train_len: Optional[int] = None,
    val_len: Optional[int] = None,
    test_len: Optional[int] = None,
    split_lengths: Optional[Dict[str, int]] = None,
):
    """
    统一在一张图中绘制 5 条线：Train(True)、Val(True/Pred)、Test(True/Pred)。
    可视 6/2/2：按时间将真值分为 Train/Val/Test 三段分别着色；验证/测试预测画整段。

    Args:
        train_df (pd.DataFrame): 包含原始训练数据的 DataFrame。
        val_df_aligned (Optional[pd.DataFrame]): 包含对齐后验证集 y_true 和 yhat 的 DataFrame。
        test_df_aligned (Optional[pd.DataFrame]): 包含对齐后测试集 y_true 和 yhat 的 DataFrame。
        time_col (str): 时间列的名称。
        value_col (str): 目标值列的名称。
        title (str): 图表标题。
        payload (Optional[dict]): 额外数据容器。优先支持 "val_dense"/"test_dense"（DataFrame，索引为时间，列含 y_true/yhat），
                                  其次兼容旧的列式 JSON: {"timestamps","y_true","yhat"}。
        val_long (Optional[dict]): 验证集的“长序列”列式 JSON（keys: "timestamps", "y_true", "yhat"）。若提供，优先使用。
        test_long (Optional[dict]): 测试集的“长序列”列式 JSON（keys: "timestamps", "y_true", "yhat"）。若提供，优先使用。
        train_len (Optional[int]): 如果提供，则按精确计数切分 6/2/2。
        val_len (Optional[int]): 如果提供，则按精确计数切分 6/2/2。
        test_len (Optional[int]): 如果提供，则按精确计数切分 6/2/2。
        split_lengths (Optional[Dict[str,int]]): 若提供，读取 {"train":int,"val":int,"test":int} 作为 6/2/2 精确切分的计数（当 train_len/val_len/test_len 未指定时作为默认来源）。
    
    Returns:
        matplotlib.figure.Figure: 返回一个 figure 对象，以便 Streamlit 可以使用 st.pyplot() 展示。
    标签统一为 training_true、validation_true、validation_predict、test_true、test_predict；当 train_df 不含 time_col 时使用其索引作为时间轴。
    """
    # ---- 统一 6/2/2 计数优先级：函数参数 > split_lengths > 自动推断 ----
    if (train_len is None or val_len is None or test_len is None) and isinstance(split_lengths, dict):
        try:
            tl = int(split_lengths.get("train", 0))
            vl = int(split_lengths.get("val", 0))
            xl = int(split_lengths.get("test", 0))
            # 只有当三者都为正数时才采用
            if tl > 0 and vl > 0 and xl > 0:
                train_len, val_len, test_len = tl, vl, xl
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(18, 6))
    # --- NEW: Continuous-first plotting using pipeline-provided series (robust) ---
    did_continuous = False
    try:
        def _as_series_strict(x):
            """Return x as pd.Series if possible, else None."""
            if isinstance(x, pd.Series):
                return x
            if isinstance(x, (list, tuple, np.ndarray)):
                try:
                    return pd.Series(x)
                except Exception:
                    return None
            if isinstance(x, pd.DataFrame) and x.shape[1] >= 1:
                return x.iloc[:, 0]
            return None

        full_truth = None
        full_pred  = None
        phase_mask = None

        # 1) Preferred path: payload already provides continuous series
        if isinstance(payload, dict) and ("full_truth" in payload) and ("full_pred_cont" in payload):
            full_truth = _as_series_strict(payload.get("full_truth"))
            full_pred  = _as_series_strict(payload.get("full_pred_cont"))
            phase_mask = payload.get("phase_mask") if isinstance(payload.get("phase_mask"), pd.DataFrame) else None

        # 2) Fallback: build continuous series from inputs (train_df + dense val/test)
        if (full_truth is None) or (full_pred is None):
            try:
                # train truth
                if value_col in train_df.columns:
                    if time_col in train_df.columns:
                        tr_idx = pd.to_datetime(train_df[time_col], errors="coerce")
                    else:
                        tr_idx = pd.to_datetime(train_df.index, errors="coerce")
                    tr_true = pd.Series(pd.to_numeric(train_df[value_col], errors="coerce").values, index=tr_idx)
                else:
                    tr_true = pd.Series(dtype=float)

                def _pick(df):
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        idx = pd.to_datetime(df.index, errors="coerce")
                        t   = pd.to_numeric(df.get("y_true", pd.Series(index=df.index, dtype=float)), errors="coerce")
                        p   = pd.to_numeric(df.get("yhat",  pd.Series(index=df.index, dtype=float)), errors="coerce")
                        t = pd.Series(t.values, index=idx)
                        p = pd.Series(p.values, index=idx)
                        return t.dropna(), p.dropna()
                    return pd.Series(dtype=float), pd.Series(dtype=float)

                v_t, v_p = _pick(payload.get("val_dense") if isinstance(payload, dict) else None)
                t_t, t_p = _pick(payload.get("test_dense") if isinstance(payload, dict) else None)
                # If not in payload, try parameters
                if v_t.empty and isinstance(val_df_aligned, pd.DataFrame):
                    v_t, v_p = _pick(val_df_aligned)
                if t_t.empty and isinstance(test_df_aligned, pd.DataFrame):
                    t_t, t_p = _pick(test_df_aligned)

                # Build continuous truth
                pieces_truth = [s for s in [tr_true, v_t, t_t] if isinstance(s, pd.Series) and len(s)]
                if len(pieces_truth):
                    full_truth = pd.concat(pieces_truth).sort_index()
                    full_truth = full_truth[~full_truth.index.duplicated(keep="last")]

                # Build continuous pred (add splice at training end)
                splice = None
                if len(tr_true):
                    splice = pd.Series([float(tr_true.iloc[-1])], index=[tr_true.index.max()])
                pieces_pred = [s for s in [splice, v_p, t_p] if isinstance(s, pd.Series) and len(s)]
                if len(pieces_pred):
                    full_pred = pd.concat(pieces_pred).sort_index()
                    full_pred = full_pred[~full_pred.index.duplicated(keep="last")]

                # Phase mask for background spans
                if isinstance(full_truth, pd.Series) and len(full_truth):
                    tl = full_truth.index
                    phase_mask = pd.DataFrame(index=tl, data={"is_train": False, "is_val": False, "is_test": False})
                    if len(tr_true):
                        t_end = tr_true.index.max()
                        phase_mask.loc[phase_mask.index <= t_end, "is_train"] = True
                    if len(v_t):
                        v_end = v_t.index.max()
                        if len(tr_true):
                            phase_mask.loc[(phase_mask.index > tr_true.index.max()) & (phase_mask.index <= v_end), "is_val"] = True
                    if len(t_t):
                        last_val_end = v_t.index.max() if len(v_t) else (tr_true.index.max() if len(tr_true) else None)
                        if last_val_end is not None:
                            phase_mask.loc[phase_mask.index > last_val_end, "is_test"] = True
            except Exception:
                pass

        # 3) If we now have both, draw and return early
        if isinstance(full_truth, pd.Series) and len(full_truth) and isinstance(full_pred, pd.Series) and len(full_pred):
            truth_x = pd.to_datetime(pd.Index(full_truth.index)).to_pydatetime()
            truth_y = np.asarray(full_truth.values, dtype=float)
            pred_x  = pd.to_datetime(pd.Index(full_pred.index)).to_pydatetime()
            pred_y  = np.asarray(full_pred.values, dtype=float)
            ax.plot(truth_x, truth_y, label="truth",   color="#2fa3f6", linewidth=2)
            ax.plot(pred_x,  pred_y,  label="predict", color="#8419e8", linestyle="--", linewidth=2)

            # optional spans
            if isinstance(phase_mask, pd.DataFrame) and len(phase_mask.index):
                try:
                    t0 = pd.to_datetime(pd.Series(full_truth.index)).min()
                    tend = pd.to_datetime(pd.Series(full_truth.index)).max()
                    t_train_end = None
                    if "is_train" in phase_mask.columns and phase_mask["is_train"].any():
                        t_train_end = pd.to_datetime(phase_mask.index[phase_mask["is_train"]]).max()
                        ax.axvspan(_to_num_scalar(t0), _to_num_scalar(t_train_end), facecolor="#50a6e3", alpha=0.05, lw=0)
                    if "is_val" in phase_mask.columns and phase_mask["is_val"].any():
                        t_val_end = pd.to_datetime(phase_mask.index[phase_mask["is_val"]]).max()
                        if t_train_end is not None:
                            ax.axvspan(_to_num_scalar(t_train_end), _to_num_scalar(t_val_end), facecolor="#be5149a9", lw=0)
                        ax.axvspan(_to_num_scalar(t_val_end), _to_num_scalar(tend), facecolor="#26e1262b", lw=0)
                except Exception:
                    pass

            # x range & cosmetics
            try:
                min_ts = min(pd.to_datetime(pd.Series(full_truth.index)).min(), pd.to_datetime(pd.Series(full_pred.index)).min())
                max_ts = max(pd.to_datetime(pd.Series(full_truth.index)).max(), pd.to_datetime(pd.Series(full_pred.index)).max())
                if pd.notna(min_ts) and pd.notna(max_ts):
                    ax.set_xlim(_to_num_scalar(min_ts), _to_num_scalar(max_ts))
            except Exception:
                pass
            ax.margins(x=0.01)

            ax.set_title(title if isinstance(title, str) else "Time Series Forecast Results (Continuous)", fontsize=18, weight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=10)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            try:
                os.makedirs("artifacts/plots", exist_ok=True)
                fig.savefig("artifacts/plots/full_span.png", dpi=200, bbox_inches="tight")
            except Exception:
                pass

            did_continuous = True
            return fig
    except Exception as _e:
        print(f"[plot.py][continuous-first] fallback due to: {type(_e).__name__}: {_e}")
    all_ts_for_xlim: list[pd.Series] = []

    def _thin_xy(ts: pd.Series, y: pd.Series, max_points: int = 4000):
        if ts is None or y is None or len(ts) != len(y):
            return ts, y
        if len(ts) <= max_points:
            return ts, y
        step = max(1, len(ts) // max_points)
        return ts.iloc[::step], y.iloc[::step]

    def _to_dt(s: pd.Series):
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.to_datetime(s.astype(str), errors='coerce')

    def _as_series(x):
        """Ensure DatetimeIndex/Index becomes a Series for type consistency."""
        if isinstance(x, (pd.Index, pd.DatetimeIndex)):
            return pd.Series(x)
        return x

    # def _safe_series_from_df(df: Optional[pd.DataFrame], col: str) -> pd.Series:
    #     """Return a Series df[col] if present, else an empty float Series with df's index."""
    #     if isinstance(df, pd.DataFrame):
    #         if col in df.columns:
    #             s = df[col]
    #             if isinstance(s, pd.Series):
    #                 return s
    #         # fallback to empty Series aligned to index
    #         try:
    #             return pd.Series(index=df.index, dtype=float)
    #         except Exception:
    #             pass
    #     return pd.Series(dtype=float)

    # def _pick_truth_pred(df: Optional[pd.DataFrame]):
    #     """
    #     在 df 里查找统一的真值列 'y_true' 和预测列 'yhat'。
    #     仅支持列名为 'y_true' 和 'yhat'，返回 (truth_series or None, pred_series or None)
    #     """
    #     if not isinstance(df, pd.DataFrame) or df.empty:
    #         return None, None

    #     truth_candidates = ['y_true']
    #     pred_candidates  = ['yhat']

    #     t = None
    #     p = None
    #     for c in truth_candidates:
    #         if c in df.columns:
    #             t = pd.to_numeric(df[c], errors='coerce')
    #             break
    #     for c in pred_candidates:
    #         if c in df.columns:
    #             p = pd.to_numeric(df[c], errors='coerce')
    #             break
    #     return t, p

    def _select_truth_pred_columns(df: Optional[pd.DataFrame]):
        """
        仅支持统一列名 y_true / yhat。返回 (truth_series or None, pred_series or None)，均为数值化后的 Series。
        若不存在这两个列，则返回 (None, None)。
        """
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "y_true" in df.columns and "yhat" in df.columns:
                t = pd.to_numeric(df["y_true"], errors="coerce")
                p = pd.to_numeric(df["yhat"], errors="coerce")
                return t, p
        return None, None

    # 计算 val/test 起始时间（优先 long，其次 aligned df）
    val_start_ts = None
    test_start_ts = None

    # 从 payload 中提取 dense 结果（整段 1-step），优先用于绘图
    val_dense = None
    test_dense = None
    if isinstance(payload, dict):
        if isinstance(payload.get("val_dense"), pd.DataFrame) and not payload["val_dense"].empty:
            val_dense = payload["val_dense"]
        if isinstance(payload.get("test_dense"), pd.DataFrame) and not payload["test_dense"].empty:
            test_dense = payload["test_dense"]

    # 若提供了 dense，则用其最早时间更新起点
    if val_dense is not None:
        try:
            _vts2 = pd.to_datetime(val_dense.index)
            if len(_vts2) > 0:
                vt0 = pd.Series(_vts2).min()
                val_start_ts = vt0 if val_start_ts is None else min(val_start_ts, vt0)
        except Exception:
            pass
    if test_dense is not None:
        try:
            _tts2 = pd.to_datetime(test_dense.index)
            if len(_tts2) > 0:
                tt0 = pd.Series(_tts2).min()
                test_start_ts = tt0 if test_start_ts is None else min(test_start_ts, tt0)
        except Exception:
            pass

    if isinstance(val_long, dict):
        _vts = pd.to_datetime(pd.Series(val_long.get("timestamps", []), dtype="object"), errors='coerce')
        if len(_vts) > 0:
            val_start_ts = _vts.min()
    elif val_df_aligned is not None and not val_df_aligned.empty:
        try:
            _idx = pd.to_datetime(val_df_aligned.index)
        except Exception:
            _idx = pd.to_datetime(pd.Series(val_df_aligned.index.astype(str)), errors='coerce')
        if len(_idx) > 0:
            val_start_ts = pd.Series(_idx).min()

    if isinstance(test_long, dict):
        _tts = pd.to_datetime(pd.Series(test_long.get("timestamps", []), dtype="object"), errors='coerce')
        if len(_tts) > 0:
            test_start_ts = _tts.min()
    elif test_df_aligned is not None and not test_df_aligned.empty:
        try:
            _tidx = pd.to_datetime(test_df_aligned.index)
        except Exception:
            _tidx = pd.to_datetime(pd.Series(test_df_aligned.index.astype(str)), errors='coerce')
        if len(_tidx) > 0:
            test_start_ts = pd.Series(_tidx).min()

    # 如果 val_long/test_long 提供了 y_true，优先用它们绘制对应真值（更精确对齐）
    val_true_plotted = False
    test_true_plotted = False
    if isinstance(val_long, dict) and "timestamps" in val_long and "y_true" in val_long:
        try:
            v_ts_true = pd.to_datetime(pd.Series(val_long.get("timestamps", []), dtype="object"), errors="coerce")
            v_y_true = pd.to_numeric(pd.Series(val_long.get("y_true", []), dtype="float"), errors="coerce")
            m = ~v_ts_true.isna() & ~v_y_true.isna()
            if m.any():
                v_ts_true, v_y_true = v_ts_true[m], v_y_true[m]
                ax.plot(v_ts_true, v_y_true, label="validation_true", color="darkorange", linewidth=2, alpha=0.9)
                all_ts_for_xlim.append(pd.Series(v_ts_true))
                val_true_plotted = True
        except Exception:
            pass
    if isinstance(test_long, dict) and "timestamps" in test_long and "y_true" in test_long:
        try:
            t_ts_true = pd.to_datetime(pd.Series(test_long.get("timestamps", []), dtype="object"), errors="coerce")
            t_y_true = pd.to_numeric(pd.Series(test_long.get("y_true", []), dtype="float"), errors="coerce")
            m = ~t_ts_true.isna() & ~t_y_true.isna()
            if m.any():
                t_ts_true, t_y_true = t_ts_true[m], t_y_true[m]
                ax.plot(t_ts_true, t_y_true, label="test_true", color="green", linewidth=2, alpha=0.9)
                all_ts_for_xlim.append(pd.Series(t_ts_true))
                test_true_plotted = True
        except Exception:
            pass

    # 如果有 dense 结果且尚未绘制对应真值，则优先用 dense 的 y_true（保证连续）
    if (val_dense is not None) and (not val_true_plotted):
        try:
            vts = pd.to_datetime(val_dense.index)
            t_series, _ = _select_truth_pred_columns(val_dense)
            if t_series is not None:
                vy = t_series
                m = ~vts.isna() & ~vy.isna()
                if m.any():
                    ax.plot(vts[m], vy[m], label="validation_true", color="darkorange", linewidth=2, alpha=0.9)
                    all_ts_for_xlim.append(pd.Series(vts[m]))
                    val_true_plotted = True
        except Exception:
            pass
    if (test_dense is not None) and (not test_true_plotted):
        try:
            tts = pd.to_datetime(test_dense.index)
            t_series, _ = _select_truth_pred_columns(test_dense)
            if t_series is not None:
                ty = t_series
                m = ~tts.isna() & ~ty.isna()
                if m.any():
                    ax.plot(tts[m], ty[m], label="test_true", color="green", linewidth=2, alpha=0.9)
                    all_ts_for_xlim.append(pd.Series(tts[m]))
                    test_true_plotted = True
        except Exception:
            pass

    # 若仍未绘制真值且提供了对齐DF，则用其 y_true 作为真值（保证有5条线）
    if (val_df_aligned is not None) and (not val_true_plotted) and not val_df_aligned.empty:
        try:
            vts = pd.to_datetime(val_df_aligned.index)
            vy, _ = _select_truth_pred_columns(val_df_aligned)
            if vy is not None:
                m = ~vts.isna() & ~vy.isna()
                if m.any():
                    ax.plot(vts[m], vy[m], label="validation_true", color="darkorange", linewidth=2, alpha=0.9)
                    all_ts_for_xlim.append(pd.Series(vts[m]))
                    val_true_plotted = True
        except Exception:
            pass
    if (test_df_aligned is not None) and (not test_true_plotted) and not test_df_aligned.empty:
        try:
            tts = pd.to_datetime(test_df_aligned.index)
            ty, _ = _select_truth_pred_columns(test_df_aligned)
            if ty is not None:
                m = ~tts.isna() & ~ty.isna()
                if m.any():
                    ax.plot(tts[m], ty[m], label="test_true", color="green", linewidth=2, alpha=0.9)
                    all_ts_for_xlim.append(pd.Series(tts[m]))
                    test_true_plotted = True
        except Exception:
            pass

    # 1) 真值三段：Train(True) / Val(True) / Test(True)
    if value_col in train_df.columns:
        if time_col in train_df.columns:
            full_ts = _to_dt(train_df[time_col])
        else:
            full_ts = _to_dt(train_df.index.to_series())
        full_y = pd.to_numeric(train_df[value_col], errors='coerce')
        if len(full_ts) == len(full_y) and len(full_ts) > 0:
            # have_counts 代表是否有精确的 6/2/2 计数信息（来自参数或 split_lengths）
            have_counts = (train_len is not None and val_len is not None and test_len is not None)
            if have_counts:
                # Cast Optionals to plain ints for static type checkers and safe arithmetic
                tlen: int = max(0, int(train_len))   # type: ignore[arg-type]
                vlen: int = max(0, int(val_len))     # type: ignore[arg-type]
                xlen: int = max(0, int(test_len))    # type: ignore[arg-type]

                total = len(full_ts)
                # If counts don't sum to total, push the remainder to the test split to keep 6/2/2 feel.
                s = tlen + vlen + xlen
                if s != total:
                    xlen = max(0, total - (tlen + vlen))

                # Clip to valid ranges to avoid negative/overflow masks.
                tlen = min(tlen, total)
                vlen = min(vlen, max(0, total - tlen))
                xlen = max(0, total - (tlen + vlen))

                idx = np.arange(total, dtype=int)
                train_mask = idx < tlen
                val_mask = (idx >= tlen) & (idx < tlen + vlen)
                test_mask = idx >= (tlen + vlen)
            else:
                if val_start_ts is not None:
                    train_mask = full_ts < val_start_ts
                else:
                    train_mask = pd.Series([True] * len(full_ts))
                val_mask = pd.Series([False] * len(full_ts))
                test_mask = pd.Series([False] * len(full_ts))
                if val_start_ts is not None and test_start_ts is not None:
                    val_mask = (full_ts >= val_start_ts) & (full_ts < test_start_ts)
                    test_mask = full_ts >= test_start_ts
                elif val_start_ts is not None and test_start_ts is None:
                    val_mask = full_ts >= val_start_ts
                elif val_start_ts is None and test_start_ts is not None:
                    test_mask = full_ts >= test_start_ts

            # 使用已转好的 full_ts/full_y 作为横纵坐标，避免字符串被当成类别轴
            tr_ts, tr_y = _thin_xy(full_ts[train_mask], full_y[train_mask])
            if tr_ts is not None and len(tr_ts) > 0:
                ax.plot(tr_ts, tr_y, label='training_true', color='dodgerblue', linewidth=2)
                all_ts_for_xlim.append(_as_series(tr_ts))
            va_ts, va_y = _thin_xy(full_ts[val_mask], full_y[val_mask])
            if (va_ts is not None and len(va_ts) > 0) and not val_true_plotted:
                ax.plot(va_ts, va_y, label='validation_true', color='darkorange', linewidth=2)
            te_ts, te_y = _thin_xy(full_ts[test_mask], full_y[test_mask])
            if (te_ts is not None and len(te_ts) > 0) and not test_true_plotted:
                ax.plot(te_ts, te_y, label='test_true', color='green', linewidth=2)
                all_ts_for_xlim.append(_as_series(te_ts))

    # 2) 验证集预测（整段）
    # 优先使用 dense 的预测（连续 1-step）
    if val_dense is not None:
        try:
            _, p_series = _select_truth_pred_columns(val_dense)
            if p_series is not None:
                vts = pd.to_datetime(val_dense.index)
                vhat = p_series
                m = ~vts.isna() & ~vhat.isna()
                if m.any():
                    ax.plot(vts[m], vhat[m], label='validation_predict', color='red', linestyle='--')
                    all_ts_for_xlim.append(pd.Series(vts[m]))
        except Exception:
            pass
    elif isinstance(val_long, dict):
        try:
            v_ts = pd.to_datetime(pd.Series(val_long.get("timestamps", []), dtype="object"), errors='coerce')
            v_hat = pd.to_numeric(pd.Series(val_long.get("yhat", []), dtype="float"), errors='coerce')
            valid_ts = ~v_ts.isna()
            m = valid_ts & ~v_hat.isna()
            if m.any():
                ax.plot(v_ts[m], v_hat[m], label='validation_predict', color='red', linestyle='--')
                all_ts_for_xlim.append(_as_series(v_ts[valid_ts]))
        except Exception:
            pass
    elif val_df_aligned is not None and not val_df_aligned.empty:
        _, p_series = _select_truth_pred_columns(val_df_aligned)
        if p_series is not None:
            vts = pd.to_datetime(val_df_aligned.index)
            vhat = p_series
            mv = ~vhat.isna()
            if mv.any():
                ax.plot(vts[mv], vhat[mv], label='validation_predict', color='red', linestyle='--')
                all_ts_for_xlim.append(_as_series(vts[mv]))
    elif isinstance(payload, dict) and all(k in payload for k in ("timestamps", "y_true", "yhat")):
        try:
            v_ts = pd.to_datetime(pd.Series(payload.get("timestamps", []), dtype="object"), errors='coerce')
            v_hat = pd.to_numeric(pd.Series(payload.get("yhat", []), dtype="float"), errors='coerce')
            m = ~v_hat.isna()
            if m.any():
                ax.plot(v_ts[m], v_hat[m], label='validation_predict', color='red', linestyle='--')
                all_ts_for_xlim.append(_as_series(v_ts[m]))
        except Exception:
            pass

    # 3) 测试集预测（整段）
    # 优先使用 dense 的预测（连续 1-step）
    if test_dense is not None:
        try:
            _, p_series_t = _select_truth_pred_columns(test_dense)
            if p_series_t is not None:
                tts  = pd.to_datetime(test_dense.index)
                that = p_series_t
                m = ~tts.isna() & ~that.isna()
                if m.any():
                    ax.plot(tts[m], that[m], label='test_predict', color='purple', linestyle='--')
                    all_ts_for_xlim.append(pd.Series(tts[m]))
        except Exception:
            pass
    elif isinstance(test_long, dict):
        try:
            t_ts = pd.to_datetime(pd.Series(test_long.get("timestamps", []), dtype="object"), errors='coerce')
            t_hat = pd.to_numeric(pd.Series(test_long.get("yhat", []), dtype="float"), errors='coerce')
            valid_ts_t = ~t_ts.isna()
            m = valid_ts_t & ~t_hat.isna()
            if m.any():
                ax.plot(t_ts[m], t_hat[m], label='test_predict', color='purple', linestyle='--')
                all_ts_for_xlim.append(_as_series(t_ts[valid_ts_t]))
        except Exception:
            pass
    elif test_df_aligned is not None and not test_df_aligned.empty:
        _, p_series_t = _select_truth_pred_columns(test_df_aligned)
        if p_series_t is not None:
            tts = pd.to_datetime(test_df_aligned.index)
            that = p_series_t
            mt = ~that.isna()
            if mt.any():
                ax.plot(tts[mt], that[mt], label='test_predict', color='purple', linestyle='--')
                all_ts_for_xlim.append(_as_series(tts[mt]))

    # 统一 X 轴范围，防止出现左侧挤压/右侧空白
    valid_ts_segments = []
    for s in all_ts_for_xlim:
        try:
            if s is None:
                continue
            s2 = pd.to_datetime(pd.Series(s), errors="coerce").dropna()
            if len(s2) > 0:
                valid_ts_segments.append(s2)
        except Exception:
            continue
    if len(valid_ts_segments) > 0:
        try:
            min_ts = min(seg.min() for seg in valid_ts_segments)
            max_ts = max(seg.max() for seg in valid_ts_segments)
            if pd.notna(min_ts) and pd.notna(max_ts):
                ax.set_xlim(_to_num_scalar(min_ts), _to_num_scalar(max_ts))
        except Exception:
            pass
    ax.margins(x=0.01)
    try:
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])  # no-op but keeps explicit xlim set for some backends
    except Exception:
        pass

    # --- 美化图表 ---
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc='upper left', ncol=3, frameon=False, fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 去重图例标签（避免长序列与切分重复）
    try:
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen[l] = True
            new_h.append(h); new_l.append(l)
        ax.legend(new_h, new_l, loc='upper left', ncol=3, frameon=False, fontsize=10)
    except Exception:
        pass

    try:
        os.makedirs("artifacts/plots", exist_ok=True)
        fig.savefig("artifacts/plots/full_span.png", dpi=200, bbox_inches="tight")
    except Exception:
        pass
    
    return fig
