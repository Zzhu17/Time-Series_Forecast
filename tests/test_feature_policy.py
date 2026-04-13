import numpy as np
import pandas as pd


from utils.feature_missing_policy import prepare_df_for_non_informer_models  # noqa: E402


def test_prepare_df_for_non_informer_models_drops_optional_and_recomputes():
    # Build a small dataframe with an optional column containing NaN and a recomputable lag feature.
    n = 60
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": np.linspace(0, 1, n),
            "feat_clean": np.random.randn(n),
            "feat_dirty": [1.0] * 10 + [np.nan] * 50,
        }
    )
    # recomputable lag feature
    df["lag_1"] = df["value"].shift(1)

    config = {
        "model_config": {
            "Informer": {
                "feature_selection": {
                    "missing_rate_threshold": 0.4,
                    "low_variance_threshold": 0.0,
                    "redundant_corr_threshold": 0.99,
                    "max_features": None,
                }
            }
        }
    }

    df_prep, feat_cols, prep_report = prepare_df_for_non_informer_models(
        df,
        time_col="date",
        value_col="value",
        candidate_cols=["value", "feat_clean", "feat_dirty", "lag_1"],
        config=config,
    )

    # Optional with NaN should be dropped
    assert "feat_dirty" not in feat_cols
    # Recomputable lag should be kept and contain no NaN after strict trim
    assert "lag_1" in feat_cols
    assert not df_prep[feat_cols].isna().any().any()
    # value must remain first
    assert feat_cols[0] == "value"
