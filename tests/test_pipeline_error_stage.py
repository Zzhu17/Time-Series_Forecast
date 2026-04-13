import pandas as pd


from services import pipeline  # noqa: E402


def test_run_pipeline_preserves_inferred_error_stage(monkeypatch):
    def _raise(_config):
        raise ValueError("feature missing column during preprocess")

    monkeypatch.setattr(pipeline, "run_train_predict_pipeline", _raise)

    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.0]})
    out = pipeline.run_pipeline_and_update_state(
        df=df,
        config={"artifacts": {}, "data": {}},
        feature_cols=["value"],
        uploaded_name="sample.csv",
        model_name="informer",
        time_col="date",
        value_col="value",
        allow_degrade=False,
    )

    assert out["status"] == "error"
    assert out["error_stage"] == "data_prep"
    assert out["action"] == "fail"
