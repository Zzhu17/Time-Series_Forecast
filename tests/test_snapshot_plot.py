import numpy as np


from services import snapshot  # noqa: E402


def test_pack_plot_series_truncates_and_aligns():
    # create a long series (5000) to trigger truncation at max_n=4000
    ts = np.arange(5000)
    y_true = np.linspace(0, 1, 5000)
    y_pred = y_true + 0.1

    out = snapshot.pack_plot_series(ts, y_true, y_pred, max_n=4000)

    assert out is not None
    assert len(out["ts"]) == 4000  # truncated
    assert len(out["true"]) == 4000
    assert len(out["pred"]) == 4000
    # last element should align with input tail
    assert float(out["true"][-1]) == float(y_true[-1])
    # ensure ordering preserved (first element is the earliest after truncation)
    assert float(out["true"][0]) == float(y_true[-4000])
