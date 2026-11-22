"""Module C windows must be identical to Time-Series-Library's own Dataset windows."""

import numpy as np
import pandas as pd
import pytest

from qss.data import tsl_adapter

try:
    tsl_adapter.tsl_root()
    HAVE_TSL = True
except FileNotFoundError:
    HAVE_TSL = False

pytestmark = pytest.mark.skipif(not HAVE_TSL, reason="Time-Series-Library checkout not found")


def _have(name: str) -> bool:
    _, folder, filename, _, _ = tsl_adapter.DATASETS[name]
    return (tsl_adapter.tsl_root() / "dataset" / folder / filename).exists()


@pytest.mark.parametrize("name", ["ETTh1", "ETTm1", "weather"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_windows_match_tsl_getitem(name: str, split: str) -> None:
    if not _have(name):
        pytest.skip(f"{name} csv not downloaded")
    pred_len = 192
    ds = tsl_adapter.tsl_dataset(name, split, 96, 48, pred_len)
    out = tsl_adapter.load_tsl(name, split, 96, pred_len)
    assert len(out.inputs) == len(ds)
    for i in (0, len(ds) // 2, len(ds) - 1):
        seq_x, seq_y, _, _ = ds[i]
        np.testing.assert_allclose(out.inputs[i], seq_x, rtol=0, atol=1e-6)
        np.testing.assert_allclose(out.targets[i], seq_y[-pred_len:], rtol=0, atol=1e-6)
    # end_times index the global row of the first forecast step
    _, folder, filename, _, _ = tsl_adapter.DATASETS[name]
    raw = pd.read_csv(tsl_adapter.tsl_root() / "dataset" / folder / filename)
    full = ds.scaler.transform(raw[raw.columns[1:]].values)
    j = len(ds) // 3
    np.testing.assert_allclose(full[out.end_times[j]], out.targets[j][0], atol=1e-6)
    np.testing.assert_allclose(full[out.end_times[j] - 1], out.inputs[j][-1], atol=1e-6)


def test_etth1_matches_existing_loader() -> None:
    if not _have("ETTh1"):
        pytest.skip("ETTh1 csv not downloaded")
    from qss.data import load_etth1

    path = tsl_adapter.tsl_root() / "dataset" / "ETT-small" / "ETTh1.csv"
    ours = load_etth1(path, "test", pred_len=96)
    tsl = tsl_adapter.load_tsl("ETTh1", "test", 96, 96)
    np.testing.assert_array_equal(ours.end_times, tsl.end_times)
    np.testing.assert_allclose(ours.inputs, tsl.inputs, atol=1e-3)


def test_stride_subsamples_windows() -> None:
    if not _have("ETTh1"):
        pytest.skip("ETTh1 csv not downloaded")
    full = tsl_adapter.load_tsl("ETTh1", "train", 96, 96)
    sub = tsl_adapter.load_tsl("ETTh1", "train", 96, 96, stride=4)
    np.testing.assert_array_equal(sub.inputs, full.inputs[::4])
    np.testing.assert_array_equal(sub.end_times, full.end_times[::4])
