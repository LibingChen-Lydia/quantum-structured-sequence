import csv

import numpy as np

from qss.data.ett import ETTH1_TEST_END, ETTH1_TRAIN_END, ETTH1_VAL_END, load_etth1


def test_etth1_uses_time_series_library_boundaries(tmp_path) -> None:
    csv_path = tmp_path / "ETTh1.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "HUFL", "OT"])
        for hour in range(ETTH1_TEST_END):
            writer.writerow([f"2016-01-01 {hour:02}:00", hour, hour * 2])
    train = load_etth1(csv_path, "train", seq_len=96, pred_len=96)
    val = load_etth1(csv_path, "val", seq_len=96, pred_len=96)
    test = load_etth1(csv_path, "test", seq_len=96, pred_len=96)
    assert train.end_times[0] == 96
    assert val.end_times[0] == ETTH1_TRAIN_END
    assert test.end_times[0] == ETTH1_VAL_END
    assert train.inputs.shape[-1] == 2
    assert np.allclose(train.mean, [ETTH1_TRAIN_END / 2 - 0.5, ETTH1_TRAIN_END - 1])
