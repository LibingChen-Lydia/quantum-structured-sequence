# Shared sequence data protocol

## Purpose

This document defines the shared sequence-data contract used by Module A and
Module C. It keeps the upstream data representation, preprocessing, temporal
indexing, and Patch construction consistent across both modules.

## Sequence tensor contract

The shared encoder accepts a batch of multivariate time series with shape:

```text
[batch, sequence length, channels]
```

Each input window contains observations available at its query time. The
encoder returns scale-indexed Patch representations with aligned Patch start
indices, amplitude, phase, frequency, trend, residual statistics, and
normalized retrieval embeddings. Module A consumes the structured periodic
features, and Module C consumes the retrieval embeddings produced by this same
encoder.

## Datasets and temporal splits

| Dataset | Feature setting | Training partition | Validation partition | Test partition |
|---|---|---|---|---|
| ETTh1 | Multivariate (`M`) | First 12 months | Next 4 months | Final 4 months |
| ETTm1 | Multivariate (`M`) | First 12 months | Next 4 months | Final 4 months |
| Weather | Multivariate (`M`) | First 70% | Next 10% | Final 20% |
| Traffic | Multivariate (`M`) | First 70% | Next 10% | Final 20% |

The implementation follows the corresponding Time-Series-Library loaders:
`Dataset_ETT_hour` for ETTh1, `Dataset_ETT_minute` for ETTm1, and
`Dataset_Custom` for Weather and Traffic. Long-term forecasting experiments use
an input window length of `seq_len=96`.

## Preprocessing contract

1. Read timestamps for temporal alignment and numerical columns as sequence
   features.
2. Fit one mean and standard deviation per feature from training rows, using
   `float64` accumulation.
3. Normalize each split with the fitted training statistics and store model
   inputs as `float32`.
4. Build validation and test windows with the preceding `seq_len`
   observations, following the Time-Series-Library split boundaries.
5. Preserve each window's input start, input end, and forecast start timestamp
   alongside its encoded representation.

## Patch construction

The shared encoder receives the normalized input window and applies the
configured Patch lengths and sampling period. Each scale preserves its Patch
start index relative to the input window, allowing Module A periodic
interactions and Module C historical-memory entries to reference the same
temporal locations.

```python
from qss.classical import StructuredEncoderConfig, StructuredSequenceEncoder

encoder = StructuredSequenceEncoder(
    StructuredEncoderConfig(patch_lengths=(24, 48, 96), period=24)
)
encoded = encoder.fit_transform(training_windows)
```

## Shared run metadata

Each run records the dataset filename, data-loader class, split boundaries,
feature setting, channel count, Patch configuration, normalization statistics,
random seed, package versions, device, and wall-clock duration.
