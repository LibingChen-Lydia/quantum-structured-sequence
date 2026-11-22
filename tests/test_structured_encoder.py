import numpy as np

from qss.classical.encoder import StructuredEncoderConfig, StructuredSequenceEncoder


def test_encoder_returns_multiscale_quantum_and_retrieval_outputs() -> None:
    time = np.arange(96)
    series = np.stack(
        [
            np.stack((np.cos(2 * np.pi * time / 24), 0.5 * np.cos(2 * np.pi * time / 24 + 0.3)), axis=-1),
            np.stack((0.8 * np.cos(2 * np.pi * time / 24 + 0.2), np.cos(2 * np.pi * time / 24)), axis=-1),
        ]
    )
    encoder = StructuredSequenceEncoder(StructuredEncoderConfig(patch_lengths=(24, 48), period=24))
    encoded = encoder.fit_transform(series)

    assert set(encoded.scales) == {24, 48}
    scale = encoded.scales[24]
    assert scale.features.shape == (2, 4, 16)
    assert scale.embedding.shape == (2, 4, 16)
    assert scale.amplitude.shape == (2, 4, 2)
    assert np.all(np.isfinite(scale.embedding))
    assert np.allclose(scale.frequency, 1 / 24)


def test_encoder_uses_training_statistics_at_transform_time() -> None:
    train = np.arange(96, dtype=float)[None, :, None]
    test = (np.arange(96, dtype=float) + 5)[None, :, None]
    encoder = StructuredSequenceEncoder(StructuredEncoderConfig(patch_lengths=(24,), period=24)).fit(train)
    encoded = encoder.transform(test)
    assert encoded.scales[24].embedding.shape == (1, 4, 8)
