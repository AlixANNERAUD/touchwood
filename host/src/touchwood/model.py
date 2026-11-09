"""Fusion model: shared per-modality encoders, one fusion head.

The architecture was designed for MicroFlow's small operator set: Conv2D with (1, k) kernels
instead of Conv1D, AveragePooling2D over the full width instead of global pooling, ReLU only,
BatchNorm (folded at conversion). nanoglia, the engine used now, supports more (CONCATENATION,
ADD, MAX_POOL_2D, MEAN...; see root README §7), so these choices are no longer required.
Each encoder and the head are still separate models, glued together in Rust (concatenating the
embeddings, possibly received over BLE), which matches the two-board split.

Encoders are shared between boards (same weights for A and B), so both boards can run the
same firmware.
"""

from __future__ import annotations

import keras
from keras import layers

AUDIO_LEN = 800  # 50 ms @ 16 kHz
IMU_LEN = 100  # ~100 ms @ ~1 kHz
IMU_AXES = 6
AUDIO_EMB = 32
IMU_EMB = 16


def _conv_bn_relu(x, filters: int, k: int, stride: int = 1, name: str = ""):
    x = layers.Conv2D(filters, (1, k), strides=(1, stride), use_bias=True, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _dw_separable(x, filters: int, k: int, stride: int, name: str):
    # TensorFlow cannot train a DepthwiseConv2D with unequal strides such as (1, 2) on CPU,
    # so downsample with an average pooling instead (cheap).
    x = layers.DepthwiseConv2D((1, k), use_bias=True, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    if stride > 1:
        x = layers.AveragePooling2D((1, stride), name=f"{name}_down")(x)
    return _conv_bn_relu(x, filters, 1, name=f"{name}_pw")


def _global_pool(x, dim: int, name: str):
    """Global average pooling expressed as AveragePooling2D (lowers to AVERAGE_POOL_2D, not MEAN)."""
    x = layers.AveragePooling2D((1, x.shape[2]), name=f"{name}_pool")(x)
    return layers.Reshape((dim,), name=f"{name}_flat")(x)


def _scaled(n: int, width: float) -> int:
    return max(4, round(n * width))


def build_audio_encoder(width: float = 1.0, batch_size: int | None = None) -> keras.Model:
    """Raw waveform (1, AUDIO_LEN, 1) -> AUDIO_EMB-d embedding. A learned filterbank, no FFT."""
    inp = keras.Input((1, AUDIO_LEN, 1), batch_size=batch_size, name="audio")
    x = _conv_bn_relu(inp, _scaled(16, width), 9, stride=2, name="a1")
    x = _conv_bn_relu(x, _scaled(32, width), 5, stride=2, name="a2")
    x = _dw_separable(x, _scaled(32, width), 5, stride=2, name="a3")
    x = _dw_separable(x, AUDIO_EMB, 5, stride=2, name="a4")
    return keras.Model(inp, _global_pool(x, AUDIO_EMB, "a"), name="audio_encoder")


def build_imu_encoder(width: float = 1.0, batch_size: int | None = None) -> keras.Model:
    """IMU window (1, IMU_LEN, 6 axes) -> IMU_EMB-d embedding."""
    inp = keras.Input((1, IMU_LEN, IMU_AXES), batch_size=batch_size, name="imu")
    x = _conv_bn_relu(inp, _scaled(16, width), 5, name="i1")
    x = _conv_bn_relu(x, IMU_EMB, 5, stride=2, name="i2")
    return keras.Model(inp, _global_pool(x, IMU_EMB, "i"), name="imu_encoder")


def feature_dim(boards: tuple[str, ...], use_imu: bool) -> int:
    """Size of the head input: per board [audio embedding, log-energy, (imu embedding)]."""
    return len(boards) * (AUDIO_EMB + 1 + (IMU_EMB if use_imu else 0))


def build_head(in_dim: int, n_classes: int, hidden: int = 32, batch_size: int | None = None) -> keras.Model:
    inp = keras.Input((in_dim,), batch_size=batch_size, name="features")
    x = layers.Dropout(0.2, name="h_drop")(inp)
    x = layers.Dense(hidden, activation="relu", name="h_dense")(x)
    out = layers.Dense(n_classes, activation="softmax", name="h_out")(x)
    return keras.Model(inp, out, name="head")


@keras.saving.register_keras_serializable(package="touchwood")
class BranchDropout(layers.Layer):
    """Zero a whole branch (all its features) for a random subset of the batch, in training only.

    Teaches the head to cope with a missing modality or a missing board (late BLE packet),
    and prevents the dominant modality (likely the microphone) from starving the others.
    """

    def __init__(self, rate: float, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed = seed
        self.seed_gen = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if not training or self.rate <= 0:
            return x
        keep = keras.random.uniform((keras.ops.shape(x)[0], 1), seed=self.seed_gen) >= self.rate
        return x * keras.ops.cast(keep, x.dtype)

    def get_config(self):
        return {**super().get_config(), "rate": self.rate, "seed": self.seed}


def build_fusion_model(
    boards: tuple[str, ...] = ("A", "B"),
    use_imu: bool = True,
    n_classes: int = 9,
    width: float = 1.0,
    branch_dropout: float = 0.15,
) -> tuple[keras.Model, dict[str, keras.Model]]:
    """Training graph. Returns (model, parts) where parts holds the exportable sub-models.

    Inputs per board b: `audio_b` (1, AUDIO_LEN, 1), `loge_b` (1,), and `imu_b` if use_imu.
    Outputs: `zone` (fusion head) plus one auxiliary classifier per branch (`aux_audio_b`,
    `aux_imu_b`), used only as extra training signal.
    """
    audio_enc = build_audio_encoder(width)
    imu_enc = build_imu_encoder(width) if use_imu else None
    inputs, feats, outputs = {}, [], {}
    for i, b in enumerate(boards):
        inputs[f"audio_{b}"] = keras.Input((1, AUDIO_LEN, 1), name=f"audio_{b}")
        inputs[f"loge_{b}"] = keras.Input((1,), name=f"loge_{b}")
        za = layers.Concatenate(name=f"za_{b}")([audio_enc(inputs[f"audio_{b}"]), inputs[f"loge_{b}"]])
        za = BranchDropout(branch_dropout, seed=10 * i, name=f"drop_audio_{b}")(za)
        outputs[f"aux_audio_{b}"] = layers.Dense(n_classes, activation="softmax", name=f"aux_audio_{b}")(za)
        feats.append(za)
        if use_imu:
            inputs[f"imu_{b}"] = keras.Input((1, IMU_LEN, IMU_AXES), name=f"imu_{b}")
            zi = BranchDropout(branch_dropout, seed=10 * i + 1, name=f"drop_imu_{b}")(imu_enc(inputs[f"imu_{b}"]))
            outputs[f"aux_imu_{b}"] = layers.Dense(n_classes, activation="softmax", name=f"aux_imu_{b}")(zi)
            feats.append(zi)
    fused = layers.Concatenate(name="fused")(feats) if len(feats) > 1 else feats[0]
    head = build_head(feature_dim(boards, use_imu), n_classes)
    outputs = {"zone": head(fused), **outputs}
    parts = {"audio_encoder": audio_enc, "head": head}
    if imu_enc is not None:
        parts["imu_encoder"] = imu_enc
    return keras.Model(inputs, outputs, name="touchwood_fusion"), parts


def features(
    parts: dict[str, keras.Model],
    inputs: dict,
    boards: tuple[str, ...],
    use_imu: bool,
    drop: tuple[str, ...] = (),
):
    """Head input computed outside the graph, exactly as the firmware will do it.

    `drop` zeroes branches, e.g. ("audio_B", "imu_B") = board B missing, ("imu_A", "imu_B") = mic only.
    """
    import numpy as np

    cols = []
    for b in boards:
        za = np.concatenate([parts["audio_encoder"].predict(inputs[f"audio_{b}"], verbose=0), inputs[f"loge_{b}"]], 1)
        cols.append(np.zeros_like(za) if f"audio_{b}" in drop else za)
        if use_imu:
            zi = parts["imu_encoder"].predict(inputs[f"imu_{b}"], verbose=0)
            cols.append(np.zeros_like(zi) if f"imu_{b}" in drop else zi)
    return np.concatenate(cols, axis=1).astype("float32")


def for_export(part: keras.Model, builder, **kwargs) -> keras.Model:
    """Rebuild a trained sub-model with a static batch size of 1 (otherwise Flatten/Reshape emit SHAPE / PACK)."""
    fixed = builder(batch_size=1, **kwargs)
    fixed.set_weights(part.get_weights())
    return fixed
