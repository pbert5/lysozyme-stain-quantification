from typing import Optional

import numpy as np
import pytest


def pytest_configure(config):
    """Expose custom markers to ``pytest --markers`` output."""

    config.addinivalue_line(
        "markers",
        "heavy: marks tests that require large inputs or long runtimes",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that run slower than the standard unit suite",
    )


@pytest.fixture
def tiny_imgs() -> dict:
    """Return deterministic tiny images for tests."""
    rng = np.random.default_rng(0)
    S, H, W = 3, 6, 6
    dapi = [rng.random((H, W), dtype=np.float32) for _ in range(S)]
    rfp = [rng.random((H, W), dtype=np.float32) for _ in range(S)]
    return dict(subject=[f"s{i+1}" for i in range(S)], dapi=dapi, rfp=rfp)


@pytest.fixture
def many_channel_img_maker():
    """Return a configurable factory for synthetic multi-channel image stacks.

    The fixture is intentionally flexible: it supports the tiny stacks used in
    quick unit tests as well as HD-sized stress scenarios. Callers can tweak the
    subject count, image shape, or the set of synthesized source types while
    keeping deterministic randomness via the ``seed`` argument.
    """

    def _maker(
        *,
        seed: int = 0,
        subjects: Optional[int] = None,
        shape: Optional[tuple[int, int]] = None,
        channels: tuple[str, ...] = ("gray_float", "rgb_int", "mixed", "sparse"),
        profile: str = "standard",
    ) -> tuple[list[str], dict[str, list[np.ndarray]]]:
        """Generate a batch of synthetic images matching the requested profile.

        Parameters
        ----------
        seed:
            Seeds the RNG so test runs remain reproducible.
        subjects:
            Optional override for the number of subjects. ``None`` lets the
            chosen profile decide (``3`` for ``"standard"``, ``200`` for
            ``"mega"``).
        shape:
            Optional override for the ``(height, width)`` of each image. Again,
            ``None`` defers to the active profile.
        channels:
            Tuple describing which synthetic source flavours to include. The
            default mirrors the original fixture but callers can request a subset
            (e.g. ``("gray_float", "mixed")``) to control memory footprint.
        profile:
            Either ``"standard"`` for lightweight arrays or ``"mega"`` for HD
            sized stress tests.
        """

        if profile not in {"standard", "mega"}:
            raise ValueError("profile must be 'standard' or 'mega'")

        if profile == "mega":
            subjects = 200 if subjects is None else subjects
            shape = (1080, 1920) if shape is None else shape
        else:
            subjects = 3 if subjects is None else subjects
            shape = (6, 8) if shape is None else shape

        rng = np.random.default_rng(seed)
        H, W = shape
        subject_names = [f"subject_{i:04d}" for i in range(1, subjects + 1)]

        mixed_axis = 0 if rng.random() < 0.5 else -1
        sparse_axis = 0 if rng.random() < 0.5 else -1
        mixed_perm = rng.permutation(4)
        sparse_perm = rng.permutation(3)

        def random_float(scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
            """Return a float32 plane with controllable scale and bias."""

            return (rng.random((H, W), dtype=np.float32) * scale) + offset

        def random_rgb() -> np.ndarray:
            """Return a pseudo RGB cube with the channels randomly permuted."""

            base = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
            return base[..., rng.permutation(3)]

        def mixed_four_channel() -> np.ndarray:
            """Assemble a four-channel stack mixing empty planes and signals."""

            channels_stack = [
                np.zeros((H, W), dtype=np.float32),
                random_float(scale=1.0, offset=0.0),
                random_float(scale=5.0, offset=1.0),
                rng.integers(0, 256, size=(H, W), dtype=np.uint16).astype(np.float32),
            ]
            ordered = [channels_stack[i] for i in mixed_perm]
            return np.stack(ordered, axis=mixed_axis)

        def sparse_multi() -> np.ndarray:
            """Return a stack where only one channel contains meaningful values."""

            channels_stack = [
                np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=np.float32),
                random_float(scale=2.0, offset=0.5),
            ]
            ordered = [channels_stack[i] for i in sparse_perm]
            return np.stack(ordered, axis=sparse_axis)

        def gray_float_plane() -> np.ndarray:
            """Return a baseline grayscale plane with unit scale."""

            return random_float(scale=1.0)

        available_generators = {
            "gray_float": gray_float_plane,
            "rgb_int": random_rgb,
            "mixed": mixed_four_channel,
            "sparse": sparse_multi,
        }

        unknown = set(channels) - set(available_generators)
        if unknown:
            raise ValueError(f"Unknown channel types requested: {sorted(unknown)}")

        raw_inputs = {name: [] for name in channels}

        for _ in subject_names:
            for name in channels:
                raw_inputs[name].append(available_generators[name]())

        return subject_names, raw_inputs

    return _maker


@pytest.fixture
def mega_imgs(many_channel_img_maker):
    """Shortcut fixture for generating HD-sized batches with sensible defaults.

    It simply forwards to :func:`many_channel_img_maker` with the ``"mega"``
    profile so stress tests can request large stacks without repeating setup
    boilerplate.
    """

    def _mega(**overrides):
        """Delegate to :func:`many_channel_img_maker` using the ``"mega"`` profile."""

        profile = overrides.pop("profile", "mega")
        return many_channel_img_maker(profile=profile, **overrides)

    return _mega
