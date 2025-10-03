from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["generate_test_crypt_labels"]


@dataclass(frozen=True)
class PlacementConfig:
	"""Configuration describing a special placement tweak for a crypt.

	Attributes
	----------
	mode:
		Placement mode selector:
		  * ``0`` – position so that the crypt sits half outside the top border.
		  * ``1`` – apply a vertical offset relative to the crypt line.
		  * ``2`` – horizontally compress this crypt and the next one to force overlap.
	vertical_offset_px:
		Only used when ``mode == 1``. Specifies the signed vertical offset in pixels.
	overlap_ratio:
		Only used when ``mode == 2``. Desired centre-to-centre distance as a fraction
		of ``2 * radius``. Values ``< 1`` enforce overlap.
	"""

	mode: int
	vertical_offset_px: int | None = None
	overlap_ratio: float | None = None


def _infer_radius_and_count(
	width: int,
	*,
	blob_size_px: Optional[int],
	radius_px: Optional[int],
	num_crypts: Optional[int],
) -> Tuple[int, int, str]:
	"""Infer the crypt radius and count using the same heuristics as the image generator."""

	if blob_size_px is not None:
		radius = max(1, int(round(blob_size_px / 2)))
		radius_source = "blob_size_px"
		if num_crypts is None:
			num_crypts = max(1, int(np.floor(width / (2.5 * radius))))
	elif radius_px is not None:
		radius = max(1, int(radius_px))
		radius_source = "radius_px"
		if num_crypts is None:
			num_crypts = max(1, int(np.floor(width / (2.5 * radius))))
	else:
		if num_crypts is None:
			num_crypts = 5
		radius = max(1, int(round(width / (2.5 * num_crypts))))
		radius_source = "derived_from_num_crypts"

	return radius, int(num_crypts), radius_source


def _compute_base_positions(
	*,
	num_crypts: int,
	margin_left: float,
	margin_right: float,
	radius: int,
	allow_overlaps: bool,
	overlap_ratio: Optional[float],
) -> List[int]:
	if num_crypts <= 0:
		raise ValueError("num_crypts must be positive")

	if margin_right < margin_left:
		raise ValueError("image width too small for requested border margin")

	if num_crypts == 1:
		return [int(round((margin_left + margin_right) / 2.0))]

	span_available = float(margin_right - margin_left)
	if allow_overlaps:
		if overlap_ratio is None:
			overlap_ratio = 0.9
		overlap_ratio = float(overlap_ratio)
		if overlap_ratio < 0:
			raise ValueError("overlap_ratio must be non-negative")
		desired_distance = max(1.0, 2.0 * radius * overlap_ratio)
		total_span = desired_distance * (num_crypts - 1)
		if total_span > span_available and total_span > 0:
			# Clamp ratio to fit while preserving requested overlap intent.
			overlap_ratio = max(0.0, span_available / (2.0 * radius * (num_crypts - 1)))
			desired_distance = max(1.0, 2.0 * radius * overlap_ratio)
			total_span = desired_distance * (num_crypts - 1)
		start = margin_left + 0.5 * (span_available - total_span)
		return [int(round(start + i * desired_distance)) for i in range(num_crypts)]

	# Non-overlapping layout.
	minimum_spacing = 2 * radius + 1
	implied_spacing = span_available / (num_crypts - 1)
	if implied_spacing < minimum_spacing:
		raise ValueError(
			"image width insufficient to place crypts without touching; "
			"decrease num_crypts, radius, or allow overlaps"
		)
	positions = [margin_left + i * implied_spacing for i in range(num_crypts)]
	return [int(round(p)) for p in positions]


def _apply_special_placements(
	centers: List[List[int]],
	placement_modes: Sequence[PlacementConfig],
	*,
	radius: int,
	height: int,
	width: int,
	base_line_y: int,
	touch_border: bool,
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
	"""Adjust centres according to placement configurations.

	Returns the updated centres and a record of explicit overlap pairs.
	"""

	overlap_pairs: List[Tuple[int, int]] = []

	for idx, cfg in enumerate(placement_modes):
		if cfg.mode < 0:
			continue
		if cfg.mode == 0:
			centers[idx][1] = max(0, radius // 2)
		elif cfg.mode == 1:
			if cfg.vertical_offset_px is None:
				raise ValueError("placement mode 1 requires vertical_offset_px")
			target_y = base_line_y + int(cfg.vertical_offset_px)
			if target_y < 0 or target_y >= height:
				raise ValueError(
					f"crypt index {idx} with mode 1 offsets centre outside image bounds"
				)
			centers[idx][1] = target_y
		elif cfg.mode == 2:
			if idx + 1 >= len(centers):
				continue
			ratio = cfg.overlap_ratio if cfg.overlap_ratio is not None else 0.8
			ratio = float(ratio)
			if ratio < 0:
				raise ValueError("overlap ratio must be non-negative")
			target_distance = max(1, int(round(2 * radius * ratio)))
			mid = 0.5 * (centers[idx][0] + centers[idx + 1][0])
			centers[idx][0] = int(round(mid - target_distance / 2))
			centers[idx + 1][0] = int(round(mid + target_distance / 2))
			overlap_pairs.append((idx, idx + 1))
		else:
			raise ValueError(f"unsupported placement mode: {cfg.mode}")

	# Validate bounds after adjustments.
	for idx, (cx, cy) in enumerate(centers):
		if cy < 0 or cy >= height:
			raise ValueError(f"crypt index {idx} lies outside vertical bounds after placement")
		if not touch_border and not any(
			cfg.mode == 0 and idy == idx for idy, cfg in enumerate(placement_modes)
		):
			if cx < radius or cx > (width - 1 - radius):
				raise ValueError(
					"touch_border is False but a crypt centre would cause edge contact"
				)
		cx_clamped = min(max(cx, 0), width - 1)
		cy_clamped = min(max(cy, 0), height - 1)
		centers[idx][0] = cx_clamped
		centers[idx][1] = cy_clamped

	return centers, overlap_pairs


def _ensure_no_forbidden_overlaps(
	centers: Sequence[Sequence[int]],
	*,
	radius: int,
	allow_overlaps: bool,
) -> None:
	if allow_overlaps or len(centers) < 2:
		return

	for i in range(len(centers) - 1):
		cx0, cy0 = centers[i]
		cx1, cy1 = centers[i + 1]
		distance_sq = (cx0 - cx1) ** 2 + (cy0 - cy1) ** 2
		if distance_sq < (2 * radius) ** 2:
			raise ValueError(
				"unexpected overlap detected while overlaps are disabled; "
				"adjust image size, radius, or enable overlaps"
			)


def generate_test_crypt_labels(
	*,
	image_size: Tuple[int, int] = (100, 100),
	num_crypts: Optional[int] = None,
	line_height_frac: float = 0.6,
	placement_modes: Optional[Sequence[Optional[PlacementConfig | int | Dict[str, object]]]] = None,
	blob_size_px: Optional[int] = None,
	radius_px: Optional[int] = None,
	allow_overlaps: bool = False,
	overlap_ratio: Optional[float] = None,
	touch_border: bool = False,
	dtype: np.dtype = np.int32,
) -> Tuple[np.ndarray, Dict[str, object]]:
	"""Generate a labelled crypt mask for testing downstream segmentation logic.

	The generator mirrors the geometry used by ``demo_crypt_image_generator`` but returns
	only the labelled array where each crypt occupies a unique integer label. Optional
	placement modes enable specific edge cases:

	  • ``mode 0`` – half outside the top border to test edge-pruning logic.
	  • ``mode 1`` – apply a per-crypt vertical offset from the inferred crypt line.
	  • ``mode 2`` – enforce horizontal overlap with the subsequent crypt.

	Parameters mimic the original helper while focusing on geometry rather than pixel
	intensities.
	"""

	height, width = image_size
	if height <= 0 or width <= 0:
		raise ValueError("image_size must be positive")
	if not (0.0 <= line_height_frac <= 1.0):
		raise ValueError("line_height_frac must be within [0, 1]")

	radius, num_crypts, radius_source = _infer_radius_and_count(
		width,
		blob_size_px=blob_size_px,
		radius_px=radius_px,
		num_crypts=num_crypts,
	)

	line_y = int(round(line_height_frac * (height - 1)))

	margin_left = 0.0 if touch_border else float(radius)
	margin_right = float(width - 1) if touch_border else float(width - 1 - radius)

	base_positions = _compute_base_positions(
		num_crypts=num_crypts,
		margin_left=margin_left,
		margin_right=margin_right,
		radius=radius,
		allow_overlaps=allow_overlaps,
		overlap_ratio=overlap_ratio,
	)

	centers = [[int(x), line_y] for x in base_positions]

	# Normalise placement modes into PlacementConfig instances.
	if placement_modes is None:
		placement_modes = [PlacementConfig(mode=-1) for _ in range(num_crypts)]
	else:
		if len(placement_modes) != num_crypts:
			raise ValueError("placement_modes length must match num_crypts")
		normalised: List[PlacementConfig] = []
		for entry in placement_modes:
			if entry is None:
				normalised.append(PlacementConfig(mode=-1))
			elif isinstance(entry, PlacementConfig):
				normalised.append(entry)
			elif isinstance(entry, int):
				normalised.append(PlacementConfig(mode=entry))
			elif isinstance(entry, dict):
				mode_val = entry.get("mode")
				if mode_val is None:
					raise ValueError("dict placement config must include 'mode'")
				normalised.append(
					PlacementConfig(
						mode=int(mode_val),
						vertical_offset_px=entry.get("vertical_offset_px"),
						overlap_ratio=entry.get("overlap_ratio"),
					)
				)
			else:
				raise TypeError("unsupported placement configuration type")
		placement_modes = normalised

	centers, overlap_pairs = _apply_special_placements(
		centers,
		placement_modes,
		radius=radius,
		height=height,
		width=width,
		base_line_y=line_y,
		touch_border=touch_border,
	)

	_ensure_no_forbidden_overlaps(centers, radius=radius, allow_overlaps=allow_overlaps)

	yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
	distance_stack = np.full((num_crypts, height, width), np.inf, dtype=np.float64)
	per_label_masks: List[np.ndarray] = []

	radius_sq = radius * radius
	for idx, (cx, cy) in enumerate(centers):
		d2 = (xx - cx) ** 2 + (yy - cy) ** 2
		mask = d2 <= radius_sq
		per_label_masks.append(mask)
		distance_stack[idx, mask] = d2[mask]

	has_region = np.isfinite(distance_stack).any(axis=0)
	nearest = np.argmin(distance_stack, axis=0)
	labels = np.zeros((height, width), dtype=np.int32)
	labels[has_region] = nearest[has_region] + 1

	meta = {
		"radius_px": radius,
		"radius_source": radius_source,
		"line_height_frac": line_height_frac,
		"line_y": line_y,
		"centers": [tuple(center) for center in centers],
		"num_crypts": num_crypts,
		"touch_border": touch_border,
		"allow_overlaps": allow_overlaps,
		"overlap_ratio": overlap_ratio,
		"placement_modes": placement_modes,
		"overlap_pairs": overlap_pairs,
		"masks": {
			"combined": labels > 0,
			"per_label": per_label_masks,
		},
	}

	return labels.astype(dtype, copy=False), meta


if __name__ == "__main__":
	lbl, info = generate_test_crypt_labels(
		image_size=(200, 200),
		num_crypts=4,
		placement_modes=[
			PlacementConfig(mode=0),
			PlacementConfig(mode=1, vertical_offset_px=-10),
			PlacementConfig(mode=2, overlap_ratio=0.7),
			PlacementConfig(mode=-1),
		],
		allow_overlaps=True,
		touch_border=False,
	)
	print("Generated labels with shape", lbl.shape)
	print("Unique labels:", np.unique(lbl))
