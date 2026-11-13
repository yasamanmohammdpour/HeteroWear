"""
Windowing that respects per-modality native sampling rates (see plan item 5:
resampling everything to one rate is convenient but scientifically
questionable — a 700Hz ECG and a 1Hz MMASH actigraphy stream should not be
forced onto the same grid before the model ever sees them).

For a window of duration `window_sec` starting at `t0`, each channel
contributes floor(window_sec * native_hz) samples, sliced from its own
timestamp axis by time (not by index), because index-based slicing
silently breaks the moment a channel has any dropped samples.
"""

from __future__ import annotations
import numpy as np

from src.data.schema import WindowSample, ChannelMeta


def _slice_by_time(values: np.ndarray, timestamps_sec: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Slice by actual timestamp range, not by index — robust to gaps/drops.
    Binary search (O(log N)) instead of a full boolean-mask rescan (O(N)) —
    matters at PPG-DaLiA/HHAR scale with thousands of windows over
    multi-million-sample channels. Requires timestamps_sec sorted ascending,
    which every adapter now guarantees."""
    lo = np.searchsorted(timestamps_sec, t0, side="left")
    hi = np.searchsorted(timestamps_sec, t1, side="left")
    return values[lo:hi]


def make_windows(
    dataset_name: str,
    subject_id: str,
    session_id: str,
    channels: dict[str, tuple[np.ndarray, ChannelMeta, np.ndarray]],
    window_sec: float,
    stride_sec: float,
    min_coverage_frac: float = 0.5,
    activity_labels: list[tuple[float, float, str]] | None = None,
    hr_stream: tuple[np.ndarray, np.ndarray, str] | None = None,
) -> list[WindowSample]:
    """
    channels: {channel_key: (values [T, n_axes], ChannelMeta, timestamps_sec [T])}
    min_coverage_frac: a channel is INCLUDED in a window only if it has at
        least this fraction of its expected samples present (below that,
        it's treated as missing for this window rather than padded — padding
        would contaminate the missing-modality experiments).
    """
    if not channels:
        return []

    # session span = union across channels so we don't lose data because
    # one channel started later than another
    t_min = min(ts[0] for _, _, ts in channels.values() if len(ts) > 0)
    t_max = max(ts[-1] for _, _, ts in channels.values() if len(ts) > 0)

    est_n_windows = int((t_max - t_min) / stride_sec) if t_max > t_min else 0
    print(f"    [windowing] {dataset_name}/{subject_id}: t_min={t_min:.1f}s t_max={t_max:.1f}s "
          f"span={t_max - t_min:.1f}s -> ~{est_n_windows:,} windows estimated", flush=True)
    if est_n_windows > 200_000:
        print(f"    [windowing] WARNING: {est_n_windows:,} windows is suspiciously high — "
              f"likely a bad/non-monotonic timestamp inflating t_max. Check channel ts ranges below.", flush=True)
        for ch_key, (_, meta, ts) in channels.items():
            if len(ts) > 0:
                print(f"      {ch_key}: ts[0]={ts[0]:.1f} ts[-1]={ts[-1]:.1f} n={len(ts)}", flush=True)

    windows: list[WindowSample] = []
    t0 = t_min
    _progress_every = max(1, est_n_windows // 10) if est_n_windows > 0 else 5000
    _count = 0
    while t0 + window_sec <= t_max:
        _count += 1
        if _count % _progress_every == 0:
            print(f"    [windowing] {dataset_name}/{subject_id}: {_count:,} windows generated so far "
                  f"(t0={t0:.1f}s / t_max={t_max:.1f}s)", flush=True)
        t1 = t0 + window_sec
        w = WindowSample(
            dataset=dataset_name,
            subject_id=subject_id,
            session_id=session_id,
            window_start_sec=t0,
            window_sec=window_sec,
        )
        for ch_key, (values, meta, ts) in channels.items():
            seg = _slice_by_time(values, ts, t0, t1)
            expected_n = int(round(window_sec * meta.native_hz))
            if expected_n == 0:
                continue
            coverage = len(seg) / expected_n
            if coverage < min_coverage_frac:
                continue  # channel absent for this window — this IS the missing-modality signal
            w.signals[ch_key] = seg
            w.channel_meta[ch_key] = meta

        # Deduplicate channels sharing the same (modality, location) — happens
        # for UCI-HAR, where multiple experiment sessions aren't globally
        # time-offset and can genuinely overlap within one window. Keep only
        # the channel with the most samples (best coverage) for that pair;
        # without this, a window could get e.g. two "ACC/phone_pocket"
        # channels that the model can't tell apart, which isn't richer
        # information, it's an accidental near-duplicate.
        by_mod_loc = {}
        for ch_key, meta in w.channel_meta.items():
            by_mod_loc.setdefault((meta.modality, meta.location), []).append(ch_key)
        for pair, keys in by_mod_loc.items():
            if len(keys) > 1:
                keys_sorted = sorted(keys, key=lambda k: len(w.signals[k]), reverse=True)
                for dup_key in keys_sorted[1:]:
                    del w.signals[dup_key]
                    del w.channel_meta[dup_key]

        if not w.signals:
            t0 += stride_sec
            continue

        if activity_labels:
            w.activity_label = _label_for_window(activity_labels, t0, t1)

        if hr_stream is not None:
            hr_ts, hr_vals, hr_src = hr_stream
            seg_hr = _slice_by_time(hr_vals, hr_ts, t0, t1)
            if len(seg_hr) > 0:
                w.hr_bpm = float(np.nanmean(seg_hr))
                w.hr_source = hr_src

        windows.append(w)
        t0 += stride_sec

    return windows


def _label_for_window(activity_labels: list[tuple[float, float, str]], t0: float, t1: float) -> str | None:
    """Majority-overlap label for the window; None if no segment covers >50% of it."""
    window_sec = t1 - t0
    overlap_by_label: dict[str, float] = {}
    for seg_start, seg_end, label in activity_labels:
        overlap = max(0.0, min(t1, seg_end) - max(t0, seg_start))
        if overlap > 0:
            overlap_by_label[label] = overlap_by_label.get(label, 0.0) + overlap
    if not overlap_by_label:
        return None
    best_label, best_overlap = max(overlap_by_label.items(), key=lambda kv: kv[1])
    return best_label if best_overlap / window_sec >= 0.5 else None
