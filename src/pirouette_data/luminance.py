"""Normalized luminance from behavior-video frames (a proxy for external lighting).

Per sampled frame the mean luma is computed, then:

1. **Per-sample luminance** ``l_t`` — mean pixel intensity scaled to ``[0, 1]``.
2. **Two-sample average** ``L_t = (l_t + l_{t-1}) / 2`` — high only when the
   current *and* previous sample are bright, near 0 only when both are dim. The
   first sample has no previous one, so it uses its closest sample (the next one)
   as its reference: ``L_0 = (l_0 + l_1) / 2``. Nothing is dropped.
3. **Scaling** (``scaling=``), default ``"minmax"``: ``L_norm = (L - min) /
   (max - min)``, so 0 / 1 are the dimmest / brightest samples *within the
   session* (comparable within, not across, sessions; never clips). Alternatives:
   ``"absolute"`` (``L`` itself, mean pixel / 255) and ``"percentile"``
   (``clip((L - p_low) / (p_high - p_low), 0, 1)``, which clips the tails).

Fast streaming of S3 video
--------------------------
Lighting changes over seconds to minutes, so decoding every 60 Hz frame is
wasteful. The MP4 files are H.264 with a keyframe (IDR, independently decodable)
every ``GOP`` frames (250 here, ~4.2 s), and the sample index (``moov``) sits at
the front of the file. This module therefore:

* reads only the ``moov`` box over HTTP range requests and builds a frame ->
  byte-offset table (:func:`read_mp4_index`),
* fetches **only the byte ranges of the frames it needs** in parallel
  (a keyframe is ~160 kB vs ~1.2 GB for the hour),
* pipes those frames, as a raw H.264 stream, to one ``ffmpeg`` process per chunk
  that decodes them and area-averages each to a tiny image (the mean of an
  area-averaged image equals the mean of the full frame).

Distributed use
---------------
The work splits into a per-chunk step and a whole-session merge step:
:func:`select_shard` / :func:`process_chunks` / :func:`save_chunk_result` run
independently per slice of chunks (``scripts/luminance_chunks.py``), and
:func:`load_chunk_results` / :func:`combine_chunk_tables` / :func:`save_outputs`
produce the final trace (``scripts/luminance_merge.py``).
:func:`build_luminance_dataset` does both in one process.

Sampling modes (see :func:`plan_samples`):

``"keyframe"``
    Every *m*-th keyframe (``m = round(interval / GOP)``, at least 1). Fastest by
    far, but the rate is limited to one sample per GOP (~0.24 Hz here).
``"exact"``
    Frames on a regular ``sample_hz`` grid. A frame can only be decoded from its
    GOP's keyframe, so every GOP that contains a wanted frame is fetched and
    fully decoded. At rates above one sample per GOP this approaches decoding the
    whole video (slow).
``"auto"``
    ``"keyframe"`` when the requested interval is at least one GOP, else
    ``"exact"``.
"""

from __future__ import annotations

import os
import struct
import subprocess
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from pirouette_data import ingestion

# Default percentiles for relative scaling.
DEFAULT_LOW_PERCENTILE = 1.0
DEFAULT_HIGH_PERCENTILE = 99.0

# Each decoded frame is area-averaged to THUMB x THUMB before taking the mean.
_THUMB = 16

# Number of ffmpeg decodes allowed to run at once (set by build_luminance_dataset).
# Network fetches are latency-bound and may use many more threads, but decoding is
# CPU-bound: oversubscribing it makes ffmpeg's worker threads spin-wait. Default is
# half the cores so the machine stays usable for other work.
_DECODE_SLOTS = threading.BoundedSemaphore(max(1, (os.cpu_count() or 2) // 2))
_ANNEXB_START = b"\x00\x00\x00\x01"


# ---------------------------------------------------------------------------
# Steps 2-3: two-sample average and relative scaling
# ---------------------------------------------------------------------------
def two_frame_average(luminance: np.ndarray) -> np.ndarray:
    """Average each sample with its predecessor: ``L_t = (l_t + l_{t-1}) / 2``.

    The first sample has no predecessor, so its closest sample (the second) is
    used as the reference instead: ``L_0 = (l_0 + l_1) / 2``. The first and second
    outputs are therefore equal, and no sample is dropped. The result stays in the
    range of the input (a convex combination).

    Parameters
    ----------
    luminance:
        1-D luminance values in time order.

    Returns
    -------
    numpy.ndarray
        Array of the same length as *luminance*.
    """
    lum = np.asarray(luminance, dtype="float64")
    out = lum.copy()
    out[1:] = 0.5 * (lum[1:] + lum[:-1])
    if len(lum) > 1:
        out[0] = out[1]  # boundary: reference the closest (next) sample
    return out


def rescale_percentile(
    values: np.ndarray,
    low: float = DEFAULT_LOW_PERCENTILE,
    high: float = DEFAULT_HIGH_PERCENTILE,
) -> np.ndarray:
    """Relative scaling: map the *low*/*high* percentiles to 0/1 and clip.

    Parameters
    ----------
    values:
        1-D array to rescale. NaNs are ignored when computing percentiles.
    low, high:
        Percentiles (0-100) that map to 0 and 1.

    Returns
    -------
    numpy.ndarray
        Values in ``[0, 1]``. A constant series maps to all zeros.
    """
    vals = np.asarray(values, dtype="float64")
    p_low, p_high = np.nanpercentile(vals, [low, high])
    span = p_high - p_low
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(vals)
    return np.clip((vals - p_low) / span, 0.0, 1.0)


SCALINGS = ("absolute", "minmax", "percentile")
DEFAULT_SCALING = "minmax"


def rescale_minmax(values: np.ndarray) -> np.ndarray:
    """Map the series minimum to 0 and maximum to 1 (never clips)."""
    vals = np.asarray(values, dtype="float64")
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(vals)
    return (vals - lo) / span


def normalized_luminance(
    luminance: np.ndarray,
    low: float = DEFAULT_LOW_PERCENTILE,
    high: float = DEFAULT_HIGH_PERCENTILE,
    scaling: str = DEFAULT_SCALING,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sample average followed by the chosen scaling.

    ``scaling``: ``"minmax"`` (default; session min -> 0, max -> 1, no
    clipping), ``"absolute"`` (the averaged mean-pixel/255 value, already in
    ``[0, 1]``, no clipping) or ``"percentile"`` (*low*/*high* percentiles -> 0/1, **clipped**).

    Returns
    -------
    luminance_2f : numpy.ndarray
        Two-sample average, in absolute ``[0, 1]`` units.
    luminance_norm : numpy.ndarray
        Scaled two-sample average in ``[0, 1]``.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}, got {scaling!r}")
    lum_2f = two_frame_average(luminance)
    if scaling == "absolute":
        return lum_2f, lum_2f.copy()
    if scaling == "minmax":
        return lum_2f, rescale_minmax(lum_2f)
    return lum_2f, rescale_percentile(lum_2f, low, high)


# ---------------------------------------------------------------------------
# MP4 index (moov) parsing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mp4Index:
    """Per-frame byte layout of an H.264 MP4, read from its ``moov`` box.

    All arrays are in decode order and 0-based.
    """

    offsets: np.ndarray  # int64, absolute file offset of each frame
    sizes: np.ndarray  # int64, byte size of each frame
    sync: np.ndarray  # int64, indices of keyframes (IDR)
    fps: float
    nal_length_size: int
    parameter_sets: bytes  # Annex-B SPS + PPS

    @property
    def n_frames(self) -> int:
        return len(self.offsets)

    @property
    def median_gop(self) -> float:
        """Median keyframe spacing in frames."""
        if len(self.sync) < 2:
            return float(self.n_frames)
        return float(np.median(np.diff(self.sync)))


def _boxes(buf: bytes, start: int = 0, end: int | None = None):
    """Yield ``(type, payload_start, box_end)`` for sibling boxes in *buf*."""
    end = len(buf) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos : pos + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        yield typ.decode("latin1"), pos + header, pos + size
        pos += size


def _child(buf: bytes, span: tuple[int, int], *path: str) -> tuple[int, int]:
    """Descend through nested boxes; return the ``(start, end)`` of the last."""
    for name in path:
        for typ, s, e in _boxes(buf, *span):
            if typ == name:
                span = (s, e)
                break
        else:
            raise ValueError(f"MP4 box {name!r} not found")
    return span


def _u32(buf: bytes, pos: int, count: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=">u4", count=count, offset=pos).astype("int64")


def _parse_moov(moov: bytes) -> Mp4Index:
    """Build an :class:`Mp4Index` from the payload of a ``moov`` box."""
    # First video track.
    trak = None
    for typ, s, e in _boxes(moov):
        if typ == "trak":
            hdlr_s, _ = _child(moov, (s, e), "mdia", "hdlr")
            if moov[hdlr_s + 8 : hdlr_s + 12] == b"vide":
                trak = (s, e)
                break
    if trak is None:
        raise ValueError("No video track in MP4")

    mdhd_s, _ = _child(moov, trak, "mdia", "mdhd")
    if moov[mdhd_s] == 1:
        timescale = struct.unpack(">I", moov[mdhd_s + 20 : mdhd_s + 24])[0]
    else:
        timescale = struct.unpack(">I", moov[mdhd_s + 12 : mdhd_s + 16])[0]

    stbl = _child(moov, trak, "mdia", "minf", "stbl")
    tables = {t: (s, e) for t, s, e in _boxes(moov, *stbl)}

    # Frame duration -> fps (constant frame rate assumed: first stts entry).
    s, _ = tables["stts"]
    delta = struct.unpack(">I", moov[s + 12 : s + 16])[0]
    fps = timescale / delta

    # Sizes.
    s, _ = tables["stsz"]
    uniform, count = struct.unpack(">II", moov[s + 4 : s + 12])
    sizes = (
        np.full(count, uniform, dtype="int64") if uniform else _u32(moov, s + 12, count)
    )

    # Keyframes (1-based in the file).
    s, _ = tables["stss"]
    n_sync = struct.unpack(">I", moov[s + 4 : s + 8])[0]
    sync = _u32(moov, s + 8, n_sync) - 1

    # Chunk offsets and samples-per-chunk -> per-sample absolute offsets.
    if "stco" in tables:
        s, _ = tables["stco"]
        n_chunks = struct.unpack(">I", moov[s + 4 : s + 8])[0]
        chunk_off = _u32(moov, s + 8, n_chunks)
    else:
        s, _ = tables["co64"]
        n_chunks = struct.unpack(">I", moov[s + 4 : s + 8])[0]
        chunk_off = np.frombuffer(moov, dtype=">u8", count=n_chunks, offset=s + 8).astype(
            "int64"
        )

    s, _ = tables["stsc"]
    n_ent = struct.unpack(">I", moov[s + 4 : s + 8])[0]
    ent = _u32(moov, s + 8, 3 * n_ent).reshape(n_ent, 3)
    first_chunk = ent[:, 0] - 1
    next_first = np.append(first_chunk[1:], n_chunks)
    per_chunk = np.repeat(ent[:, 1], next_first - first_chunk)  # samples per chunk

    chunk_of_sample = np.repeat(np.arange(n_chunks), per_chunk)[:count]
    first_sample_of_chunk = np.concatenate([[0], np.cumsum(per_chunk)[:-1]])
    excl = np.concatenate([[0], np.cumsum(sizes)[:-1]])  # bytes before each sample
    offsets = chunk_off[chunk_of_sample] + (
        excl - excl[first_sample_of_chunk[chunk_of_sample]]
    )

    # avcC: NAL length size + SPS/PPS.
    stsd_s, stsd_e = tables["stsd"]
    avcc = None
    entry_s = stsd_s + 8  # skip version/flags + entry_count
    for typ, s2, e2 in _boxes(moov, entry_s, stsd_e):
        # Visual sample entry: 78 bytes of fields precede its child boxes.
        for t3, s3, e3 in _boxes(moov, s2 + 78, e2):
            if t3 == "avcC":
                avcc = moov[s3:e3]
    if avcc is None:
        raise ValueError("Not an H.264 (avcC) MP4")
    nal_length_size = (avcc[4] & 0x03) + 1
    pos = 6
    params = b""
    for _ in range(avcc[5] & 0x1F):  # SPS
        ln = struct.unpack(">H", avcc[pos : pos + 2])[0]
        params += _ANNEXB_START + avcc[pos + 2 : pos + 2 + ln]
        pos += 2 + ln
    for _ in range(avcc[pos]):  # PPS
        ln = struct.unpack(">H", avcc[pos + 1 : pos + 3])[0]
        params += _ANNEXB_START + avcc[pos + 3 : pos + 3 + ln]
        pos += 2 + ln

    return Mp4Index(
        offsets=offsets,
        sizes=sizes,
        sync=sync,
        fps=float(fps),
        nal_length_size=nal_length_size,
        parameter_sets=params,
    )


def _get_range(s3, bucket: str, key: str, start: int, end: int) -> bytes:
    """Inclusive byte-range GET."""
    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


def read_mp4_index(s3, bucket: str, key: str) -> Mp4Index:
    """Read an MP4's frame index from S3 using only range requests.

    Requires the ``moov`` box to be readable (it is at the file start for
    ``faststart`` files, as these videos are, but any position works).

    Parameters
    ----------
    s3:
        boto3 S3 client.
    bucket, key:
        Location of the ``.mp4``.

    Returns
    -------
    Mp4Index
    """
    size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    pos = 0
    while pos < size:
        hdr = _get_range(s3, bucket, key, pos, pos + 15)
        box_size, typ = struct.unpack(">I4s", hdr[:8])
        header = 8
        if box_size == 1:
            box_size = struct.unpack(">Q", hdr[8:16])[0]
            header = 16
        if typ == b"moov":
            payload = _get_range(s3, bucket, key, pos + header, pos + box_size - 1)
            return _parse_moov(payload)
        if box_size == 0:
            break
        pos += box_size
    raise ValueError(f"No moov box found in s3://{bucket}/{key}")


# ---------------------------------------------------------------------------
# Sample planning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplePlan:
    """Which frames to measure and which GOPs must be decoded to get them.

    ``frames`` are the (sorted, unique) frame indices to measure. ``units`` lists
    ``(first_frame, n_decode)``: decode ``n_decode`` consecutive packets starting
    at ``first_frame`` (a keyframe).
    """

    mode: str
    frames: np.ndarray
    units: list[tuple[int, int]]


def plan_samples(idx: Mp4Index, sample_hz: float, mode: str = "auto") -> SamplePlan:
    """Choose the frames to sample for a target rate.

    Parameters
    ----------
    idx:
        Frame index of the video chunk.
    sample_hz:
        Target sampling rate in Hz (e.g. ``1.0`` or ``0.1``).
    mode:
        ``"auto"``, ``"keyframe"`` or ``"exact"`` (see the module docstring).

    Returns
    -------
    SamplePlan
    """
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    if mode not in ("auto", "keyframe", "exact"):
        raise ValueError(f"Unknown mode {mode!r}")
    interval = idx.fps / sample_hz  # frames between samples
    if mode == "auto":
        mode = "keyframe" if interval >= idx.median_gop else "exact"

    if mode == "keyframe":
        step = max(1, int(round(interval / idx.median_gop)))
        frames = idx.sync[::step]
        return SamplePlan("keyframe", frames, [(int(f), 1) for f in frames])

    frames = np.unique(np.round(np.arange(0, idx.n_frames, interval)).astype("int64"))
    frames = frames[frames < idx.n_frames]
    gop_of = np.searchsorted(idx.sync, frames, side="right") - 1
    gop_ends = np.append(idx.sync[1:], idx.n_frames)
    units = [(int(idx.sync[g]), int(gop_ends[g] - idx.sync[g])) for g in np.unique(gop_of)]
    return SamplePlan("exact", frames, units)


# ---------------------------------------------------------------------------
# Fetch + decode
# ---------------------------------------------------------------------------
def _avcc_to_annexb(packet: memoryview, nal_length_size: int) -> bytes:
    """Convert a length-prefixed (AVCC) access unit to Annex-B."""
    out = []
    pos, n = 0, len(packet)
    while pos < n:
        ln = int.from_bytes(packet[pos : pos + nal_length_size], "big")
        pos += nal_length_size
        out.append(_ANNEXB_START)
        out.append(packet[pos : pos + ln])
        pos += ln
    return b"".join(out)


def _fetch_unit(s3, bucket, key, idx: Mp4Index, first: int, count: int) -> bytes:
    """Fetch ``count`` frames from ``first`` and return them as Annex-B."""
    last = first + count - 1
    start = int(idx.offsets[first])
    end = int(idx.offsets[last] + idx.sizes[last]) - 1
    blob = memoryview(_get_range(s3, bucket, key, start, end))
    parts = [idx.parameter_sets]
    for i in range(first, last + 1):
        o = int(idx.offsets[i]) - start
        parts.append(_avcc_to_annexb(blob[o : o + int(idx.sizes[i])], idx.nal_length_size))
    return b"".join(parts)


def _decode_means(stream: bytes, threads: int = 1) -> np.ndarray:
    """Decode an Annex-B H.264 stream with ffmpeg; return per-frame mean luma.

    Each frame is converted from limited-range luma to full range, area-averaged
    to ``_THUMB`` x ``_THUMB`` and averaged, giving a value in ``[0, 1]``.
    """
    import imageio_ffmpeg

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner", "-loglevel", "error",
        "-threads", str(threads),
        "-filter_threads", "1", "-filter_complex_threads", "1",
        "-f", "h264", "-i", "pipe:0",
        "-vf", f"scale={_THUMB}:{_THUMB}:flags=area:in_range=tv:out_range=pc,format=gray",
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    # Below-normal priority (Windows) keeps the desktop responsive during long runs.
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    with _DECODE_SLOTS:
        proc = subprocess.run(
            cmd, input=stream, capture_output=True, check=False, creationflags=flags
        )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:500]}")
    arr = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, _THUMB * _THUMB)
    return arr.mean(axis=1, dtype="float64") / 255.0


def video_luminance_s3(
    s3,
    bucket: str,
    key: str,
    sample_hz: float,
    mode: str = "auto",
    io_threads: int = 8,
    decode_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, SamplePlan]:
    """Sampled luminance of one S3 video, fetching only the bytes needed.

    Parameters
    ----------
    s3:
        boto3 S3 client (thread-safe).
    bucket, key:
        Location of the ``.mp4``.
    sample_hz:
        Target sampling rate (see :func:`plan_samples`).
    mode:
        ``"auto"``, ``"keyframe"`` or ``"exact"``.
    io_threads:
        Parallel range requests per video.
    decode_threads:
        ffmpeg decoder threads (keep at 1; parallelism comes from running one
        decode per CPU core).

    Returns
    -------
    frames : numpy.ndarray
        Frame indices that were measured.
    luminance : numpy.ndarray
        Mean luminance in ``[0, 1]`` for each of *frames*.
    plan : SamplePlan
    """
    idx = read_mp4_index(s3, bucket, key)
    plan = plan_samples(idx, sample_hz, mode)
    with ThreadPoolExecutor(max_workers=io_threads) as pool:
        chunks = list(
            pool.map(lambda u: _fetch_unit(s3, bucket, key, idx, *u), plan.units)
        )
    stream = b"".join(chunks)
    del chunks  # release the per-unit pieces before decoding
    means = _decode_means(stream, threads=decode_threads)
    del stream

    expected = sum(n for _, n in plan.units)
    if len(means) != expected:
        raise RuntimeError(
            f"{key}: decoded {len(means)} frames, expected {expected}."
        )
    # Select the wanted frames from each decoded unit (output is display order,
    # GOPs are closed so a unit's outputs are exactly its own frames).
    out, base = [], 0
    for first, n in plan.units:
        wanted = plan.frames[(plan.frames >= first) & (plan.frames < first + n)]
        out.append(means[base + (wanted - first)])
        base += n
    return plan.frames, np.concatenate(out), plan


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------
def make_s3_client(anonymous: bool = True, max_connections: int = 64):
    """boto3 S3 client with a connection pool sized for threaded range requests.

    Parameters
    ----------
    anonymous:
        Unsigned requests (public open-data bucket) when ``True``.
    max_connections:
        Size of the HTTP connection pool.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    cfg = Config(
        max_pool_connections=max_connections,
        retries={"max_attempts": 10, "mode": "standard"},
        **({"signature_version": UNSIGNED} if anonymous else {}),
    )
    return boto3.client("s3", config=cfg)


def _process_chunk(
    s3,
    s3_video_uri: str,
    camera: str,
    csv_key: str,
    sample_hz: float,
    mode: str,
    io_threads: int,
    decode_threads: int,
    cache: Path | None,
) -> tuple[pd.DataFrame, str]:
    """Sampled luminance + Harp time for one video chunk."""
    bucket, _ = ingestion._parse_s3_uri(s3_video_uri)
    stem = Path(csv_key).stem
    _, timestamp, _ = ingestion.parse_camera_and_timestamp(Path(csv_key).name)
    harp = ingestion.load_harp_seconds(s3_video_uri, camera, timestamp, s3_client=s3)

    cache_file = (
        cache / f"{stem}_{mode}_{sample_hz:g}hz.npz" if cache is not None else None
    )
    if cache_file is not None and cache_file.exists():
        z = np.load(cache_file)
        frames, lum, used_mode = z["frames"], z["lum"], str(z["mode"])
    else:
        frames, lum, plan = video_luminance_s3(
            s3, bucket, csv_key[: -len(".csv")] + ".mp4",
            sample_hz, mode, io_threads, decode_threads,
        )
        used_mode = plan.mode
        if cache_file is not None:
            np.savez(cache_file, frames=frames, lum=lum, mode=used_mode)

    keep = frames < len(harp)
    if not keep.all():
        warnings.warn(
            f"{stem}: video has frames beyond the {len(harp)} CSV rows; dropping them.",
            stacklevel=2,
        )
        frames, lum = frames[keep], lum[keep]
    df = pd.DataFrame(
        {
            "source_file": stem,
            "frame": frames,
            "harp_time": harp[frames],
            "luminance": lum,
        }
    )
    return df, used_mode


def select_shard(keys: Sequence[str], shard_index: int, num_shards: int) -> list[str]:
    """Return shard *shard_index* of *num_shards* contiguous, near-equal slices.

    Contiguous slices keep each shard's output a continuous time span. A shard
    can be empty when ``num_shards`` exceeds ``len(keys)``.

    Parameters
    ----------
    keys:
        Chronologically sorted chunk keys.
    shard_index:
        Zero-based shard to return.
    num_shards:
        Total number of shards.
    """
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index} of {num_shards}.")
    bounds = np.linspace(0, len(keys), num_shards + 1).round().astype(int)
    return list(keys[bounds[shard_index] : bounds[shard_index + 1]])


def process_chunks(
    s3_video_uri: str,
    camera: str,
    csv_keys: Sequence[str],
    sample_hz: float = 0.25,
    mode: str = "keyframe",
    cache_dir: str | Path | None = None,
    anonymous: bool = True,
    n_workers: int | None = None,
    max_decodes: int | None = None,
    io_threads: int = 8,
    decode_threads: int = 1,
    progress: Callable[..., object] = tqdm,
) -> Iterator[tuple[str, pd.DataFrame, str]]:
    """Process video chunks concurrently, yielding each as it completes.

    Threads fetch keyframes; the heavy decoding runs in separate ``ffmpeg``
    processes. Consume the iterator fully (it owns the thread pool).

    Parameters
    ----------
    s3_video_uri:
        S3 URI of the ``behavior-videos`` directory.
    camera:
        Camera sub-directory name.
    csv_keys:
        Chunk CSV keys to process (see :func:`ingestion.list_camera_csv_keys`).
    sample_hz, mode:
        Sampling rate and mode (see :func:`plan_samples`).
    cache_dir:
        Directory for per-chunk caches (``None`` disables).
    anonymous:
        Unsigned S3 requests.
    n_workers:
        Chunks in flight at once (default: ``max_decodes + 1``). Each holds ~140 MB
        of fetched keyframes (x2-3 while joining), so raising this raises memory.
    max_decodes:
        Maximum simultaneous ffmpeg decodes, i.e. CPU cores used (default: half
        the cores).
    io_threads:
        Parallel range requests per chunk.
    decode_threads:
        ffmpeg threads per decode (keep at 1).
    progress:
        Iterable wrapper for progress reporting.

    Yields
    ------
    csv_key : str
    table : pandas.DataFrame
        Columns ``source_file``, ``frame``, ``harp_time``, ``luminance``.
    used_mode : str
        ``"keyframe"`` or ``"exact"``.
    """
    global _DECODE_SLOTS
    decodes = max_decodes or max(1, (os.cpu_count() or 2) // 2)
    _DECODE_SLOTS = threading.BoundedSemaphore(decodes)
    # Each in-flight chunk holds ~140 MB of fetched keyframes (x2-3 while joining),
    # so keep one chunk per decode slot plus one prefetching. This bounds memory
    # to roughly (decodes + 1) x 400 MB.
    n_workers = n_workers or decodes + 1
    keys = list(csv_keys)
    if not keys:
        return
    s3 = make_s3_client(anonymous, max_connections=n_workers * io_threads + 8)
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)

    warned = False
    with ThreadPoolExecutor(max_workers=min(n_workers, len(keys))) as pool:
        futures = {
            pool.submit(
                _process_chunk, s3, s3_video_uri, camera, k, sample_hz, mode,
                io_threads, decode_threads, cache,
            ): k
            for k in keys
        }
        for fut in progress(
            as_completed(futures), total=len(futures), desc="videos", unit="file"
        ):
            df_part, used = fut.result()
            if used == "exact" and not warned:
                warned = True
                warnings.warn(
                    "Used 'exact' mode: every GOP containing a sample is fetched "
                    "and fully decoded, so this approaches decoding the whole "
                    "video. Use sample_hz <= 1 sample per GOP (~0.24 Hz) for the "
                    "fast keyframe-only path.",
                    stacklevel=2,
                )
            yield futures[fut], df_part, used


# ---------------------------------------------------------------------------
# Per-chunk result files (hand-off between a chunk step and the merge step)
# ---------------------------------------------------------------------------
CHUNK_RESULT_SUFFIX = "_luminance_chunk.npz"


def save_chunk_result(
    save_dir: str | Path,
    table: pd.DataFrame,
    session: str,
    camera: str,
    sample_hz: float,
    mode: str,
) -> Path:
    """Write one chunk's sampled luminance + Harp timestamps to ``save_dir``.

    The file is ``<chunk stem>_luminance_chunk.npz`` and is self-describing
    (session, camera, rate and mode are stored inside), so a merge step needs
    nothing but these files.

    Parameters
    ----------
    save_dir:
        Output directory (created if missing).
    table:
        Output of one chunk from :func:`process_chunks` (single ``source_file``).
    session, camera:
        Session name (e.g. ``854393_2026-08-28_20-34-08``) and camera name.
    sample_hz, mode:
        Sampling parameters used.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stems = table["source_file"].unique()
    if len(stems) != 1:
        raise ValueError(f"Expected one chunk per file, got {list(stems)}.")
    path = save_dir / f"{stems[0]}{CHUNK_RESULT_SUFFIX}"
    np.savez(
        path,
        source_file=str(stems[0]),
        frame=table["frame"].to_numpy(dtype="int64"),
        harp_time=table["harp_time"].to_numpy(dtype="float64"),
        luminance=table["luminance"].to_numpy(dtype="float64"),
        session=session,
        camera=camera,
        sample_hz=float(sample_hz),
        mode=mode,
    )
    return path


def load_chunk_results(paths: Sequence[str | Path]) -> tuple[list[pd.DataFrame], dict]:
    """Load files written by :func:`save_chunk_result`.

    Returns
    -------
    tables : list[pandas.DataFrame]
        One table per file (``source_file``, ``frame``, ``harp_time``,
        ``luminance``).
    meta : dict
        ``session``, ``camera``, ``sample_hz`` and ``mode`` from the first file.
    """
    tables, meta = [], {}
    for path in paths:
        with np.load(path) as z:
            if not meta:
                meta = {
                    "session": str(z["session"]),
                    "camera": str(z["camera"]),
                    "sample_hz": float(z["sample_hz"]),
                    "mode": str(z["mode"]),
                }
            tables.append(
                pd.DataFrame(
                    {
                        "source_file": str(z["source_file"]),
                        "frame": z["frame"],
                        "harp_time": z["harp_time"],
                        "luminance": z["luminance"],
                    }
                )
            )
    return tables, meta


# ---------------------------------------------------------------------------
# Merge: whole-session steps (two-sample average + percentile scaling)
# ---------------------------------------------------------------------------
def combine_chunk_tables(
    parts: Sequence[pd.DataFrame],
    low: float = DEFAULT_LOW_PERCENTILE,
    high: float = DEFAULT_HIGH_PERCENTILE,
    tz: str = ingestion.DEFAULT_TIMEZONE,
    start_harp: float | None = None,
    scaling: str = DEFAULT_SCALING,
) -> pd.DataFrame:
    """Concatenate chunk tables and apply the whole-session normalization.

    *scaling* selects ``"minmax"`` (default, no clipping), ``"absolute"`` or
    ``"percentile"`` (clipped); see :func:`normalized_luminance`.

    Chunks are ordered by Harp time (duplicates of a ``(source_file, frame)`` are
    dropped), then the two-sample average and percentile scaling are applied
    across the full series, so chunk boundaries are handled seamlessly.

    Parameters
    ----------
    parts:
        Chunk tables with ``source_file``, ``frame``, ``harp_time``, ``luminance``.
    low, high:
        Percentiles for relative scaling.
    tz:
        Timezone for ``datetime_pacific``.
    start_harp:
        Experiment start (Harp seconds) for ``time_since_start``. Defaults to the
        earliest Harp time present, which is the experiment start when the first
        chunk is included.

    Returns
    -------
    pandas.DataFrame
        ``source_file``, ``frame``, ``harp_time``, ``time_since_start``,
        ``datetime_pacific``, ``luminance`` (``[0, 1]``), ``luminance_2f`` and
        ``luminance_norm`` (relative-scaled, ``[0, 1]``).
    """
    df = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(["source_file", "frame"])
        .sort_values("harp_time", kind="stable")
        .reset_index(drop=True)
    )
    if start_harp is None:
        start_harp = float(df["harp_time"].min())
    df["time_since_start"] = df["harp_time"] - start_harp
    df["datetime_pacific"] = ingestion.harp_to_datetime(df["harp_time"], tz=tz)
    df["luminance_2f"], df["luminance_norm"] = normalized_luminance(
        df["luminance"].to_numpy(), low, high, scaling
    )
    return df[
        [
            "source_file",
            "frame",
            "harp_time",
            "time_since_start",
            "datetime_pacific",
            "luminance",
            "luminance_2f",
            "luminance_norm",
        ]
    ]


def build_luminance_dataset(
    s3_video_uri: str,
    camera: str = "TopCamera",
    sample_hz: float = 0.25,
    mode: str = "keyframe",
    cache_dir: str | Path | None = None,
    anonymous: bool = True,
    max_files: int | None = None,
    low: float = DEFAULT_LOW_PERCENTILE,
    high: float = DEFAULT_HIGH_PERCENTILE,
    tz: str = ingestion.DEFAULT_TIMEZONE,
    n_workers: int | None = None,
    max_decodes: int | None = None,
    io_threads: int = 8,
    decode_threads: int = 1,
    progress: Callable[..., object] = tqdm,
    scaling: str = DEFAULT_SCALING,
) -> pd.DataFrame:
    """Sampled, normalized luminance for every video chunk, in one process.

    Convenience wrapper: :func:`process_chunks` followed by
    :func:`combine_chunk_tables`. For distributed runs call those two steps
    separately (see ``scripts/luminance_chunks.py`` and
    ``scripts/luminance_merge.py``). Per-chunk results are cached in *cache_dir*
    (keyed by chunk, mode and rate) so an interrupted run resumes.

    Parameters
    ----------
    s3_video_uri:
        S3 URI of the ``behavior-videos`` directory.
    camera:
        Camera sub-directory name.
    sample_hz, mode:
        Sampling rate and mode (see :func:`plan_samples`).
    cache_dir:
        Directory for per-chunk caches (``None`` disables).
    anonymous:
        Unsigned S3 requests (public bucket).
    max_files:
        Process only the first N chunks. Percentile scaling is then relative to
        those chunks only.
    low, high:
        Percentiles for relative scaling.
    tz:
        Timezone for ``datetime_pacific``.
    n_workers, max_decodes, io_threads, decode_threads, progress:
        See :func:`process_chunks`.

    Returns
    -------
    pandas.DataFrame
        See :func:`combine_chunk_tables`.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found for the camera.
    """
    s3 = make_s3_client(anonymous, max_connections=8)
    csv_keys = ingestion.list_camera_csv_keys(s3_video_uri, camera, s3_client=s3)
    if not csv_keys:
        raise FileNotFoundError(
            f"No CSV files found for camera '{camera}' under {s3_video_uri!r}."
        )
    if max_files is not None:
        csv_keys = csv_keys[:max_files]
    start_harp = ingestion.get_experiment_start_harp(
        s3_video_uri, camera, s3_client=s3, anonymous=anonymous
    )
    parts = [
        table
        for _, table, _ in process_chunks(
            s3_video_uri, camera, csv_keys, sample_hz, mode, cache_dir, anonymous,
            n_workers, max_decodes, io_threads, decode_threads, progress,
        )
    ]
    return combine_chunk_tables(parts, low, high, tz, start_harp, scaling)


# ---------------------------------------------------------------------------
# Outputs: .npy, plot, table
# ---------------------------------------------------------------------------
DEFAULT_GAP_FACTOR = 10.0  # a gap is a sample spacing > this x the median spacing


def find_gaps(
    harp_time: np.ndarray, gap_seconds: float | None = None
) -> np.ndarray:
    """Indices ``i`` where the time from sample ``i`` to ``i + 1`` is a gap.

    A gap is a spacing larger than *gap_seconds* (default: ``DEFAULT_GAP_FACTOR``
    x the median spacing, ~42 s for 0.24 Hz sampling).
    """
    dt = np.diff(np.asarray(harp_time, dtype="float64"))
    if dt.size == 0:
        return np.empty(0, dtype=int)
    if gap_seconds is None:
        gap_seconds = DEFAULT_GAP_FACTOR * float(np.median(dt))
    return np.flatnonzero(dt > gap_seconds)


def plot_luminance(
    df: pd.DataFrame,
    path: str | Path,
    show: bool = False,
    shade_gaps: bool = True,
    gap_seconds: float | None = None,
) -> None:
    """Black normalized-luminance trace vs. Pacific wall-clock time; saved to *path*.

    With *shade_gaps* (default) every gap in the recording (see :func:`find_gaps`)
    is shaded salmon and the line is broken there instead of drawn straight across.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    tz = df["datetime_pacific"].dt.tz  # America/Los_Angeles (DST-aware)
    times = pd.DatetimeIndex(df["datetime_pacific"]).to_pydatetime()
    y = df["luminance_norm"].to_numpy(dtype="float64")
    gaps = find_gaps(df["harp_time"].to_numpy(), gap_seconds) if shade_gaps else []
    fig, ax = plt.subplots(figsize=(12, 4))
    if len(gaps):
        # NaN after each pre-gap sample breaks the line across the gap.
        x_line = np.insert(times, gaps + 1, times[gaps])
        y_line = np.insert(y, gaps + 1, np.nan)
        for i in gaps:  # lw keeps very short gaps visible as a thin gray line
            ax.axvspan(times[i], times[i + 1], facecolor="salmon", edgecolor="salmon", alpha=0.5,
                       lw=1.0, zorder=0)
    else:
        x_line, y_line = times, y
    ax.plot(x_line, y_line, color="black", lw=0.6)
    locator = mdates.AutoDateLocator(tz=tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=tz))
    ax.set_xlabel(f"Time ({tz})")
    ax.set_ylabel("Normalized luminance")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    save_dir: str | Path,
    stem: str,
    output_format: str = "parquet",
    show: bool = False,
    shade_gaps: bool = True,
    gap_seconds: float | None = None,
) -> dict[str, Path]:
    """Write the ``.npy``, plot and full table for a combined luminance table.

    * ``<stem>.npy`` — float64 ``(N, 2)``: column 0 Harp time (s), column 1
      normalized luminance. float64 is required: Harp seconds are ~3.9e9, which
      float32 cannot resolve to sub-second precision.
    * ``<stem>.png`` — black trace vs. Pacific time.
    * ``<stem>.parquet`` / ``.csv`` — every column.

    Parameters
    ----------
    df:
        Output of :func:`combine_chunk_tables`.
    save_dir:
        Output directory (created if missing).
    stem:
        File name stem (no extension).
    output_format:
        ``"parquet"`` or ``"csv"`` for the table.
    show:
        Also display the plot.
    shade_gaps, gap_seconds:
        Shade recording gaps gray in the plot; see :func:`plot_luminance`.

    Returns
    -------
    dict
        Paths keyed ``"npy"``, ``"table"``, ``"png"``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    npy = save_dir / f"{stem}.npy"
    np.save(npy, df[["harp_time", "luminance_norm"]].to_numpy(dtype="float64"))
    table = save_dir / f"{stem}.{output_format}"
    if output_format == "parquet":
        df.to_parquet(table, index=False)
    else:
        df.to_csv(table, index=False)
    png = save_dir / f"{stem}.png"
    plot_luminance(df, png, show=show, shade_gaps=shade_gaps,
                   gap_seconds=gap_seconds)
    return {"npy": npy, "table": table, "png": png}
